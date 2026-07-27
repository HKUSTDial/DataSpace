from __future__ import annotations

import json
from pathlib import Path

from ..config import HarnessConfig, HarnessExperimentConfig
from ..runners.filesystem_jail import JAIL_HOME, JAIL_OUTPUT, JAIL_WORKSPACE
from .native_common import (
    BASELINE_ROOT,
    clean_environment,
    configured_runtime_binds,
    jailed_invocation,
    native_runner,
    prompt_for,
    write_text,
)

_AUTO_COMPACT_PERCENT = 70
_MAX_IMAGES_PER_REQUEST = 5
_RELAY_PATH = "/mnt/runtime/dataspace-relays/anthropic_messages.py"

_MODEL_ENVIRONMENT_KEYS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "CLAUDE_CONTEXT_COLLAPSE_MODEL",
    "CLAUDE_CODE_BG_CLASSIFIER_MODEL",
    "CLAUDE_CODE_AUTO_MODE_MODEL",
)


def _auto_compact_window(context_window: int, max_output_tokens: int) -> int:
    reserved = max_output_tokens + 8192
    window = context_window - reserved
    if window <= 0:
        raise ValueError(
            "Claude Code context window must exceed max output tokens "
            "plus an 8192-token compaction reserve"
        )
    return window


def _single_backbone_environment(
    model: str,
    *,
    context_window: int,
    max_output_tokens: int,
) -> dict[str, str]:
    auto_compact_window = _auto_compact_window(
        context_window, max_output_tokens
    )
    environment = {key: model for key in _MODEL_ENVIRONMENT_KEYS}
    environment.update(
        {
            "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
            "CLAUDE_CODE_DISABLE_FAST_MODE": "1",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context_window),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output_tokens),
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(auto_compact_window),
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(
                _AUTO_COMPACT_PERCENT
            ),
        }
    )
    return environment


def _exit_classification(
    run_root: Path, _return_code: int
) -> tuple[str, str | None] | None:
    path = run_root / "stdout.log"
    if not path.is_file():
        return None
    result_text = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result_text = str(event.get("result", ""))
    if "output token maximum" in result_text:
        return "model_output_exhausted", result_text
    normalized_result = result_text.lower()
    if (
        "maximum context length" in normalized_result
        or "context window" in normalized_result
        or "prompt is too long" in normalized_result
        or "autocompact is thrashing" in normalized_result
    ):
        return "context_window_exhausted", result_text
    if result_text.startswith("API Error:"):
        return "runtime_error", result_text
    return None


def _validate_single_backbone(run_root: Path, expected_model: str) -> None:
    """Reject a completed trace that exposes any non-experiment model."""

    path = run_root / "stdout.log"
    if not path.is_file():
        return
    unexpected: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message")
            if isinstance(message, dict):
                model = message.get("model")
                if (
                    isinstance(model, str)
                    and model not in {expected_model, "<synthetic>"}
                ):
                    unexpected.add(model)
            model_usage = event.get("modelUsage")
            if isinstance(model_usage, dict):
                for model in model_usage:
                    if model not in {expected_model, f"{expected_model}[1m]"}:
                        unexpected.add(str(model))
    if unexpected:
        raise RuntimeError(
            "Claude Code used model(s) outside the single-backbone policy: "
            f"{', '.join(sorted(unexpected))}; expected {expected_model}"
        )


def _require_nested_sandbox(config: HarnessExperimentConfig) -> None:
    rootfs = config.workbench.rootfs_path
    bwrap_candidates = (rootfs / "usr/bin/bwrap", rootfs / "bin/bwrap")
    socat_candidates = (rootfs / "usr/bin/socat", rootfs / "bin/socat")
    has_bwrap = any(
        path.is_file() and path.stat().st_mode & 0o111 for path in bwrap_candidates
    )
    has_socat = any(
        path.is_file() and path.stat().st_mode & 0o111 for path in socat_candidates
    )
    if has_bwrap and has_socat:
        return
    raise RuntimeError(
        "Claude Code requires bubblewrap and socat inside the Data Workbench rootfs; "
        "rebuild and export the shared runtime with "
        "scripts/build_harness_images.sh --rebuild"
    )


def _trace(run_root: Path) -> tuple[int, int, dict[str, int]]:
    path = run_root / "stdout.log"
    if not path.is_file():
        return 0, 0, {}
    assistant_turns = 0
    tool_actions = 0
    final_turns: int | None = None
    final_usage: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", ""))
            if event_type == "assistant":
                assistant_turns += 1
                message = event.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, list):
                        tool_actions += sum(
                            isinstance(block, dict)
                            and block.get("type") == "tool_use"
                            for block in content
                        )
            elif event_type == "result":
                raw_turns = event.get("num_turns")
                if isinstance(raw_turns, int):
                    final_turns = raw_turns
                raw_usage = event.get("usage")
                if isinstance(raw_usage, dict):
                    final_usage = {
                        str(key): int(value)
                        for key, value in raw_usage.items()
                        if isinstance(value, int)
                    }
    return final_turns or assistant_turns, tool_actions, final_usage


def build_claude_code(
    config: HarnessExperimentConfig, harness: HarnessConfig
):
    _require_nested_sandbox(config)
    executable = harness.executable or f"{JAIL_HOME}/.local/bin/claude"
    runtime_binds = configured_runtime_binds(harness)
    relay_source = BASELINE_ROOT / "src" / "dataspace_baselines" / "relays"
    runtime_binds += (
        (relay_source.resolve(), "/mnt/runtime/dataspace-relays"),
    )

    def invocation(task, run_root, output_root):
        settings_path = run_root / "runtime" / "home" / ".claude" / "settings.json"
        model_environment = _single_backbone_environment(
            config.model.model,
            context_window=config.model.context_window,
            max_output_tokens=config.model.max_output_tokens,
        )
        settings = {
            "permissions": {
                "allow": ["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
                "deny": ["WebSearch", "WebFetch"],
            },
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "allowUnsandboxedCommands": False,
                "filesystem": {
                    "denyWrite": [JAIL_WORKSPACE],
                    "allowWrite": [JAIL_OUTPUT],
                },
                "network": {"allowedDomains": []},
                "credentials": {
                    "envVars": [
                        {"name": config.model.api_key_env, "mode": "deny"},
                        {"name": "ANTHROPIC_AUTH_TOKEN", "mode": "deny"},
                        {"name": "ANTHROPIC_API_KEY", "mode": "deny"},
                    ]
                },
            },
            "env": {
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                **model_environment,
            },
        }
        write_text(
            settings_path,
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
        )
        claude_command = (
            executable,
            "-p",
            prompt_for(task),
            "--model",
            config.model.model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--no-chrome",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--permission-mode",
            "dontAsk",
            "--disallowedTools",
            "WebSearch",
            "WebFetch",
            "--add-dir",
            JAIL_WORKSPACE,
            "--settings",
            f"{JAIL_HOME}/.claude/settings.json",
        )
        inner = (
            "/usr/local/bin/python3",
            _RELAY_PATH,
            "--upstream-base",
            config.model.anthropic_base_url,
            "--max-images",
            str(_MAX_IMAGES_PER_REQUEST),
            "--event-log",
            f"{JAIL_HOME}/.claude/dataspace-relay.jsonl",
            "--",
            *claude_command,
        )
        environment = clean_environment(config)
        api_key = environment[config.model.api_key_env]
        environment.update(
            {
                "ANTHROPIC_BASE_URL": config.model.anthropic_base_url,
                "ANTHROPIC_AUTH_TOKEN": api_key,
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                **model_environment,
            }
        )
        return jailed_invocation(
            config=config,
            task=task,
            run_root=run_root,
            output_root=output_root,
            inner_command=inner,
            environment=environment,
            metadata={
                "web_tools": "disabled",
                "command_network": "disabled",
                "command_credential_env": "filtered",
                "sandbox_fail_closed": True,
                "model_policy": "single-backbone",
                "model_roles": list(_MODEL_ENVIRONMENT_KEYS),
                "model_fallback": "disabled",
                "context_window": config.model.context_window,
                "max_output_tokens": config.model.max_output_tokens,
                "auto_compact_window": int(
                    model_environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]
                ),
                "auto_compact_percent": _AUTO_COMPACT_PERCENT,
                "image_history_policy": {
                    "max_images_per_request": _MAX_IMAGES_PER_REQUEST,
                    "retention": "most_recent",
                    "older_images": "text_placeholder",
                },
                "runtime_version": str(harness.options.get("version", "")),
            },
            runtime_binds=runtime_binds,
        )

    return native_runner(
        harness_key="claude-code",
        config=config,
        harness=harness,
        invocation_factory=invocation,
        trace_parser=_trace,
        exit_classifier=_exit_classification,
        run_validator=lambda run_root: _validate_single_backbone(
            run_root, config.model.model
        ),
        execution_policy={
            "version": "single-backbone-v4",
            "model": config.model.model,
            "roles": list(_MODEL_ENVIRONMENT_KEYS),
            "fallback": "disabled",
            "context_window": config.model.context_window,
            "max_output_tokens": config.model.max_output_tokens,
            "auto_compact_window": _auto_compact_window(
                config.model.context_window,
                config.model.max_output_tokens,
            ),
            "auto_compact_percent": _AUTO_COMPACT_PERCENT,
            "image_history_policy": {
                "max_images_per_request": _MAX_IMAGES_PER_REQUEST,
                "retention": "most_recent",
                "older_images": "text_placeholder",
            },
        },
    )
