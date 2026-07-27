from __future__ import annotations

import json
from pathlib import Path

from ..config import HarnessConfig, HarnessExperimentConfig
from ..relays.openai_chat import RELAY_BASE_PLACEHOLDER
from ..runners.filesystem_jail import JAIL_HOME
from .native_common import (
    BASELINE_ROOT,
    clean_environment,
    configured_runtime_binds,
    jailed_invocation,
    native_runner,
    prompt_for,
    write_text,
)


_MODEL_ROLES = ("default", "image_description", "session_summary", "web_search")
_API_BACKENDS = frozenset({"chat_completions", "responses", "messages"})
_SANDBOX_PROFILE = "dataspace-strict"
_TOOLS_MANIFEST = "/opt/dataspace/TOOLS.md"


def _toml_string_assignments(content: str, section_name: str) -> dict[str, str]:
    section = ""
    assignments: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != section_name or "=" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split("=", 1))
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            assignments[key] = value
    return assignments


def _validate_single_backbone(
    run_root: Path,
    expected_model: str,
    expected_api_backend: str = "chat_completions",
) -> None:
    config_path = run_root / "runtime" / "home" / ".grok" / "config.toml"
    if not config_path.is_file():
        raise RuntimeError("Grok Build task-local config is missing")
    content = config_path.read_text(encoding="utf-8", errors="replace")
    roles = _toml_string_assignments(content, "models")
    missing_roles = [
        role for role in _MODEL_ROLES if roles.get(role) != "dataspace-backbone"
    ]
    model_config = _toml_string_assignments(
        content, "model.dataspace-backbone"
    )
    model = model_config.get("model")
    api_backend = model_config.get("api_backend")
    if (
        missing_roles
        or model != expected_model
        or api_backend != expected_api_backend
    ):
        details = []
        if missing_roles:
            details.append(f"unmapped roles: {', '.join(missing_roles)}")
        if model != expected_model:
            details.append(f"configured model: {model!r}")
        if api_backend != expected_api_backend:
            details.append(f"API backend: {api_backend!r}")
        raise RuntimeError(
            "invalid Grok Build single-backbone config ("
            + "; ".join(details)
            + ")"
        )

    unexpected: set[str] = set()
    runtime_log = (
        run_root / "runtime" / "home" / ".grok" / "logs" / "unified.jsonl"
    )
    if runtime_log.is_file():
        with runtime_log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                message = str(event.get("msg", ""))
                context = event.get("ctx")
                if not isinstance(context, dict):
                    continue
                candidates: list[object] = []
                if message == "backend_search: model switch":
                    candidates.append(context.get("new_model"))
                elif message.startswith("subagent read parent config"):
                    candidates.append(context.get("parent_model"))
                elif message == "subagent model resolved":
                    candidates.append(context.get("parent_model"))
                elif message == "subagent spawn credentials":
                    candidates.append(context.get("effective_model_raw"))
                unexpected.update(
                    str(candidate)
                    for candidate in candidates
                    if isinstance(candidate, str) and candidate != expected_model
                )
    if unexpected:
        raise RuntimeError(
            "Grok Build used model(s) outside the single-backbone policy: "
            f"{', '.join(sorted(unexpected))}; expected {expected_model}"
        )

    sandbox_events = (
        run_root / "runtime" / "home" / ".grok" / "sandbox-events.jsonl"
    )
    if sandbox_events.is_file():
        profile_applied = False
        tools_violation = False
        with sandbox_events.open(
            "r", encoding="utf-8", errors="replace"
        ) as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                read_only_paths = event.get("read_only_paths")
                if (
                    event.get("event_type") == "ProfileApplied"
                    and event.get("profile") == _SANDBOX_PROFILE
                    and event.get("enforced") is True
                    and isinstance(read_only_paths, list)
                    and "/opt/dataspace" in read_only_paths
                ):
                    profile_applied = True
                if (
                    event.get("event_type") == "FsViolation"
                    and event.get("target") == _TOOLS_MANIFEST
                ):
                    tools_violation = True
        if not profile_applied:
            raise RuntimeError(
                "Grok Build did not enforce the DataSpace sandbox profile"
            )
        if tools_violation:
            raise RuntimeError(
                "Grok Build sandbox denied the Data Workbench tools manifest"
            )


def _exit_classification(
    run_root: Path, _return_code: int
) -> tuple[str, str | None] | None:
    fragments: list[str] = []
    for name in ("stderr.log", "stdout.log"):
        path = run_root / name
        if path.is_file():
            fragments.append(
                path.read_text(encoding="utf-8", errors="replace")
            )
    message = "\n".join(fragments)
    if (
        "max_tokens_truncation" in message
        or "response truncated by max_tokens" in message
    ):
        return (
            "model_output_exhausted",
            "Grok Build model response was truncated by max_tokens",
        )
    if "serialization error" in message:
        return "runtime_error", "Grok Build provider protocol error"
    return None

def _trace(run_root: Path) -> tuple[int, int, dict[str, int]]:
    """Summarize Grok's headless trace without rewriting its raw events.

    In ``streaming-json`` mode Grok emits response chunks followed by one
    ``end`` event. Tool executions are intentionally absent from stdout and
    are recorded in the task-local unified runtime log instead.
    """

    turns: int | None = None
    usage: dict[str, int] = {}
    stdout_path = run_root / "stdout.log"
    if stdout_path.is_file():
        with stdout_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("type") not in {
                    "end",
                    "error",
                }:
                    continue
                raw_turns = event.get("num_turns")
                if isinstance(raw_turns, int) and not isinstance(raw_turns, bool):
                    turns = raw_turns
                raw_usage = event.get("usage")
                if isinstance(raw_usage, dict):
                    usage = {
                        str(key): value
                        for key, value in raw_usage.items()
                        if isinstance(value, int) and not isinstance(value, bool)
                    }

    tool_actions = 0
    inferred_turns = 0
    runtime_log = (
        run_root / "runtime" / "home" / ".grok" / "logs" / "unified.jsonl"
    )
    if runtime_log.is_file():
        with runtime_log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                message = event.get("msg")
                if message == "shell.tool.exec_done":
                    tool_actions += 1
                elif message == "shell.turn.inference_done":
                    inferred_turns += 1

    return turns if turns is not None else inferred_turns, tool_actions, usage


def build_grok_build(
    config: HarnessExperimentConfig, harness: HarnessConfig
):
    executable = harness.executable or f"{JAIL_HOME}/.grok/bin/grok"
    runtime_binds = configured_runtime_binds(harness)
    api_backend = str(harness.options.get("api_backend", "chat_completions"))
    if api_backend not in _API_BACKENDS:
        raise ValueError(
            "Grok Build api_backend must be one of: "
            + ", ".join(sorted(_API_BACKENDS))
        )
    normalize_finish_reasons = bool(
        harness.options.get("normalize_finish_reasons", False)
    )
    if normalize_finish_reasons and api_backend != "chat_completions":
        raise ValueError(
            "Grok Build finish-reason normalization requires "
            "api_backend=chat_completions"
        )
    if normalize_finish_reasons:
        relay_source = BASELINE_ROOT / "src" / "dataspace_baselines" / "relays"
        runtime_binds += (
            (relay_source.resolve(), "/mnt/runtime/dataspace-relays"),
        )

    def invocation(task, run_root, output_root):
        grok_home = run_root / "runtime" / "home" / ".grok"
        model_base_url = (
            RELAY_BASE_PLACEHOLDER
            if normalize_finish_reasons
            else config.model.openai_base_url
        )
        content = f"""[models]
default = "dataspace-backbone"
image_description = "dataspace-backbone"
session_summary = "dataspace-backbone"
web_search = "dataspace-backbone"

[model.dataspace-backbone]
model = {json.dumps(config.model.model)}
base_url = {json.dumps(model_base_url)}
name = {json.dumps(config.model.key)}
env_key = {json.dumps(config.model.api_key_env)}
api_backend = {json.dumps(api_backend)}
max_completion_tokens = {config.model.max_output_tokens}
context_window = {config.model.context_window}
"""
        write_text(grok_home / "config.toml", content)
        sandbox_content = f"""[profiles.{_SANDBOX_PROFILE}]
extends = "strict"
restrict_network = true
read_only = ["/opt/dataspace"]
"""
        write_text(grok_home / "sandbox.toml", sandbox_content)
        grok_command = (
            executable,
            "-p",
            prompt_for(task),
            "--cwd",
            "/mnt",
            "--model",
            "dataspace-backbone",
            "--output-format",
            "streaming-json",
            "--disable-web-search",
            "--sandbox",
            _SANDBOX_PROFILE,
            "--always-approve",
            "--no-memory",
        )
        if normalize_finish_reasons:
            inner = (
                "/usr/local/bin/python3",
                "/mnt/runtime/dataspace-relays/openai_chat.py",
                "--upstream-base",
                config.model.openai_base_url,
                "--config",
                f"{JAIL_HOME}/.grok/config.toml",
                "--",
                *grok_command,
            )
        else:
            inner = grok_command
        environment = clean_environment(config)
        environment.update(
            {
                "GROK_CONFIG_HOME": f"{JAIL_HOME}/.grok",
                "GROK_HOME": f"{JAIL_HOME}/.grok",
                "GROK_DISABLE_AUTO_UPDATE": "1",
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
                "web_search": "disabled",
                "sandbox": _SANDBOX_PROFILE,
                "sandbox_base": "strict",
                "sandbox_profile": _SANDBOX_PROFILE,
                "tools_manifest_access": "read-only",
                "tool_approval": "automatic",
                "memory": "disabled",
                "model_policy": "single-backbone",
                "model_roles": list(_MODEL_ROLES),
                "api_backend": api_backend,
                "finish_reason_normalization": normalize_finish_reasons,
                "context_window": config.model.context_window,
                "runtime_version": str(harness.options.get("version", "")),
            },
            working_directory="/mnt",
            runtime_binds=runtime_binds,
        )

    return native_runner(
        harness_key="grok-build",
        config=config,
        harness=harness,
        invocation_factory=invocation,
        trace_parser=_trace,
        exit_classifier=_exit_classification,
        run_validator=lambda run_root: _validate_single_backbone(
            run_root, config.model.model, api_backend
        ),
        execution_policy={
            "version": "single-backbone-v5",
            "model": config.model.model,
            "roles": list(_MODEL_ROLES),
            "api_backend": api_backend,
            "finish_reason_normalization": normalize_finish_reasons,
            "sandbox_profile": _SANDBOX_PROFILE,
            "tools_manifest_access": "read-only",
            "context_window": config.model.context_window,
            "max_output_tokens": config.model.max_output_tokens,
        },
    )
