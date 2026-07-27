from __future__ import annotations

import json
from pathlib import Path

from ..config import HarnessConfig, HarnessExperimentConfig
from ..runners.filesystem_jail import JAIL_HOME, JAIL_OUTPUT
from .native_common import (
    clean_environment,
    configured_runtime_binds,
    jailed_invocation,
    jsonl_trace_summary,
    native_runner,
    prompt_for,
    write_text,
)


def _trace(run_root: Path) -> tuple[int, int, dict[str, int]]:
    return jsonl_trace_summary(
        run_root / "stdout.log",
        turn_events={"turn.completed"},
        action_item_types={
            "command_execution",
            "file_change",
            "mcp_tool_call",
            "web_search",
        },
    )


def _exit_classification(
    run_root: Path, _return_code: int
) -> tuple[str, str | None] | None:
    path = run_root / "stdout.log"
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    if (
        "reason: max_output_tokens" in content
        or "reason=max_output_tokens" in content
    ):
        return (
            "model_output_exhausted",
            "Codex model response was truncated by max_output_tokens",
        )
    return None


def build_codex(
    config: HarnessExperimentConfig, harness: HarnessConfig
):
    executable = harness.executable or f"{JAIL_HOME}/.local/node/bin/codex"
    wire_api = str(harness.options.get("wire_api", "responses")).strip().lower()
    if wire_api not in {"chat", "responses"}:
        raise ValueError("Codex wire_api must be 'chat' or 'responses'")
    runtime_binds = configured_runtime_binds(harness)
    auto_compact_limit = (
        config.model.context_window - config.model.max_output_tokens - 8192
    )
    if auto_compact_limit <= 0:
        raise ValueError(
            "Codex context window must exceed max output tokens plus an "
            "8192-token compaction reserve"
        )

    def invocation(task, run_root, output_root):
        codex_home = run_root / "runtime" / "home" / ".codex"
        model = json.dumps(config.model.model)
        base_url = json.dumps(config.model.openai_base_url)
        api_key_env = json.dumps(config.model.api_key_env)
        provider_options = f'wire_api = {json.dumps(wire_api)}'
        if wire_api == "responses":
            provider_options += "\nsupports_websockets = false"
        content = f"""model = {model}
model_provider = "vercel"
model_context_window = {config.model.context_window}
model_auto_compact_token_limit = {auto_compact_limit}
approval_policy = "never"
sandbox_mode = "workspace-write"
web_search = "disabled"
check_for_update_on_startup = false
allow_login_shell = false

[model_providers.vercel]
name = "Vercel AI Gateway"
base_url = {base_url}
env_key = {api_key_env}
{provider_options}

[sandbox_workspace_write]
network_access = false
writable_roots = [{json.dumps(JAIL_OUTPUT)}]

[shell_environment_policy]
inherit = "core"
exclude = [{json.dumps(config.model.api_key_env)}, "ANTHROPIC_*", "*TOKEN*", "*SECRET*"]

[features]
apps = false
remote_plugin = false
enable_request_compression = false

[history]
persistence = "none"
"""
        write_text(codex_home / "config.toml", content)
        inner_parts = [
            executable,
            "exec",
        ]
        if wire_api == "responses":
            inner_parts.append("--strict-config")
        inner_parts.append("--json")
        if wire_api == "responses":
            inner_parts.append("--ephemeral")
        inner_parts.append("--skip-git-repo-check")
        if wire_api == "responses":
            inner_parts.append("--ignore-rules")
        inner_parts.extend(
            (
                "--color",
                "never",
                "-C",
                JAIL_OUTPUT,
                "--model",
                config.model.model,
                prompt_for(task),
            )
        )
        inner = tuple(inner_parts)
        environment = clean_environment(config)
        environment["CODEX_HOME"] = f"{JAIL_HOME}/.codex"
        return jailed_invocation(
            config=config,
            task=task,
            run_root=run_root,
            output_root=output_root,
            inner_command=inner,
            environment=environment,
            metadata={
                "web_search": "disabled",
                "command_network": "disabled",
                "command_credential_env": "filtered",
                "request_compression": "disabled",
                "session": "task-local",
                "context_window": config.model.context_window,
                "auto_compact_token_limit": auto_compact_limit,
                "wire_api": "chat_completions"
                if wire_api == "chat"
                else "responses",
                "runtime_version": str(harness.options.get("version", "")),
            },
            runtime_binds=runtime_binds,
        )

    return native_runner(
        harness_key="codex",
        config=config,
        harness=harness,
        invocation_factory=invocation,
        trace_parser=_trace,
        exit_classifier=_exit_classification,
        execution_policy={
            "version": "native-context-v2",
            "context_window": config.model.context_window,
            "max_output_tokens": config.model.max_output_tokens,
            "auto_compact_token_limit": auto_compact_limit,
        },
    )
