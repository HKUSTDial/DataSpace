from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import HarnessConfig, HarnessExperimentConfig
from ..core.prompt import PROMPT_CONTRACT_VERSION, relation_task_prompt
from ..core.task import TaskSpec
from ..runners.filesystem_jail import (
    JAIL_HOME,
    JAIL_OUTPUT,
    JAIL_WORKSPACE,
    BubblewrapJail,
    minimal_harness_environment,
)
from ..runners.native_cli import NativeCliInvocation, NativeCliRunner
from ..workbench import runtime_identity, runtime_metadata


BASELINE_ROOT = Path(__file__).resolve().parents[3]


def configured_runtime_binds(
    harness: HarnessConfig,
) -> tuple[tuple[Path, str], ...]:
    path_value = harness.options.get("runtime_path")
    mount_value = harness.options.get("runtime_mount")
    if path_value is None and mount_value is None:
        return ()
    if path_value is None or mount_value is None:
        raise ValueError("runtime_path and runtime_mount must be set together")
    version = str(harness.options.get("version", ""))
    source = Path(str(path_value).replace("{version}", version))
    if not source.is_absolute():
        source = BASELINE_ROOT / source
    return ((source.resolve(), str(mount_value)),)


def prompt_for(
    task: TaskSpec,
    *,
    workspace_path: str = JAIL_WORKSPACE,
    output_path: str = f"{JAIL_OUTPUT}/prediction.csv",
) -> str:
    return relation_task_prompt(
        task,
        workspace_path=workspace_path,
        output_path=output_path,
    )


def clean_environment(config: HarnessExperimentConfig) -> dict[str, str]:
    return minimal_harness_environment(config.model.api_key_env)


def jailed_invocation(
    *,
    config: HarnessExperimentConfig,
    task: TaskSpec,
    run_root: Path,
    output_root: Path,
    inner_command: tuple[str, ...],
    environment: dict[str, str],
    metadata: dict[str, Any],
    working_directory: str = JAIL_OUTPUT,
    runtime_binds: tuple[tuple[Path, str], ...] = (),
) -> NativeCliInvocation:
    home_root = run_root / "runtime" / "home"
    command = BubblewrapJail().command(
        task_context=task.context_dir,
        output_root=output_root,
        home_root=home_root,
        inner_command=inner_command,
        working_directory=working_directory,
        runtime_binds=runtime_binds,
        rootfs_root=config.workbench.rootfs_path,
    )
    invocation_metadata = dict(metadata)
    invocation_metadata["data_workbench"] = runtime_metadata(
        config.workbench
    )
    return NativeCliInvocation(
        command=command,
        cwd=run_root,
        environment=environment,
        metadata=invocation_metadata,
        inherit_environment=False,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def jsonl_trace_summary(
    path: Path,
    *,
    turn_events: set[str],
    action_item_types: set[str],
) -> tuple[int, int, dict[str, int]]:
    if not path.is_file():
        return 0, 0, {}
    turns = 0
    actions = 0
    seen_actions: set[tuple[str, str]] = set()
    usage: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", ""))
            if event_type in turn_events:
                turns += 1
            item = event.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                if item_type in action_item_types:
                    item_id = item.get("id")
                    if isinstance(item_id, (str, int)) and not isinstance(
                        item_id, bool
                    ):
                        action_key = (item_type, str(item_id))
                        if action_key not in seen_actions:
                            seen_actions.add(action_key)
                            actions += 1
                    elif event_type == "item.completed":
                        # Codex normally supplies stable item IDs. For a
                        # compatible stream without IDs, count only the
                        # terminal event so started/completed pairs are not
                        # double-counted.
                        actions += 1
            candidates = [event.get("usage")]
            result = event.get("result")
            if isinstance(result, dict):
                candidates.append(result.get("usage"))
            message = event.get("message")
            if isinstance(message, dict):
                candidates.append(message.get("usage"))
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                for key, value in candidate.items():
                    if isinstance(value, int):
                        usage[str(key)] = usage.get(str(key), 0) + value
    return turns, actions, usage


def native_runner(
    *,
    harness_key: str,
    config: HarnessExperimentConfig,
    harness: HarnessConfig,
    invocation_factory,
    trace_parser,
    run_validator=None,
    exit_classifier=None,
    cleanup=None,
    execution_policy=None,
) -> NativeCliRunner:
    resume_identity = {
        "runner": harness_key,
        "model": {
            "key": config.model.key,
            "model": config.model.model,
            "openai_base_url": config.model.openai_base_url,
            "anthropic_base_url": config.model.anthropic_base_url,
            "max_output_tokens": config.model.max_output_tokens,
            "context_window": config.model.context_window,
        },
        "wall_time_seconds": config.wall_time_seconds,
        "workbench": runtime_identity(config.workbench),
        "harness": {
            "executable": harness.executable,
            "options": harness.options,
        },
        "prompt_contract": PROMPT_CONTRACT_VERSION,
    }
    if execution_policy is not None:
        resume_identity["execution_policy"] = dict(execution_policy)
    return NativeCliRunner(
        harness_key=harness_key,
        model_key=config.model.key,
        wall_time_seconds=config.wall_time_seconds,
        invocation_factory=invocation_factory,
        trace_parser=trace_parser,
        run_validator=run_validator,
        exit_classifier=exit_classifier,
        cleanup=cleanup,
        resume_identity=resume_identity,
    )
