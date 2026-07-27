from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

from ..config import HarnessConfig, HarnessExperimentConfig
from .native_common import clean_environment, native_runner, prompt_for, write_text
from ..runners.native_cli import NativeCliInvocation
from ..workbench import runtime_metadata


def _trace(run_root: Path) -> tuple[int, int, dict[str, int]]:
    path = run_root / "runtime" / "smolagents_trace.json"
    if not path.is_file():
        return 0, 0, {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    steps = payload.get("steps")
    if not isinstance(steps, list):
        steps = []
    turns = sum(
        isinstance(step, dict)
        and ("step_number" in step or "plan" in step)
        for step in steps
    )
    actions = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list):
            actions += len(tool_calls)
        elif step.get("code_action") is not None:
            # Compatibility fallback for traces produced by an older
            # smolagents release without serialized tool_calls.
            actions += 1
    raw_usage = payload.get("usage")
    usage = (
        {
            str(key): int(value)
            for key, value in raw_usage.items()
            if isinstance(value, int)
        }
        if isinstance(raw_usage, dict)
        else {}
    )
    return int(turns), int(actions), usage


def _cleanup(run_root: Path) -> None:
    path = run_root / "runtime" / "smolagents_resources.json"
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        resources = json.load(handle)
    container = str(resources.get("container", "")).strip()
    network = str(resources.get("network", "")).strip()
    if container:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    if network:
        subprocess.run(
            ["docker", "network", "rm", network],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


def build_smolagents(
    config: HarnessExperimentConfig, harness: HarnessConfig
):
    image = str(
        harness.options.get(
            "image", "dataspace-smolagents:1.0"
        )
    )

    def invocation(task, run_root, output_root):
        runtime = run_root / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        prompt_path = runtime / "prompt.txt"
        trace_path = runtime / "smolagents_trace.json"
        write_text(
            prompt_path,
            prompt_for(
                task,
                workspace_path="/workspace",
                output_path="/output/prediction.csv",
            ),
        )
        token = uuid.uuid4().hex[:12]
        container_name = f"dataspace-smolagents-{token}"
        network_name = f"dataspace-smolagents-{token}"
        write_text(
            runtime / "smolagents_resources.json",
            json.dumps(
                {"container": container_name, "network": network_name},
                indent=2,
            )
            + "\n",
        )
        command = (
            sys.executable,
            "-m",
            "dataspace_baselines.harnesses.smolagents_entry",
            "--prompt-file",
            str(prompt_path),
            "--context",
            str(task.context_dir),
            "--output",
            str(output_root),
            "--trace",
            str(trace_path),
            "--model",
            config.model.model,
            "--base-url",
            config.model.openai_base_url,
            "--api-key-env",
            config.model.api_key_env,
            "--max-output-tokens",
            str(config.model.max_output_tokens),
            "--image",
            image,
            "--container-name",
            container_name,
            "--network-name",
            network_name,
        )
        environment = clean_environment(config)
        environment.update(
            {
                "HOME": str(runtime / "home"),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
        return NativeCliInvocation(
            command=command,
            cwd=run_root,
            environment=environment,
            metadata={
                "executor": "official DockerExecutor",
                "container_network": "internal",
                "image": image,
                "data_workbench": runtime_metadata(config.workbench),
            },
            inherit_environment=False,
        )

    return native_runner(
        harness_key="smolagents",
        config=config,
        harness=harness,
        invocation_factory=invocation,
        trace_parser=_trace,
        cleanup=_cleanup,
    )
