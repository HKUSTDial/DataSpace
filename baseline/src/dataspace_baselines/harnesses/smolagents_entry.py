from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any


AUTHORIZED_IMPORTS = [
    "csv",
    "cv2",
    "fitz",
    "glob",
    "json",
    "numpy",
    "openpyxl",
    "os",
    "pandas",
    "pathlib",
    "PIL",
    "pyarrow",
    "pypdf",
    "shutil",
    "sqlite3",
    "subprocess",
    "tempfile",
    "xml",
    "zipfile",
]


class TerminationRequested(SystemExit):
    pass


def _terminate(_signum, _frame) -> None:
    raise TerminationRequested(143)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--max-output-tokens", required=True, type=int)
    parser.add_argument("--image", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--network-name", required=True)
    return parser


def _trace_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = (
        "step_number",
        "timing",
        "tool_calls",
        "error",
        "model_output",
        "code_action",
        "observations",
        "action_output",
        "token_usage",
        "is_final_answer",
        "plan",
    )
    return [
        {key: step.get(key) for key in kept if key in step}
        for step in steps
    ]


def _write_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        handle.write("\n")
    temporary.replace(path)


def _agent_snapshot(agent: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Capture completed steps and usage from an interrupted CodeAgent."""

    memory = getattr(agent, "memory", None)
    raw_steps: list[dict[str, Any]] = []
    if memory is not None:
        get_steps = getattr(memory, "get_succinct_steps", None)
        if callable(get_steps):
            candidate = get_steps()
            if isinstance(candidate, list):
                raw_steps = [step for step in candidate if isinstance(step, dict)]

    usage: dict[str, int] = {}
    monitor = getattr(agent, "monitor", None)
    get_usage = getattr(monitor, "get_total_token_counts", None)
    if callable(get_usage):
        token_usage = get_usage()
        as_dict = getattr(token_usage, "dict", None)
        candidate = as_dict() if callable(as_dict) else token_usage
        if isinstance(candidate, dict):
            usage = {
                str(key): value
                for key, value in candidate.items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
    if "total_tokens" not in usage:
        total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if total:
            usage["total_tokens"] = total
    return _trace_steps(raw_steps), usage


def _container_run_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": args.container_name,
        "network": args.network_name,
        "volumes": {
            str(args.context.resolve()): {
                "bind": "/workspace",
                "mode": "ro",
            },
            str(args.output.resolve()): {
                "bind": "/output",
                "mode": "rw",
            },
        },
        "working_dir": "/output",
        "read_only": True,
        "tmpfs": {
            "/tmp": "rw,nosuid,nodev,size=2g",
        },
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "pids_limit": 256,
        "mem_limit": "16g",
        "nano_cpus": 4_000_000_000,
        "user": f"{os.getuid()}:{os.getgid()}",
        "environment": {
            "HOME": "/tmp",
            "JUPYTER_RUNTIME_DIR": "/tmp/jupyter-runtime",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "labels": {
            "dataspace.harness": "smolagents",
            "dataspace.run": args.container_name,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"environment variable {args.api_key_env} is not set")

    try:
        import docker
        from smolagents import CodeAgent, OpenAIModel
        from smolagents.monitoring import AgentLogger
        from dataspace_baselines.harnesses.smolagents_executor import (
            InternalDockerExecutor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "smolagents with its Docker extra is required for this harness"
        ) from exc

    client = docker.from_env()
    network = None
    agent = None
    executor = None
    run_start_time: float | None = None
    trace_payload: dict[str, Any] = {
        "harness": "smolagents",
        "model": args.model,
        "state": "runtime_error",
    }
    try:
        network = client.networks.create(
            args.network_name,
            driver="bridge",
            internal=True,
            check_duplicate=True,
            labels={"dataspace.harness": "smolagents"},
        )
        logger = AgentLogger()
        executor = InternalDockerExecutor(
            AUTHORIZED_IMPORTS,
            logger=logger,
            image_name=args.image,
            container_run_kwargs=_container_run_kwargs(args),
        )
        model = OpenAIModel(
            model_id=args.model,
            api_base=args.base_url,
            api_key=api_key,
            max_tokens=args.max_output_tokens,
        )
        agent = CodeAgent(
            tools=[],
            model=model,
            executor=executor,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            max_steps=1_000_000,
            return_full_result=True,
            logger=logger,
        )
        prompt = args.prompt_file.read_text(encoding="utf-8")
        run_start_time = time.time()
        result = agent.run(prompt, return_full_result=True)
        usage = result.token_usage.dict() if result.token_usage is not None else {}
        trace_payload = {
            "harness": "smolagents",
            "model": args.model,
            "state": result.state,
            "output": result.output,
            "usage": usage,
            "timing": result.timing.dict(),
            "steps": _trace_steps(result.steps),
        }
        _write_trace(args.trace, trace_payload)
        return 0
    except BaseException as exc:
        trace_payload.update(
            {
                "state": (
                    "terminated"
                    if isinstance(exc, (TerminationRequested, KeyboardInterrupt))
                    else "runtime_error"
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if agent is not None:
            steps, usage = _agent_snapshot(agent)
            trace_payload["steps"] = steps
            trace_payload["usage"] = usage
        if run_start_time is not None:
            end_time = time.time()
            trace_payload["timing"] = {
                "start_time": run_start_time,
                "end_time": end_time,
                "duration": end_time - run_start_time,
            }
        _write_trace(args.trace, trace_payload)
        if not isinstance(exc, (TerminationRequested, KeyboardInterrupt)):
            traceback.print_exc(file=sys.stderr)
        return 143 if isinstance(exc, TerminationRequested) else 1
    finally:
        if agent is not None:
            agent.cleanup()
        elif executor is not None:
            executor.cleanup()
        if network is not None:
            try:
                network.remove()
            except Exception:
                pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
