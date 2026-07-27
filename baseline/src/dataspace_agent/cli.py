from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dataspace_baselines.core.batch import (
    BatchOptions,
    finalized_attempt as _finalized_attempt,
    next_attempt as _next_attempt,
    result_payload as _result_payload,
    run_batch,
    task_directories as _task_directories,
)

from .agent import DataSpaceAgent
from .backend import OpenAICompatibleBackend
from .config import load_run_config
from .sandbox import DockerSandbox, docker_image_metadata
from .task import TaskSpec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataspace-agent",
        description="Run the DataSpace-Agent baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one benchmark task")
    run_parser.add_argument("--task-dir", required=True, type=Path)
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--model", required=True, dest="model_key")
    run_parser.add_argument("--run-dir", required=True, type=Path)

    batch_parser = subparsers.add_parser(
        "batch", help="run every task under a benchmark input directory"
    )
    batch_parser.add_argument("--input-root", required=True, type=Path)
    batch_parser.add_argument("--config", required=True, type=Path)
    batch_parser.add_argument("--model", required=True, dest="model_key")
    batch_parser.add_argument("--run-root", required=True, type=Path)
    batch_parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="number of tasks to run concurrently (default: 8)",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="skip tasks that already have a finalized run",
    )
    return parser


def _build_agent(config: Any) -> DataSpaceAgent:
    backend = OpenAICompatibleBackend(config.model)

    def sandbox_factory(workspace: Path, output: Path) -> DockerSandbox:
        return DockerSandbox(
            workspace_root=workspace,
            output_root=output,
            config=config.sandbox,
            max_output_chars=config.agent.max_tool_output_chars,
        )

    return DataSpaceAgent(
        backend=backend,
        agent_config=config.agent,
        sandbox_factory=sandbox_factory,
        run_metadata={
            "model": {
                "key": config.model.key,
                "requested_model": config.model.model,
                "base_url": config.model.base_url,
                "max_output_tokens": config.model.max_output_tokens,
                "temperature": config.model.temperature,
                "extra_body": config.model.extra_body,
                "api_key_env": config.model.api_key_env,
            },
            "sandbox": asdict(config.sandbox),
            "data_workbench": docker_image_metadata(config.sandbox.image),
        },
    )


def _run(args: argparse.Namespace) -> int:
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError(
            f"run directory already exists and is not empty: {args.run_dir}"
        )
    config = load_run_config(args.config, args.model_key)
    task = TaskSpec.load(args.task_dir)
    agent = _build_agent(config)
    result = agent.run(task, args.run_dir)
    print(json.dumps(_result_payload(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "submitted" else 1


def _run_batch(args: argparse.Namespace) -> int:
    config = load_run_config(args.config, args.model_key)
    agent = _build_agent(config)
    return run_batch(
        BatchOptions(
            input_root=args.input_root,
            run_root=args.run_root,
            concurrency=args.concurrency,
            resume=args.resume,
            model_key=args.model_key,
        ),
        agent,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "batch":
            return _run_batch(args)
        parser.error(f"unsupported command: {args.command}")
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
