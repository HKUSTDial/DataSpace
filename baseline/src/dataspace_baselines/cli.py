from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_harness_experiment_config
from .core.batch import BatchOptions, result_payload, run_batch
from .core.task import TaskSpec
from .doctor import run_doctor
from .harnesses import available_harnesses, build_harness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataspace-baselines",
        description="Run the controlled DataSpace harness comparison.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list supported harnesses")
    list_parser.add_argument("--config", type=Path)

    doctor_parser = subparsers.add_parser(
        "doctor", help="validate local harness dependencies and isolation runtimes"
    )
    doctor_parser.add_argument("--config", required=True, type=Path)

    run_parser = subparsers.add_parser("run", help="run one task")
    run_parser.add_argument("--harness", required=True, choices=available_harnesses())
    run_parser.add_argument("--task-dir", required=True, type=Path)
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--run-dir", required=True, type=Path)

    batch_parser = subparsers.add_parser("batch", help="run a full benchmark")
    batch_parser.add_argument(
        "--harness", required=True, choices=available_harnesses()
    )
    batch_parser.add_argument("--input-root", required=True, type=Path)
    batch_parser.add_argument("--config", required=True, type=Path)
    batch_parser.add_argument("--run-root", required=True, type=Path)
    batch_parser.add_argument("--concurrency", type=int)
    batch_parser.add_argument("--resume", action="store_true")
    return parser


def _list(args: argparse.Namespace) -> int:
    if args.config is None:
        for key in available_harnesses():
            print(key)
        return 0
    config = load_harness_experiment_config(args.config)
    for key in available_harnesses():
        harness = config.harnesses.get(key)
        state = "enabled" if harness is not None and harness.enabled else "disabled"
        print(f"{key}\t{state}")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError(
            f"run directory already exists and is not empty: {args.run_dir}"
        )
    config = load_harness_experiment_config(args.config)
    runner = build_harness(args.harness, config)
    task = TaskSpec.load(args.task_dir)
    result = runner.run(task, args.run_dir)
    print(json.dumps(result_payload(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "submitted" else 1


def _doctor(args: argparse.Namespace) -> int:
    config = load_harness_experiment_config(args.config)
    checks = run_doctor(config)
    for check in checks:
        print(f"[{'ok' if check.ok else 'fail'}] {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


def _batch(args: argparse.Namespace) -> int:
    config = load_harness_experiment_config(args.config)
    runner = build_harness(args.harness, config)
    return run_batch(
        BatchOptions(
            input_root=args.input_root,
            run_root=args.run_root,
            concurrency=(
                args.concurrency
                if args.concurrency is not None
                else config.concurrency
            ),
            resume=args.resume,
            model_key=config.model.key,
            harness_key=args.harness,
        ),
        runner,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            return _list(args)
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "run":
            return _run(args)
        if args.command == "batch":
            return _batch(args)
        parser.error(f"unsupported command: {args.command}")
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
