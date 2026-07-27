from __future__ import annotations

import json
import re
import shutil
import signal
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .interfaces import HarnessRunner
from .task import TaskSpec
from .types import RunResult


RETRYABLE_RUN_STATUSES = frozenset({"runtime_error", "batch_error", "stopped"})
EXPERIMENT_MANIFEST_NAME = "experiment.json"


@dataclass(frozen=True)
class BatchOptions:
    input_root: Path
    run_root: Path
    concurrency: int = 8
    resume: bool = False
    model_key: str | None = None
    harness_key: str | None = None


def result_payload(result: RunResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "status": result.status,
        "prediction_path": (
            str(result.prediction_path)
            if result.prediction_path is not None
            else None
        ),
        "model_turns": result.model_turns,
        "tool_actions": result.tool_actions,
        "wall_time_seconds": round(result.wall_time_seconds, 6),
        "usage": result.usage,
        "error": result.error,
    }


def task_number(path: Path) -> int:
    match = re.fullmatch(r"task_(\d+)", path.name)
    if match is None:
        raise ValueError(f"invalid task directory name: {path.name}")
    return int(match.group(1))


def task_directories(input_root: Path) -> list[Path]:
    root = input_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {root}")
    tasks = [
        path
        for path in root.iterdir()
        if path.is_dir() and re.fullmatch(r"task_\d+", path.name)
    ]
    if not tasks:
        raise ValueError(f"no task_N directories found under {root}")
    return sorted(tasks, key=task_number)


def attempt_directories(task_root: Path) -> list[Path]:
    if not task_root.is_dir():
        return []
    attempts = [
        path
        for path in task_root.iterdir()
        if path.is_dir() and re.fullmatch(r"run_\d+", path.name)
    ]
    return sorted(attempts, key=lambda path: int(path.name.removeprefix("run_")))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def recorded_question(attempt: Path) -> str | None:
    events_path = attempt / "events.jsonl"
    if not events_path.is_file():
        return None
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") == "run_started":
                question = event.get("question")
                return str(question) if question is not None else None
    return None


def finalized_attempt(
    task_root: Path, task: TaskSpec | None = None
) -> tuple[Path, dict[str, Any]] | None:
    for attempt in reversed(attempt_directories(task_root)):
        summary_path = attempt / "run.json"
        if not summary_path.is_file():
            continue
        record = load_json(summary_path)
        if str(record.get("status", "unknown")) in RETRYABLE_RUN_STATUSES:
            continue
        if task is not None:
            previous_question = recorded_question(attempt)
            if previous_question is not None and previous_question != task.question:
                continue
        return attempt, record
    return None


def next_attempt(task_root: Path) -> Path:
    attempts = attempt_directories(task_root)
    next_number = (
        int(attempts[-1].name.removeprefix("run_")) + 1 if attempts else 1
    )
    return task_root / f"run_{next_number}"


def synchronize_prediction(
    attempt: Path,
    predictions_root: Path,
    task_id: str,
    *,
    status: str,
) -> Path | None:
    target_dir = predictions_root / task_id
    target = target_dir / "prediction.csv"
    if status != "submitted":
        target.unlink(missing_ok=True)
        try:
            target_dir.rmdir()
        except OSError:
            pass
        return None

    source = attempt / "output" / "prediction.csv"
    if not source.is_file():
        target.unlink(missing_ok=True)
        try:
            target_dir.rmdir()
        except OSError:
            pass
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def write_batch_summary(
    *,
    path: Path,
    options: BatchOptions,
    task_count: int,
    records: list[dict[str, Any]],
) -> None:
    status_counts: dict[str, int] = {}
    usage: dict[str, int] = {}
    wall_time_seconds = 0.0
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        wall_time_seconds += float(record.get("wall_time_seconds", 0) or 0)
        task_usage = record.get("usage")
        if isinstance(task_usage, dict):
            for key, value in task_usage.items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
    payload: dict[str, Any] = {
        "input_root": str(options.input_root.resolve()),
        "task_count": task_count,
        "concurrency": options.concurrency,
        "finalized_task_count": len(records),
        "pending_task_count": task_count - len(records),
        "status_counts": status_counts,
        "wall_time_seconds": round(wall_time_seconds, 6),
        "usage": usage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": records,
    }
    if options.model_key is not None:
        payload["model"] = options.model_key
    if options.harness_key is not None:
        payload["harness"] = options.harness_key
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def ensure_experiment_identity(
    run_root: Path,
    runner: HarnessRunner,
    *,
    resume: bool,
    had_existing_artifacts: bool,
) -> None:
    fingerprint = getattr(runner, "resume_fingerprint", None)
    identity = getattr(runner, "resume_identity", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        return
    if not isinstance(identity, dict):
        identity = {}
    path = run_root / EXPERIMENT_MANIFEST_NAME
    if path.is_file():
        recorded = load_json(path)
        recorded_fingerprint = recorded.get("fingerprint")
        if recorded_fingerprint != fingerprint:
            raise ValueError(
                "run root belongs to a different experiment configuration: "
                f"{run_root}; use a new --run-id or --run-root"
            )
        return
    if resume and had_existing_artifacts:
        raise ValueError(
            "existing run root predates experiment identity tracking and "
            f"cannot be safely resumed: {run_root}; use a new --run-id or "
            "--run-root"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {"fingerprint": fingerprint, "identity": identity},
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")
    temporary.replace(path)


@contextmanager
def _termination_as_interrupt():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in signals}

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    try:
        for number in signals:
            signal.signal(number, interrupt)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_batch(options: BatchOptions, runner: HarnessRunner) -> int:
    with _termination_as_interrupt():
        return _run_batch(options, runner)


def _run_batch(options: BatchOptions, runner: HarnessRunner) -> int:
    if options.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    tasks = task_directories(options.input_root)
    run_root = options.run_root.resolve()
    had_existing_artifacts = run_root.exists() and any(run_root.iterdir())
    if had_existing_artifacts and not options.resume:
        raise ValueError(
            f"run root already exists and is not empty: {run_root}; "
            "pass --resume to continue it"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    ensure_experiment_identity(
        run_root,
        runner,
        resume=options.resume,
        had_existing_artifacts=had_existing_artifacts,
    )
    tasks_root = run_root / "tasks"
    predictions_root = run_root / "predictions"
    tasks_root.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "batch_summary.json"

    records_by_task: dict[str, dict[str, Any]] = {}
    pending: list[tuple[int, TaskSpec, Path]] = []
    for index, task_dir in enumerate(tasks, start=1):
        task = TaskSpec.load(task_dir)
        task_root = tasks_root / task.task_id
        finalized = finalized_attempt(task_root, task)
        if finalized is not None:
            attempt, record = finalized
            if not options.resume:
                raise ValueError(f"existing finalized task run: {attempt}")
            records_by_task[task.task_id] = record
            synchronize_prediction(
                attempt,
                predictions_root,
                task.task_id,
                status=str(record.get("status", "unknown")),
            )
            print(
                f"[{index}/{len(tasks)}] {task.task_id}: "
                f"skip finalized status={record.get('status')}"
            )
            continue
        pending.append((index, task, next_attempt(task_root)))

    def write_current_summary() -> None:
        ordered = [
            records_by_task[path.name]
            for path in tasks
            if path.name in records_by_task
        ]
        write_batch_summary(
            path=summary_path,
            options=options,
            task_count=len(tasks),
            records=ordered,
        )

    write_current_summary()
    print(
        f"batch start: tasks={len(tasks)} finalized={len(records_by_task)} "
        f"pending={len(pending)} concurrency={options.concurrency}"
    )

    def run_one(job: tuple[int, TaskSpec, Path]) -> tuple[int, TaskSpec, Path, RunResult]:
        index, task, attempt = job
        return index, task, attempt, runner.run(task, str(attempt))

    completed = len(records_by_task)
    with ThreadPoolExecutor(
        max_workers=options.concurrency,
        thread_name_prefix="dataspace-task",
    ) as executor:
        futures = {executor.submit(run_one, job): job for job in pending}
        try:
            for future in as_completed(futures):
                index, task, attempt = futures[future]
                try:
                    _, _, _, result = future.result()
                    record = result_payload(result)
                    prediction = synchronize_prediction(
                        attempt,
                        predictions_root,
                        task.task_id,
                        status=result.status,
                    )
                    detail = (
                        f"status={result.status} turns={result.model_turns} "
                        f"actions={result.tool_actions} "
                        f"seconds={result.wall_time_seconds:.2f} "
                        f"prediction={'yes' if prediction else 'no'}"
                    )
                except Exception as exc:
                    synchronize_prediction(
                        attempt,
                        predictions_root,
                        task.task_id,
                        status="batch_error",
                    )
                    record = {
                        "task_id": task.task_id,
                        "status": "batch_error",
                        "prediction_path": None,
                        "model_turns": 0,
                        "tool_actions": 0,
                        "wall_time_seconds": 0.0,
                        "usage": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    detail = f"status=batch_error error={record['error']}"
                record["run_dir"] = str(attempt)
                records_by_task[task.task_id] = record
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}; source={index}] "
                    f"{task.task_id}: {detail}"
                )
                write_current_summary()
        except KeyboardInterrupt:
            cancel = getattr(runner, "cancel", None)
            if callable(cancel):
                cancel()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            write_current_summary()
            raise

    ordered_records = [records_by_task[path.name] for path in tasks]
    write_batch_summary(
        path=summary_path,
        options=options,
        task_count=len(tasks),
        records=ordered_records,
    )
    submitted = sum(
        record.get("status") == "submitted" for record in ordered_records
    )
    print(
        f"batch complete: {submitted}/{len(tasks)} tasks submitted; "
        f"summary={summary_path}; predictions={predictions_root}"
    )
    return 0 if submitted == len(tasks) else 1
