from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging import RunLogger, utc_now
from ..core.prediction import validate_prediction
from ..core.task import TaskSpec
from ..core.types import RunResult


@dataclass(frozen=True)
class NativeCliInvocation:
    command: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    inherit_environment: bool = True


InvocationFactory = Callable[[TaskSpec, Path, Path], NativeCliInvocation]
TraceParser = Callable[[Path], tuple[int, int, dict[str, int]]]
RunCleanup = Callable[[Path], None]
RunValidator = Callable[[Path], None]
ExitClassifier = Callable[[Path, int], tuple[str, str | None] | None]


class NativeCliRunner:
    """Run a stock harness CLI with one hard wall-clock deadline.

    Harness-specific planning, tools, memory, and command execution remain
    inside the invoked program. This runner only enforces lifecycle and the
    released prediction-file contract.
    """

    def __init__(
        self,
        *,
        harness_key: str,
        model_key: str,
        wall_time_seconds: int,
        invocation_factory: InvocationFactory,
        trace_parser: TraceParser | None = None,
        run_validator: RunValidator | None = None,
        exit_classifier: ExitClassifier | None = None,
        cleanup: RunCleanup | None = None,
        termination_grace_seconds: float = 5.0,
        resume_identity: Mapping[str, Any] | None = None,
    ):
        if wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive")
        if termination_grace_seconds < 0:
            raise ValueError("termination_grace_seconds must be non-negative")
        self.harness_key = harness_key
        self.model_key = model_key
        self.wall_time_seconds = wall_time_seconds
        self.invocation_factory = invocation_factory
        self.trace_parser = trace_parser
        self.run_validator = run_validator
        self.exit_classifier = exit_classifier
        self.cleanup = cleanup
        self.termination_grace_seconds = termination_grace_seconds
        self.resume_identity = dict(resume_identity or {})
        self._cancel_event = threading.Event()
        self._process_lock = threading.Lock()
        self._active_processes: dict[int, subprocess.Popen[bytes]] = {}

    @property
    def resume_fingerprint(self) -> str | None:
        if not self.resume_identity:
            return None
        serialized = json.dumps(
            self.resume_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def cancel(self) -> None:
        """Stop all active task processes; stopped attempts remain resumable."""

        self._cancel_event.set()
        with self._process_lock:
            processes = list(self._active_processes.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + self.termination_grace_seconds
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @staticmethod
    def _terminate_process_group(
        process: subprocess.Popen[bytes], grace_seconds: float
    ) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()

    def run(self, task: TaskSpec, run_dir: str | Path) -> RunResult:
        run_root = Path(run_dir).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        output_root = run_root / "output"
        runtime_root = run_root / "runtime"
        output_root.mkdir(parents=True, exist_ok=True)
        runtime_root.mkdir(parents=True, exist_ok=True)
        prediction_path = output_root / "prediction.csv"
        logger = RunLogger(run_root)

        invocation = self.invocation_factory(task, run_root, output_root)
        if not invocation.command:
            raise ValueError("native harness command must not be empty")
        invocation_cwd = invocation.cwd.resolve()
        if not invocation_cwd.is_dir():
            raise FileNotFoundError(
                f"native harness working directory does not exist: {invocation_cwd}"
            )

        stdout_path = run_root / "stdout.log"
        stderr_path = run_root / "stderr.log"
        environment = os.environ.copy() if invocation.inherit_environment else {}
        environment.update({str(k): str(v) for k, v in invocation.environment.items()})
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        status = "runtime_error"
        error: str | None = None
        return_code: int | None = None

        logger.event(
            "run_started",
            {
                "task_id": task.task_id,
                "question": task.question,
                "workspace": str(task.context_dir),
                "started_at": utc_now(),
                "harness": self.harness_key,
                "model": self.model_key,
                "experiment_fingerprint": self.resume_fingerprint,
                "limits": {"max_wall_time_seconds": self.wall_time_seconds},
                "invocation": {
                    "command": list(invocation.command),
                    "cwd": str(invocation_cwd),
                    "environment_keys": sorted(invocation.environment),
                    "metadata": dict(invocation.metadata),
                },
            },
        )

        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    list(invocation.command),
                    cwd=invocation_cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                with self._process_lock:
                    self._active_processes[process.pid] = process
                if self._cancel_event.is_set():
                    self._terminate_process_group(
                        process, self.termination_grace_seconds
                    )
                try:
                    return_code = process.wait(timeout=self.wall_time_seconds)
                except subprocess.TimeoutExpired:
                    status = "wall_time_exhausted"
                    self._terminate_process_group(
                        process, self.termination_grace_seconds
                    )
                    return_code = process.returncode

            if self._cancel_event.is_set():
                status = "stopped"
                error = None
            elif status != "wall_time_exhausted":
                if return_code != 0:
                    classification = None
                    if self.exit_classifier is not None:
                        try:
                            classification = self.exit_classifier(
                                run_root, return_code
                            )
                        except Exception as exc:
                            logger.event(
                                "exit_classification_error",
                                {"error": f"{type(exc).__name__}: {exc}"},
                            )
                    if classification is None:
                        status = "runtime_error"
                        error = (
                            f"native harness exited with code {return_code}"
                        )
                    else:
                        status, error = classification
                        logger.event(
                            "exit_classified",
                            {
                                "status": status,
                                "return_code": return_code,
                                "error": error,
                            },
                        )
                else:
                    try:
                        if self.run_validator is not None:
                            self.run_validator(run_root)
                    except Exception as exc:
                        status = "runtime_error"
                        error = (
                            "run validation failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        logger.event("run_validation_failed", {"error": error})
                    else:
                        if not prediction_path.is_file():
                            status = "no_prediction"
                            csv_candidates = sorted(
                                path.name
                                for path in output_root.glob("*.csv")
                                if path.is_file()
                            )
                            if csv_candidates:
                                error = (
                                    "required prediction.csv was not created; "
                                    "found: " + ", ".join(csv_candidates)
                                )
                            else:
                                error = "required prediction.csv was not created"
                        else:
                            try:
                                prediction_metadata = validate_prediction(
                                    prediction_path
                                )
                            except Exception as exc:
                                status = "invalid_prediction"
                                error = f"{type(exc).__name__}: {exc}"
                            else:
                                status = "submitted"
                                logger.event(
                                    "prediction_validated",
                                    {"prediction": prediction_metadata},
                                )
        except Exception as exc:
            if self._cancel_event.is_set():
                status = "stopped"
                error = None
            else:
                status = "runtime_error"
                error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                self._terminate_process_group(
                    process, self.termination_grace_seconds
                )
        finally:
            if process is not None:
                with self._process_lock:
                    self._active_processes.pop(process.pid, None)

        if self.cleanup is not None:
            try:
                self.cleanup(run_root)
            except Exception as exc:
                logger.event(
                    "cleanup_error",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

        wall_time = time.monotonic() - started
        model_turns = 0
        tool_actions = 0
        usage: dict[str, int] = {}
        if self.trace_parser is not None:
            try:
                model_turns, tool_actions, usage = self.trace_parser(run_root)
            except Exception as exc:
                logger.event(
                    "trace_parse_error",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

        accepted_prediction = prediction_path if status == "submitted" else None
        result = RunResult(
            task_id=task.task_id,
            status=status,
            run_dir=run_root,
            prediction_path=accepted_prediction,
            model_turns=model_turns,
            tool_actions=tool_actions,
            wall_time_seconds=wall_time,
            usage=usage,
            error=error,
        )
        logger.event(
            "run_finished",
            {
                "status": status,
                "return_code": return_code,
                "wall_time_seconds": round(wall_time, 6),
                "prediction_path": (
                    str(accepted_prediction)
                    if accepted_prediction is not None
                    else None
                ),
                "error": error,
            },
        )
        logger.write_summary(
            {
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
                "harness": self.harness_key,
                "model": self.model_key,
                "return_code": return_code,
            }
        )
        return result
