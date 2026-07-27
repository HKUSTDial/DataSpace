from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from dataspace_agent.cli import (
    _build_parser,
    _finalized_attempt,
    _next_attempt,
    _run_batch,
    _task_directories,
)
from dataspace_agent.task import TaskSpec
from dataspace_agent.types import RunResult


class BatchCliTests(unittest.TestCase):
    def test_batch_default_concurrency_is_eight(self) -> None:
        args = _build_parser().parse_args(
            [
                "batch",
                "--input-root",
                "input",
                "--config",
                "config.yaml",
                "--model",
                "demo",
                "--run-root",
                "runs/demo",
            ]
        )
        self.assertEqual(args.concurrency, 8)

    def test_discovers_tasks_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("task_10", "task_2", "task_1", "notes"):
                (root / name).mkdir()
            tasks = _task_directories(root)
        self.assertEqual([path.name for path in tasks], ["task_1", "task_2", "task_10"])

    def test_uses_new_attempt_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "task_1"
            (task_root / "run_1").mkdir(parents=True)
            self.assertEqual(_next_attempt(task_root).name, "run_2")

    def test_finds_latest_finalized_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "task_1"
            run_1 = task_root / "run_1"
            run_2 = task_root / "run_2"
            run_1.mkdir(parents=True)
            run_2.mkdir()
            (run_1 / "run.json").write_text(
                json.dumps({"status": "submitted"}), encoding="utf-8"
            )
            finalized = _finalized_attempt(task_root)
        self.assertIsNotNone(finalized)
        assert finalized is not None
        self.assertEqual(finalized[0].name, "run_1")
        self.assertEqual(finalized[1]["status"], "submitted")

    def test_runtime_error_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "task_1"
            run_1 = task_root / "run_1"
            run_1.mkdir(parents=True)
            (run_1 / "run.json").write_text(
                json.dumps(
                    {
                        "status": "runtime_error",
                        "error": "APIStatusError: insufficient_funds",
                    }
                ),
                encoding="utf-8",
            )

            finalized = _finalized_attempt(task_root)
            next_attempt = _next_attempt(task_root)

        self.assertIsNone(finalized)
        self.assertEqual(next_attempt.name, "run_2")

    def test_changed_question_is_not_treated_as_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task_1"
            run_1 = task_root / "run_1"
            context = root / "input" / "task_1" / "context"
            run_1.mkdir(parents=True)
            context.mkdir(parents=True)
            (run_1 / "run.json").write_text(
                json.dumps({"status": "submitted"}), encoding="utf-8"
            )
            (run_1 / "events.jsonl").write_text(
                json.dumps(
                    {"event": "run_started", "question": "Old question"}
                )
                + "\n",
                encoding="utf-8",
            )
            task = TaskSpec(
                task_id="task_1",
                question="Revised question",
                task_dir=context.parent,
                context_dir=context,
            )
            finalized = _finalized_attempt(task_root, task)
        self.assertIsNone(finalized)

    def test_runs_tasks_concurrently_and_collects_predictions(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0

            def run(self, task, run_dir) -> RunResult:
                run_root = Path(run_dir)
                output = run_root / "output"
                output.mkdir(parents=True)
                prediction = output / "prediction.csv"
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.03)
                prediction.write_text("answer\n1\n", encoding="utf-8")
                with self.lock:
                    self.active -= 1
                return RunResult(
                    task_id=task.task_id,
                    status="submitted",
                    run_dir=run_root,
                    prediction_path=prediction,
                    model_turns=1,
                    tool_actions=1,
                    wall_time_seconds=0.03,
                    usage={"total_tokens": 1},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            for number in range(1, 4):
                task_dir = input_root / f"task_{number}"
                (task_dir / "context").mkdir(parents=True)
                (task_dir / "task.json").write_text(
                    json.dumps(
                        {
                            "task_id": f"task_{number}",
                            "question": "Return one.",
                        }
                    ),
                    encoding="utf-8",
                )
            run_root = root / "runs"
            args = argparse.Namespace(
                input_root=input_root,
                config=root / "config.yaml",
                model_key="demo",
                run_root=run_root,
                concurrency=2,
                resume=False,
            )
            agent = FakeAgent()
            with mock.patch(
                "dataspace_agent.cli.load_run_config", return_value=object()
            ), mock.patch(
                "dataspace_agent.cli._build_agent", return_value=agent
            ):
                exit_code = _run_batch(args)
            summary = json.loads(
                (run_root / "batch_summary.json").read_text(encoding="utf-8")
            )
            predictions = list(
                (run_root / "predictions").glob("task_*/prediction.csv")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(agent.max_active, 2)
        self.assertEqual(summary["concurrency"], 2)
        self.assertEqual(summary["status_counts"], {"submitted": 3})
        self.assertEqual(len(predictions), 3)

    def test_resume_does_not_materialize_exhausted_draft(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            task_dir = input_root / "task_1"
            (task_dir / "context").mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {"task_id": "task_1", "question": "Return one."}
                ),
                encoding="utf-8",
            )

            run_root = root / "runs"
            attempt = run_root / "tasks" / "task_1" / "run_1"
            output = attempt / "output"
            output.mkdir(parents=True)
            (output / "prediction.csv").write_text(
                "answer\n1\n", encoding="utf-8"
            )
            (attempt / "events.jsonl").write_text(
                json.dumps(
                    {
                        "event": "run_started",
                        "question": "Return one.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (attempt / "run.json").write_text(
                json.dumps(
                    {
                        "task_id": "task_1",
                        "status": "tool_actions_exhausted",
                        "prediction_path": None,
                    }
                ),
                encoding="utf-8",
            )

            stale = run_root / "predictions" / "task_1" / "prediction.csv"
            stale.parent.mkdir(parents=True)
            stale.write_text("answer\n1\n", encoding="utf-8")

            args = argparse.Namespace(
                input_root=input_root,
                config=root / "config.yaml",
                model_key="demo",
                run_root=run_root,
                concurrency=1,
                resume=True,
            )
            with mock.patch(
                "dataspace_agent.cli.load_run_config", return_value=object()
            ), mock.patch("dataspace_agent.cli._build_agent"):
                exit_code = _run_batch(args)

            summary = json.loads(
                (run_root / "batch_summary.json").read_text(encoding="utf-8")
            )
            stale_exists = stale.exists()

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            summary["status_counts"], {"tool_actions_exhausted": 1}
        )
        self.assertFalse(stale_exists)


if __name__ == "__main__":
    unittest.main()
