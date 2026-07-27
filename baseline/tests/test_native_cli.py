from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from dataspace_baselines.core.prompt import relation_task_prompt
from dataspace_baselines.core.task import TaskSpec
from dataspace_baselines.runners.native_cli import (
    NativeCliInvocation,
    NativeCliRunner,
)


def _task(root: Path) -> TaskSpec:
    task_dir = root / "task_1"
    context = task_dir / "context"
    context.mkdir(parents=True)
    (context / "data.csv").write_text("value\n42\n", encoding="utf-8")
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_1", "question": "Return 42."}),
        encoding="utf-8",
    )
    return TaskSpec.load(task_dir)


class NativeCliRunnerTests(unittest.TestCase):
    def test_common_prompt_names_workspace_output_and_empty_relation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = _task(Path(temporary))
            prompt = relation_task_prompt(
                task,
                workspace_path="/workspace",
                output_path="/output/prediction.csv",
            )
        self.assertIn("Return 42.", prompt)
        self.assertIn("/workspace", prompt)
        self.assertIn("/output/prediction.csv", prompt)
        self.assertIn("header-only", prompt)
        self.assertIn("Do not finish the task until the exact file path", prompt)
        self.assertIn("filename is exactly prediction.csv", prompt)

    def test_accepts_prediction_only_after_clean_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)

            def invocation(task, run_root, output_root):
                script = (
                    "from pathlib import Path; "
                    f"Path({str(output_root / 'prediction.csv')!r}).write_text("
                    "'answer\\n42\\n', encoding='utf-8')"
                )
                return NativeCliInvocation(
                    command=(sys.executable, "-c", script),
                    cwd=task.context_dir,
                )

            result = NativeCliRunner(
                harness_key="fake",
                model_key="fake-model",
                wall_time_seconds=5,
                invocation_factory=invocation,
            ).run(task, root / "run")

            self.assertEqual(result.status, "submitted")
            self.assertTrue(result.prediction_path.is_file())
            self.assertTrue((root / "run" / "stdout.log").is_file())
            self.assertTrue((root / "run" / "run.json").is_file())

    def test_timeout_does_not_accept_draft_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)
            cleaned: list[Path] = []

            def invocation(task, run_root, output_root):
                script = (
                    "import time; from pathlib import Path; "
                    f"Path({str(output_root / 'prediction.csv')!r}).write_text("
                    "'answer\\ndraft\\n', encoding='utf-8'); "
                    "time.sleep(30)"
                )
                return NativeCliInvocation(
                    command=(sys.executable, "-c", script),
                    cwd=task.context_dir,
                )

            result = NativeCliRunner(
                harness_key="fake",
                model_key="fake-model",
                wall_time_seconds=1,
                invocation_factory=invocation,
                cleanup=cleaned.append,
                termination_grace_seconds=0.1,
            ).run(task, root / "run")

            self.assertEqual(result.status, "wall_time_exhausted")
            self.assertIsNone(result.prediction_path)
            self.assertTrue(
                (root / "run" / "output" / "prediction.csv").is_file()
            )
            self.assertEqual(cleaned, [(root / "run").resolve()])

    def test_rejects_non_rectangular_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)

            def invocation(task, run_root, output_root):
                script = (
                    "from pathlib import Path; "
                    f"Path({str(output_root / 'prediction.csv')!r}).write_text("
                    "'a,b\\n1\\n', encoding='utf-8')"
                )
                return NativeCliInvocation(
                    command=(sys.executable, "-c", script),
                    cwd=task.context_dir,
                )

            result = NativeCliRunner(
                harness_key="fake",
                model_key="fake-model",
                wall_time_seconds=5,
                invocation_factory=invocation,
            ).run(task, root / "run")

            self.assertEqual(result.status, "invalid_prediction")
            self.assertIsNone(result.prediction_path)

    def test_reports_misnamed_csv_without_accepting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)

            def invocation(task, run_root, output_root):
                script = (
                    "from pathlib import Path; "
                    f"Path({str(output_root / 'predication.csv')!r}).write_text("
                    "'answer\\n42\\n', encoding='utf-8')"
                )
                return NativeCliInvocation(
                    command=(sys.executable, "-c", script),
                    cwd=task.context_dir,
                )

            result = NativeCliRunner(
                harness_key="fake",
                model_key="fake-model",
                wall_time_seconds=5,
                invocation_factory=invocation,
            ).run(task, root / "run")

            self.assertEqual(result.status, "no_prediction")
            self.assertIsNone(result.prediction_path)
            self.assertIn("predication.csv", result.error)

    def test_run_validator_rejects_prediction_after_clean_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)

            def invocation(task, run_root, output_root):
                script = (
                    "from pathlib import Path; "
                    f"Path({str(output_root / 'prediction.csv')!r}).write_text("
                    "'answer\\n42\\n', encoding='utf-8')"
                )
                return NativeCliInvocation(
                    command=(sys.executable, "-c", script),
                    cwd=task.context_dir,
                )

            def reject(_run_root):
                raise RuntimeError("unexpected side model")

            result = NativeCliRunner(
                harness_key="fake",
                model_key="fake-model",
                wall_time_seconds=5,
                invocation_factory=invocation,
                run_validator=reject,
            ).run(task, root / "run")

            self.assertEqual(result.status, "runtime_error")
            self.assertIsNone(result.prediction_path)
            self.assertIn("unexpected side model", result.error)

    def test_exit_classifier_finalizes_model_budget_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)

            def invocation(task, run_root, output_root):
                return NativeCliInvocation(
                    command=(sys.executable, "-c", "raise SystemExit(1)"),
                    cwd=task.context_dir,
                )

            result = NativeCliRunner(
                harness_key="fake",
                model_key="fake-model",
                wall_time_seconds=5,
                invocation_factory=invocation,
                exit_classifier=lambda _root, _code: (
                    "model_output_exhausted",
                    "response reached max tokens",
                ),
            ).run(task, root / "run")

            self.assertEqual(result.status, "model_output_exhausted")
            self.assertIsNone(result.prediction_path)
            self.assertEqual(result.error, "response reached max tokens")

    def test_cancel_stops_process_and_marks_attempt_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = _task(root)
            invocation_ready = threading.Event()
            result_holder = []

            def invocation(task, run_root, output_root):
                invocation_ready.set()
                return NativeCliInvocation(
                    command=(sys.executable, "-c", "import time; time.sleep(30)"),
                    cwd=task.context_dir,
                )

            runner = NativeCliRunner(
                harness_key="fake",
                model_key="fake-model",
                wall_time_seconds=30,
                invocation_factory=invocation,
                termination_grace_seconds=0.1,
            )
            thread = threading.Thread(
                target=lambda: result_holder.append(
                    runner.run(task, root / "run")
                )
            )
            thread.start()
            self.assertTrue(invocation_ready.wait(timeout=1))
            time.sleep(0.05)
            runner.cancel()
            thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(result_holder[0].status, "stopped")
            self.assertIsNone(result_holder[0].prediction_path)


if __name__ == "__main__":
    unittest.main()
