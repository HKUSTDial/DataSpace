from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataspace_baselines.core.batch import (
    ensure_experiment_identity,
    finalized_attempt,
)


class _Runner:
    def __init__(self, fingerprint: str, identity: dict[str, object]):
        self.resume_fingerprint = fingerprint
        self.resume_identity = identity


class ExperimentIdentityTests(unittest.TestCase):
    def test_model_budget_failure_is_finalized_but_runtime_error_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "task_1"
            run_1 = task_root / "run_1"
            run_2 = task_root / "run_2"
            run_1.mkdir(parents=True)
            run_2.mkdir()
            (run_1 / "run.json").write_text(
                json.dumps({"status": "model_output_exhausted"}),
                encoding="utf-8",
            )
            (run_2 / "run.json").write_text(
                json.dumps({"status": "runtime_error"}),
                encoding="utf-8",
            )

            finalized = finalized_attempt(task_root)

        self.assertIsNotNone(finalized)
        assert finalized is not None
        self.assertEqual(finalized[0].name, "run_1")

    def test_records_and_accepts_matching_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            runner = _Runner("same", {"workbench": "1.0"})

            ensure_experiment_identity(
                run_root,
                runner,
                resume=False,
                had_existing_artifacts=False,
            )
            ensure_experiment_identity(
                run_root,
                runner,
                resume=True,
                had_existing_artifacts=True,
            )
            record = json.loads(
                (run_root / "experiment.json").read_text(encoding="utf-8")
            )

        self.assertEqual(record["fingerprint"], "same")
        self.assertEqual(record["identity"], {"workbench": "1.0"})

    def test_rejects_resume_with_different_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            ensure_experiment_identity(
                run_root,
                _Runner("old", {"workbench": "old"}),
                resume=False,
                had_existing_artifacts=False,
            )

            with self.assertRaisesRegex(
                ValueError, "different experiment configuration"
            ):
                ensure_experiment_identity(
                    run_root,
                    _Runner("new", {"workbench": "1.0"}),
                    resume=True,
                    had_existing_artifacts=True,
                )

    def test_rejects_legacy_nonempty_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            (run_root / "tasks").mkdir()

            with self.assertRaisesRegex(
                ValueError, "predates experiment identity tracking"
            ):
                ensure_experiment_identity(
                    run_root,
                    _Runner("new", {"workbench": "1.0"}),
                    resume=True,
                    had_existing_artifacts=True,
                )


if __name__ == "__main__":
    unittest.main()
