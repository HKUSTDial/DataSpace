from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "evaluate.py"


class EvaluatorCliTests(unittest.TestCase):
    def test_requires_explicit_config_root(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                str(EVALUATOR_PATH),
                "--prediction-root",
                "predictions",
                "--gold-root",
                "gold",
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--config-root", completed.stderr)

    def test_reads_prediction_gold_and_config_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "predictions" / "task_1"
            gold = root / "gold" / "task_1"
            configs = root / "configs"
            prediction.mkdir(parents=True)
            gold.mkdir(parents=True)
            configs.mkdir()
            unscored_prediction = root / "predictions" / "task_2"
            unscored_prediction.mkdir()
            (prediction / "prediction.csv").write_text(
                "answer\nA\n",
                encoding="utf-8",
            )
            (unscored_prediction / "prediction.csv").write_text(
                "answer\nB\n",
                encoding="utf-8",
            )
            (gold / "gold.csv").write_text(
                "value\nA\n",
                encoding="utf-8",
            )
            (configs / "task_1.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "task_id": "task_1",
                        "order_sensitive": False,
                        "columns": [
                            {
                                "gold_index": 0,
                                "gold_name": "value",
                                "type": "text",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "summary.json"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(EVALUATOR_PATH),
                    "--prediction-root",
                    str(root / "predictions"),
                    "--gold-root",
                    str(root / "gold"),
                    "--config-root",
                    str(configs),
                    "--output",
                    str(output),
                ),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["metric"], "task_accuracy")
            self.assertEqual(summary["task_count"], 1)
            self.assertEqual(summary["passed_task_count"], 1)
            self.assertEqual(summary["task_accuracy"], 1.0)
            self.assertEqual(summary["unscored_prediction_tasks"], ["task_2"])
            self.assertNotIn("unexpected_prediction_tasks", summary)


if __name__ == "__main__":
    unittest.main()
