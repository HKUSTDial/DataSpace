from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataspace_baselines.config import load_harness_experiment_config


class HarnessExperimentConfigTests(unittest.TestCase):
    def test_loads_shared_model_and_harnesses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.yaml"
            path.write_text(
                """
experiment:
  model:
    key: mimo
    model: vendor/mimo
    api_key_env: TEST_KEY
    openai_base_url: https://example.test/v1
    anthropic_base_url: https://example.test
  wall_time_seconds: 90
  concurrency: 3
  workbench:
    image: workbench:test
    rootfs_path: workbench-rootfs
harnesses:
  dataspace-agent:
    enabled: true
  codex:
    executable: /bin/codex
""",
                encoding="utf-8",
            )
            config = load_harness_experiment_config(path)

        self.assertEqual(config.model.model, "vendor/mimo")
        self.assertEqual(config.model.context_window, 200000)
        self.assertEqual(config.wall_time_seconds, 90)
        self.assertEqual(config.concurrency, 3)
        self.assertEqual(config.workbench.image, "workbench:test")
        self.assertEqual(
            config.workbench.rootfs_path,
            (path.parent / "workbench-rootfs").resolve(),
        )
        self.assertEqual(config.harnesses["codex"].executable, "/bin/codex")

    def test_rejects_output_budget_not_smaller_than_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.yaml"
            path.write_text(
                """
experiment:
  model:
    key: demo
    model: vendor/demo
    api_key_env: TEST_KEY
    openai_base_url: https://example.test/v1
    anthropic_base_url: https://example.test
    max_output_tokens: 100
    context_window: 100
harnesses:
  codex: {}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be smaller"):
                load_harness_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
