from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataspace_agent.config import load_run_config


class ConfigTests(unittest.TestCase):
    def test_loads_selected_model_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                """
models:
  demo:
    model: provider/demo
    api_key_env: DEMO_API_KEY
agent:
  max_tool_actions: 7
sandbox:
  image: demo-sandbox
""",
                encoding="utf-8",
            )
            config = load_run_config(path, "demo")
        self.assertEqual(config.model.model, "provider/demo")
        self.assertEqual(config.agent.max_tool_actions, 7)
        self.assertEqual(config.agent.max_model_turns, 60)
        self.assertEqual(config.sandbox.image, "demo-sandbox")

    def test_rejects_unknown_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text("models: {}\n", encoding="utf-8")
            with self.assertRaises(KeyError):
                load_run_config(path, "missing")


if __name__ == "__main__":
    unittest.main()
