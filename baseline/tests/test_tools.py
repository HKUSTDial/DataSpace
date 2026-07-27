from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataspace_agent.media import ImageObserver
from dataspace_agent.task import TaskPaths
from dataspace_agent.tools import ToolExecutor, validate_prediction
from dataspace_agent.types import CommandResult, ToolCall


class FakeSandbox:
    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def execute(
        self, command: str, timeout_seconds: int
    ) -> CommandResult:
        return CommandResult(
            exit_code=0,
            stdout=f"ran: {command}",
            stderr="",
            timed_out=False,
            duration_seconds=0.01,
        )


class ToolTests(unittest.TestCase):
    def test_validates_rectangular_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.csv"
            path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            metadata = validate_prediction(path)
            self.assertEqual(metadata["columns"], 2)
            self.assertEqual(metadata["data_rows"], 2)

            path.write_text("a,b\n1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_prediction(path)

    def test_submit_requires_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            output = root / "output"
            workspace.mkdir()
            output.mkdir()
            prediction = output / "prediction.csv"
            prediction.write_text("answer\n1\n", encoding="utf-8")
            paths = TaskPaths(workspace, output)
            executor = ToolExecutor(
                FakeSandbox(),
                paths,
                ImageObserver(paths, 800, 90),
                max_command_seconds=30,
            )

            accepted = executor.execute(
                ToolCall(
                    call_id="call-1",
                    name="submit_answer",
                    arguments_json='{"path": "/output/prediction.csv"}',
                )
            )
            rejected = executor.execute(
                ToolCall(
                    call_id="call-2",
                    name="submit_answer",
                    arguments_json='{"path": "/workspace/prediction.csv"}',
                )
            )

            self.assertTrue(accepted.result["accepted"])
            self.assertEqual(accepted.submitted_path, prediction.resolve())
            self.assertFalse(rejected.result["ok"])
            self.assertIsNone(rejected.submitted_path)


if __name__ == "__main__":
    unittest.main()
