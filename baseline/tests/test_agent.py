from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataspace_agent.agent import DataSpaceAgent
from dataspace_agent.config import AgentConfig
from dataspace_agent.task import TaskSpec
from dataspace_agent.types import (
    CommandResult,
    ModelTurn,
    ToolCall,
)


class FakeBackend:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.calls = 0

    def complete(
        self, messages, tools, timeout_seconds=None
    ) -> ModelTurn:
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "bash-1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "solve"}',
                            },
                        }
                    ],
                },
                tool_calls=[
                    ToolCall("bash-1", "bash", '{"command": "solve"}')
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 4},
                resolved_model="fake-model",
            )
        return ModelTurn(
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "submit-1",
                        "type": "function",
                        "function": {
                            "name": "submit_answer",
                            "arguments": (
                                '{"path": "/output/prediction.csv"}'
                            ),
                        },
                    }
                ],
            },
            tool_calls=[
                ToolCall(
                    "submit-1",
                    "submit_answer",
                    '{"path": "/output/prediction.csv"}',
                )
            ],
            usage={"prompt_tokens": 12, "completion_tokens": 3},
            resolved_model="fake-model",
        )


class FakeSandbox:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.started = False
        self.closed = False
        self.command_timeouts: list[int] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def execute(
        self, command: str, timeout_seconds: int
    ) -> CommandResult:
        self.command_timeouts.append(timeout_seconds)
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "prediction.csv").write_text(
            "answer\n42\n", encoding="utf-8"
        )
        return CommandResult(
            exit_code=0,
            stdout="created prediction",
            stderr="",
            timed_out=False,
            duration_seconds=0.01,
        )


class AgentTests(unittest.TestCase):
    def test_end_to_end_fake_run_submits_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "task_1"
            context = task_dir / "context"
            context.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {"task_id": "task_1", "question": "Return 42."}
                ),
                encoding="utf-8",
            )
            run_dir = root / "run"
            output_root = run_dir / "output"
            backend = FakeBackend(output_root)
            sandbox_holder = {}

            def sandbox_factory(workspace: Path, output: Path):
                sandbox = FakeSandbox(output)
                sandbox_holder["sandbox"] = sandbox
                return sandbox

            agent = DataSpaceAgent(
                backend=backend,
                agent_config=AgentConfig(
                    max_model_turns=5,
                    max_tool_actions=5,
                    max_wall_time_seconds=30,
                ),
                sandbox_factory=sandbox_factory,
            )
            result = agent.run(TaskSpec.load(task_dir), run_dir)

            self.assertEqual(result.status, "submitted")
            self.assertEqual(result.tool_actions, 2)
            self.assertEqual(result.model_turns, 2)
            self.assertEqual(result.usage["input_tokens"], 22)
            self.assertTrue(result.prediction_path.is_file())
            self.assertTrue(sandbox_holder["sandbox"].closed)
            self.assertTrue((run_dir / "events.jsonl").is_file())
            self.assertTrue((run_dir / "run.json").is_file())
            # Compatibility default: the historical command cap is unchanged.
            self.assertEqual(sandbox_holder["sandbox"].command_timeouts, [180])

    def test_common_harness_caps_command_to_remaining_wall_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "task_1"
            (task_dir / "context").mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps({"task_id": "task_1", "question": "Return 42."}),
                encoding="utf-8",
            )
            run_dir = root / "run"
            output_root = run_dir / "output"
            backend = FakeBackend(output_root)
            sandbox_holder = {}

            def sandbox_factory(workspace: Path, output: Path):
                sandbox = FakeSandbox(output)
                sandbox_holder["sandbox"] = sandbox
                return sandbox

            agent = DataSpaceAgent(
                backend=backend,
                agent_config=AgentConfig(
                    max_model_turns=5,
                    max_tool_actions=5,
                    max_wall_time_seconds=2,
                    max_command_seconds=180,
                ),
                sandbox_factory=sandbox_factory,
                enforce_remaining_wall_time=True,
            )
            result = agent.run(TaskSpec.load(task_dir), run_dir)

            self.assertEqual(result.status, "submitted")
            self.assertLessEqual(sandbox_holder["sandbox"].command_timeouts[0], 2)


if __name__ == "__main__":
    unittest.main()
