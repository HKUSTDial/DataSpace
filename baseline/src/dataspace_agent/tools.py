from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataspace_baselines.core.prediction import validate_prediction

from .media import ImageObserver
from .sandbox import Sandbox
from .task import TaskPaths
from .types import ToolCall, ToolOutcome


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command inside the isolated task container. "
                "The task workspace is read-only at /workspace. Write scripts, "
                "derived media, and prediction files under /output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Requested timeout in seconds. It is capped by "
                            "the benchmark harness."
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": (
                "Inspect exactly one image. Use bash to render a PDF page or "
                "extract a video frame into /output, then pass that image path. "
                "The tool performs no OCR, retrieval, or semantic analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "An image path under /workspace or /output."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit a rectangular CSV under /output as the final answer "
                "and terminate the run. No evaluation feedback is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the final CSV under /output."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolExecutor:
    def __init__(
        self,
        sandbox: Sandbox,
        paths: TaskPaths,
        image_observer: ImageObserver,
        max_command_seconds: int,
    ):
        self.sandbox = sandbox
        self.paths = paths
        self.image_observer = image_observer
        self.max_command_seconds = max_command_seconds

    @staticmethod
    def _arguments(call: ToolCall) -> dict[str, Any]:
        try:
            parsed = json.loads(call.arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed

    def execute(
        self,
        call: ToolCall,
        *,
        command_timeout_cap: int | None = None,
    ) -> ToolOutcome:
        try:
            arguments = self._arguments(call)
            if call.name == "bash":
                return self._bash(arguments, command_timeout_cap)
            if call.name == "view_image":
                return self._view_image(arguments)
            if call.name == "submit_answer":
                return self._submit(arguments)
            raise ValueError(f"unknown tool: {call.name}")
        except Exception as exc:
            return ToolOutcome(
                result={
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    def _bash(
        self,
        arguments: dict[str, Any],
        command_timeout_cap: int | None = None,
    ) -> ToolOutcome:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("bash.command must be a non-empty string")
        requested_timeout = arguments.get(
            "timeout", self.max_command_seconds
        )
        if isinstance(requested_timeout, bool):
            raise ValueError("bash.timeout must be an integer")
        timeout = max(
            1, min(int(requested_timeout), self.max_command_seconds)
        )
        if command_timeout_cap is not None:
            timeout = min(timeout, max(1, int(command_timeout_cap)))
        result = self.sandbox.execute(command, timeout)
        return ToolOutcome(
            result={
                "ok": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "duration_seconds": round(result.duration_seconds, 6),
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            }
        )

    def _view_image(self, arguments: dict[str, Any]) -> ToolOutcome:
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("view_image.path must be a non-empty string")
        observation = self.image_observer.observe(path)
        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Visual observation returned by view_image for "
                        f"{path}. Analyze the pixels directly."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": observation.data_url,
                        "detail": "high",
                    },
                },
            ],
        }
        return ToolOutcome(
            result={"ok": True, "image": observation.metadata},
            image_message=image_message,
        )

    def _submit(self, arguments: dict[str, Any]) -> ToolOutcome:
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("submit_answer.path must be a non-empty string")
        host_path = self.paths.resolve_agent_path(
            path, output_only=True, must_exist=True
        )
        metadata = validate_prediction(host_path)
        return ToolOutcome(
            result={
                "ok": True,
                "accepted": True,
                "message": "Answer recorded; no evaluation feedback is provided.",
                "prediction": metadata,
            },
            submitted_path=host_path,
        )
