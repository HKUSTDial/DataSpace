from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataspace_baselines.core.types import RunResult


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str


@dataclass
class ModelTurn:
    assistant_message: dict[str, Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    response_id: str | None = None
    resolved_model: str | None = None


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True)
class ImageObservation:
    agent_path: str
    data_url: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ToolOutcome:
    result: dict[str, Any]
    image_message: dict[str, Any] | None = None
    submitted_path: Path | None = None
