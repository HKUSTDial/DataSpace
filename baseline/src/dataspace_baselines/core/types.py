from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    task_id: str
    status: str
    run_dir: Path
    prediction_path: Path | None
    model_turns: int
    tool_actions: int
    wall_time_seconds: float
    usage: dict[str, int]
    error: str | None = None
