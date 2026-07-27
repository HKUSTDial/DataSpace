from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .task import TaskSpec
from .types import RunResult


class HarnessRunner(Protocol):
    """Minimal contract shared by in-process and native-CLI harnesses."""

    def run(self, task: TaskSpec, run_dir: str | Path) -> RunResult:
        ...

    def cancel(self) -> None:
        ...
