from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    question: str
    task_dir: Path
    context_dir: Path

    @classmethod
    def load(cls, task_dir: str | Path) -> "TaskSpec":
        root = Path(task_dir).resolve()
        task_json = root / "task.json"
        context_dir = root / "context"
        if not task_json.is_file():
            raise FileNotFoundError(f"missing task.json: {task_json}")
        if not context_dir.is_dir():
            raise FileNotFoundError(f"missing context directory: {context_dir}")
        with task_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        task_id = str(payload.get("task_id", "")).strip()
        question = str(payload.get("question", "")).strip()
        if not task_id:
            raise ValueError(f"task_id is missing from {task_json}")
        if not question:
            raise ValueError(f"question is missing from {task_json}")
        return cls(
            task_id=task_id,
            question=question,
            task_dir=root,
            context_dir=context_dir.resolve(),
        )


@dataclass(frozen=True)
class TaskPaths:
    workspace_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", self.workspace_root.resolve())
        object.__setattr__(self, "output_root", self.output_root.resolve())

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def resolve_agent_path(
        self,
        agent_path: str,
        *,
        output_only: bool = False,
        must_exist: bool = True,
    ) -> Path:
        raw = str(agent_path).strip()
        if not raw:
            raise ValueError("path must not be empty")

        if raw == "/workspace":
            candidate = self.workspace_root
        elif raw.startswith("/workspace/"):
            candidate = self.workspace_root / raw.removeprefix("/workspace/")
        elif raw == "/output":
            candidate = self.output_root
        elif raw.startswith("/output/"):
            candidate = self.output_root / raw.removeprefix("/output/")
        elif raw.startswith("/"):
            raise ValueError("absolute paths must be under /workspace or /output")
        else:
            candidate = self.workspace_root / raw

        resolved = candidate.resolve()
        allowed_roots = (
            (self.output_root,)
            if output_only
            else (self.workspace_root, self.output_root)
        )
        if not any(self._inside(resolved, root) for root in allowed_roots):
            raise ValueError("path escapes the permitted workspace/output roots")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"path does not exist: {agent_path}")
        return resolved
