from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), "event": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    def write_summary(self, summary: dict[str, Any]) -> None:
        target = self.run_dir / "run.json"
        temporary = self.run_dir / "run.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        temporary.replace(target)
