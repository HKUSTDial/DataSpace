from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any


def validate_prediction(path: Path) -> dict[str, Any]:
    """Validate only the released CSV contract, never answer correctness."""

    if not path.is_file():
        raise ValueError("submitted path is not a file")
    byte_size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)

    csv.field_size_limit(min(sys.maxsize, 128 * 1024 * 1024))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("prediction CSV is empty") from exc
        width = len(header)
        if width == 0:
            raise ValueError("prediction CSV has no columns")
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            if len(row) != width:
                raise ValueError(
                    f"row {row_number} has {len(row)} columns; expected {width}"
                )
            row_count += 1
    return {
        "columns": width,
        "data_rows": row_count,
        "bytes": byte_size,
        "sha256": digest.hexdigest(),
    }
