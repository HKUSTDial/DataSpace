#!/usr/bin/env python3
"""Official relation evaluator for DataSpace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any, Hashable


EVALUATOR_NAME = "DataSpace Evaluator"
EVALUATOR_VERSION = "1.0.0"
CONFIG_SCHEMA_VERSION = "1.0"

MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_CSV_ROWS = 100_000
MAX_CSV_COLUMNS = 64
MAX_CELL_CHARACTERS = 2 * 1024 * 1024

NON_TEXT_NULL_TOKENS = {"", "null", "none", "nan", "nat", "<na>"}
BOOLEAN_TRUE_TOKENS = {"1", "true", "yes", "y", "是"}
BOOLEAN_FALSE_TOKENS = {"0", "false", "no", "n", "否"}
THOUSANDS_NUMBER_PATTERN = re.compile(
    r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)
PLAIN_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
TASK_ID_PATTERN = re.compile(r"^task_(\d+)$")

NULL_VALUE: tuple[str] = ("null",)
CanonicalCell = Hashable
CanonicalRow = tuple[CanonicalCell, ...]


class EvaluationFatalError(RuntimeError):
    """An evaluator/config/gold problem that must abort the run."""


class PredictionFormatError(ValueError):
    """A participant-output problem that fails only the affected task."""


class CellNormalizationError(ValueError):
    """A cell cannot be interpreted under the configured target type."""


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    gold_index: int
    gold_name: str
    type: str
    comparison_mode: str | None = None
    comparison_digits: int | None = None
    allow_percent_sign: bool = False
    numeric_unit: str = "plain"


@dataclass(frozen=True, slots=True)
class TaskConfig:
    task_id: str
    order_sensitive: bool
    columns: tuple[ColumnSpec, ...]


@dataclass(frozen=True, slots=True)
class CsvTable:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class TaskEvaluationResult:
    task_id: str
    passed: bool
    score: int
    order_sensitive: bool
    gold_column_count: int
    prediction_column_count: int
    gold_row_count: int
    prediction_row_count: int
    column_mapping: list[dict[str, int]] | None
    error_code: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "score": self.score,
            "order_sensitive": self.order_sensitive,
            "gold_column_count": self.gold_column_count,
            "prediction_column_count": self.prediction_column_count,
            "gold_row_count": self.gold_row_count,
            "prediction_row_count": self.prediction_row_count,
            "column_mapping": self.column_mapping,
            "error_code": self.error_code,
            "error": self.error,
        }


def natural_task_key(task_id: str) -> int:
    match = TASK_ID_PATTERN.fullmatch(task_id)
    if not match:
        raise ValueError(f"Invalid task id: {task_id}")
    return int(match.group(1))


def read_strict_csv(path: Path, *, participant_output: bool) -> CsvTable:
    error_type = PredictionFormatError if participant_output else EvaluationFatalError

    if not path.exists():
        raise error_type(f"Missing CSV file: {path}")
    if not path.is_file():
        raise error_type(f"CSV path is not a regular file: {path}")
    if participant_output and path.is_symlink():
        raise error_type(f"Symbolic links are not allowed: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise error_type(f"Cannot stat CSV file {path}: {exc}") from exc
    if size > MAX_CSV_BYTES:
        raise error_type(
            f"CSV exceeds the {MAX_CSV_BYTES}-byte limit: {path} ({size} bytes)"
        )

    previous_field_limit = csv.field_size_limit()
    csv.field_size_limit(MAX_CELL_CHARACTERS)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            parsed_rows: list[tuple[str, ...]] = []
            for row_number, row in enumerate(reader, start=1):
                if row_number > MAX_CSV_ROWS + 1:
                    raise error_type(
                        f"CSV exceeds the {MAX_CSV_ROWS}-data-row limit: {path}"
                    )
                if any("\x00" in cell for cell in row):
                    raise error_type(
                        f"NUL byte in CSV cell at row {row_number}: {path}"
                    )
                parsed_rows.append(tuple(row))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise error_type(f"Cannot parse CSV {path}: {exc}") from exc
    finally:
        csv.field_size_limit(previous_field_limit)

    if not parsed_rows:
        raise error_type(f"CSV is empty and has no header row: {path}")
    headers = parsed_rows[0]
    if not headers:
        raise error_type(f"CSV header has zero columns: {path}")
    if len(headers) > MAX_CSV_COLUMNS:
        raise error_type(
            f"CSV exceeds the {MAX_CSV_COLUMNS}-column limit: {path}"
        )
    for row_number, row in enumerate(parsed_rows[1:], start=2):
        if len(row) != len(headers):
            raise error_type(
                f"Non-rectangular CSV {path}: row {row_number} has "
                f"{len(row)} cells; expected {len(headers)}"
            )

    return CsvTable(headers=headers, rows=tuple(parsed_rows[1:]))


def load_task_config(path: Path) -> TaskConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationFatalError(f"Cannot load task config {path}: {exc}") from exc

    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise EvaluationFatalError(
            f"Unsupported config schema in {path}: {raw.get('schema_version')!r}"
        )
    task_id = raw.get("task_id")
    if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
        raise EvaluationFatalError(f"Invalid task_id in {path}: {task_id!r}")
    if path.stem != task_id:
        raise EvaluationFatalError(
            f"Config filename/task_id mismatch: {path.name} vs {task_id}"
        )
    if not isinstance(raw.get("order_sensitive"), bool):
        raise EvaluationFatalError(f"Invalid order_sensitive in {path}")

    raw_columns = raw.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise EvaluationFatalError(f"Config has no columns: {path}")

    columns: list[ColumnSpec] = []
    for expected_index, raw_column in enumerate(raw_columns):
        if not isinstance(raw_column, dict):
            raise EvaluationFatalError(f"Invalid column config in {path}")
        if raw_column.get("gold_index") != expected_index:
            raise EvaluationFatalError(
                f"gold_index must be contiguous in {path}: expected {expected_index}"
            )
        gold_name = raw_column.get("gold_name")
        column_type = raw_column.get("type")
        if not isinstance(gold_name, str):
            raise EvaluationFatalError(f"Invalid gold_name in {path}")
        if column_type not in {"text", "number", "date", "datetime", "boolean"}:
            raise EvaluationFatalError(
                f"Unsupported column type in {path}: {column_type!r}"
            )

        comparison_mode: str | None = None
        comparison_digits: int | None = None
        if column_type == "number":
            comparison = raw_column.get("comparison")
            if not isinstance(comparison, dict):
                raise EvaluationFatalError(
                    f"Numeric column lacks comparison config in {path}"
                )
            comparison_mode = comparison.get("mode")
            if comparison_mode not in {
                "integer",
                "decimal_places",
                "significant_digits",
            }:
                raise EvaluationFatalError(
                    f"Invalid numeric comparison mode in {path}: "
                    f"{comparison_mode!r}"
                )
            if comparison_mode != "integer":
                comparison_digits = comparison.get("digits")
                if (
                    not isinstance(comparison_digits, int)
                    or comparison_digits < 0
                    or comparison_digits > 30
                ):
                    raise EvaluationFatalError(
                        f"Invalid numeric comparison digits in {path}"
                    )
            numeric_unit = raw_column.get("unit", "plain")
            if numeric_unit not in {"plain", "percentage_points", "fraction"}:
                raise EvaluationFatalError(
                    f"Invalid numeric unit in {path}: {numeric_unit!r}"
                )
        else:
            numeric_unit = "plain"

        columns.append(
            ColumnSpec(
                gold_index=expected_index,
                gold_name=gold_name,
                type=column_type,
                comparison_mode=comparison_mode,
                comparison_digits=comparison_digits,
                allow_percent_sign=bool(
                    raw_column.get("allow_percent_sign", False)
                ),
                numeric_unit=numeric_unit,
            )
        )

    return TaskConfig(
        task_id=task_id,
        order_sensitive=raw["order_sensitive"],
        columns=tuple(columns),
    )


def load_configs(config_root: Path) -> list[TaskConfig]:
    if not config_root.is_dir():
        raise EvaluationFatalError(f"Missing config directory: {config_root}")

    paths = sorted(
        config_root.glob("task_*.json"),
        key=lambda path: natural_task_key(path.stem),
    )
    if not paths:
        raise EvaluationFatalError(f"No task configs found under {config_root}")

    configs = [load_task_config(path) for path in paths]
    if len({config.task_id for config in configs}) != len(configs):
        raise EvaluationFatalError("Duplicate task_id values in evaluator configs")
    return configs


def hash_config_set(config_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        config_root.glob("task_*.json"),
        key=lambda path: natural_task_key(path.stem),
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_text(text: str) -> CanonicalCell:
    normalized = unicodedata.normalize(
        "NFC",
        text.replace("\r\n", "\n").replace("\r", "\n").strip(),
    )
    if normalized == "":
        return NULL_VALUE
    return ("text", normalized)


def parse_decimal(text: str, spec: ColumnSpec) -> Decimal | None:
    normalized = text.strip()
    if normalized.casefold() in NON_TEXT_NULL_TOKENS:
        return None

    has_percent_sign = normalized.endswith("%")
    if has_percent_sign:
        if not spec.allow_percent_sign:
            raise CellNormalizationError(
                f"Percent sign is not allowed for gold column {spec.gold_index}"
            )
        normalized = normalized[:-1].strip()

    if "," in normalized:
        if not THOUSANDS_NUMBER_PATTERN.fullmatch(normalized):
            raise CellNormalizationError(
                f"Invalid thousands separators in number: {text!r}"
            )
        normalized = normalized.replace(",", "")
    if not PLAIN_NUMBER_PATTERN.fullmatch(normalized):
        raise CellNormalizationError(f"Invalid number: {text!r}")

    try:
        value = Decimal(normalized)
    except InvalidOperation as exc:
        raise CellNormalizationError(f"Invalid number: {text!r}") from exc
    if not value.is_finite():
        raise CellNormalizationError(f"Non-finite number: {text!r}")
    if has_percent_sign and spec.numeric_unit == "fraction":
        value /= Decimal(100)
    return value


def canonical_decimal_string(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def normalize_number(text: str, spec: ColumnSpec) -> CanonicalCell:
    value = parse_decimal(text, spec)
    if value is None:
        return NULL_VALUE

    with localcontext() as context:
        context.prec = 100
        if spec.comparison_mode == "integer":
            if value != value.to_integral_value():
                raise CellNormalizationError(
                    f"Expected an integer for gold column {spec.gold_index}: {text!r}"
                )
            normalized = value.to_integral_value()
        elif spec.comparison_mode == "decimal_places":
            assert spec.comparison_digits is not None
            quantum = Decimal(1).scaleb(-spec.comparison_digits)
            normalized = value.quantize(quantum, rounding=ROUND_HALF_UP)
        elif spec.comparison_mode == "significant_digits":
            assert spec.comparison_digits is not None
            if value == 0:
                normalized = Decimal(0)
            else:
                quantum = Decimal(1).scaleb(
                    value.adjusted() - spec.comparison_digits + 1
                )
                normalized = value.quantize(quantum, rounding=ROUND_HALF_UP)
        else:
            raise EvaluationFatalError(
                f"Missing numeric comparison mode for column {spec.gold_index}"
            )
    return ("number", canonical_decimal_string(normalized))


def parse_iso_datetime(text: str) -> datetime:
    candidate = text.strip().replace("Z", "+00:00")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
            return datetime.combine(date.fromisoformat(candidate), time.min)
        return datetime.fromisoformat(candidate)
    except (ValueError, OverflowError) as exc:
        raise CellNormalizationError(f"Invalid ISO date/datetime: {text!r}") from exc


def normalize_date(text: str) -> CanonicalCell:
    normalized = text.strip()
    if normalized.casefold() in NON_TEXT_NULL_TOKENS:
        return NULL_VALUE
    parsed = parse_iso_datetime(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if parsed.time() != time.min:
        raise CellNormalizationError(
            f"Expected a date or midnight datetime, got: {text!r}"
        )
    return ("date", parsed.date().isoformat())


def normalize_datetime(text: str) -> CanonicalCell:
    normalized = text.strip()
    if normalized.casefold() in NON_TEXT_NULL_TOKENS:
        return NULL_VALUE
    parsed = parse_iso_datetime(normalized)
    if parsed.tzinfo is not None:
        rendered = (
            parsed.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    else:
        rendered = parsed.isoformat()
    return ("datetime", rendered)


def normalize_boolean(text: str) -> CanonicalCell:
    normalized = text.strip().casefold()
    if normalized in NON_TEXT_NULL_TOKENS:
        return NULL_VALUE
    if normalized in BOOLEAN_TRUE_TOKENS:
        return ("boolean", True)
    if normalized in BOOLEAN_FALSE_TOKENS:
        return ("boolean", False)
    raise CellNormalizationError(f"Invalid boolean: {text!r}")


def normalize_cell(text: str, spec: ColumnSpec) -> CanonicalCell:
    if spec.type == "text":
        return normalize_text(text)
    if spec.type == "number":
        return normalize_number(text, spec)
    if spec.type == "date":
        return normalize_date(text)
    if spec.type == "datetime":
        return normalize_datetime(text)
    if spec.type == "boolean":
        return normalize_boolean(text)
    raise EvaluationFatalError(f"Unsupported type in config: {spec.type}")


def normalize_gold_rows(
    table: CsvTable,
    config: TaskConfig,
) -> tuple[CanonicalRow, ...]:
    if len(table.headers) != len(config.columns):
        raise EvaluationFatalError(
            f"Gold/config column-count mismatch for {config.task_id}: "
            f"{len(table.headers)} vs {len(config.columns)}"
        )
    try:
        return tuple(
            tuple(
                normalize_cell(row[column.gold_index], column)
                for column in config.columns
            )
            for row in table.rows
        )
    except CellNormalizationError as exc:
        raise EvaluationFatalError(
            f"Gold normalization failed for {config.task_id}: {exc}"
        ) from exc


def precompute_prediction_columns(
    table: CsvTable,
    config: TaskConfig,
) -> list[list[tuple[CanonicalCell, ...] | None]]:
    matrix: list[list[tuple[CanonicalCell, ...] | None]] = []
    for spec in config.columns:
        normalized_for_spec: list[tuple[CanonicalCell, ...] | None] = []
        for prediction_index in range(len(table.headers)):
            try:
                column = tuple(
                    normalize_cell(row[prediction_index], spec)
                    for row in table.rows
                )
            except CellNormalizationError:
                column = None
            normalized_for_spec.append(column)
        matrix.append(normalized_for_spec)
    return matrix


def rows_for_permutation(
    matrix: list[list[tuple[CanonicalCell, ...] | None]],
    permutation: tuple[int, ...],
    row_count: int,
) -> tuple[CanonicalRow, ...] | None:
    selected_columns: list[tuple[CanonicalCell, ...]] = []
    for gold_index, prediction_index in enumerate(permutation):
        column = matrix[gold_index][prediction_index]
        if column is None:
            return None
        selected_columns.append(column)
    return tuple(
        tuple(column[row_index] for column in selected_columns)
        for row_index in range(row_count)
    )


def relations_equal(
    gold_rows: tuple[CanonicalRow, ...],
    prediction_rows: tuple[CanonicalRow, ...],
    *,
    order_sensitive: bool,
) -> bool:
    if order_sensitive:
        return gold_rows == prediction_rows
    return Counter(gold_rows) == Counter(prediction_rows)


def failed_result(
    *,
    config: TaskConfig,
    gold_columns: int,
    prediction_columns: int,
    gold_rows: int,
    prediction_rows: int,
    error_code: str,
    error: str,
) -> TaskEvaluationResult:
    return TaskEvaluationResult(
        task_id=config.task_id,
        passed=False,
        score=0,
        order_sensitive=config.order_sensitive,
        gold_column_count=gold_columns,
        prediction_column_count=prediction_columns,
        gold_row_count=gold_rows,
        prediction_row_count=prediction_rows,
        column_mapping=None,
        error_code=error_code,
        error=error,
    )


def evaluate_task(
    *,
    config: TaskConfig,
    prediction_path: Path,
    gold_path: Path,
) -> TaskEvaluationResult:
    gold_table = read_strict_csv(gold_path, participant_output=False)
    gold_rows = normalize_gold_rows(gold_table, config)

    if not prediction_path.exists():
        return failed_result(
            config=config,
            gold_columns=len(config.columns),
            prediction_columns=0,
            gold_rows=len(gold_rows),
            prediction_rows=0,
            error_code="missing_prediction",
            error=f"Missing prediction CSV: {prediction_path}",
        )

    try:
        prediction_table = read_strict_csv(
            prediction_path,
            participant_output=True,
        )
    except PredictionFormatError as exc:
        return failed_result(
            config=config,
            gold_columns=len(gold_table.headers),
            prediction_columns=0,
            gold_rows=len(gold_table.rows),
            prediction_rows=0,
            error_code="invalid_prediction_csv",
            error=str(exc),
        )

    if len(prediction_table.headers) != len(config.columns):
        return failed_result(
            config=config,
            gold_columns=len(config.columns),
            prediction_columns=len(prediction_table.headers),
            gold_rows=len(gold_rows),
            prediction_rows=len(prediction_table.rows),
            error_code="column_count_mismatch",
            error=(
                f"Expected {len(config.columns)} columns; "
                f"received {len(prediction_table.headers)}"
            ),
        )
    if len(prediction_table.rows) != len(gold_rows):
        return failed_result(
            config=config,
            gold_columns=len(config.columns),
            prediction_columns=len(prediction_table.headers),
            gold_rows=len(gold_rows),
            prediction_rows=len(prediction_table.rows),
            error_code="row_count_mismatch",
            error=(
                f"Expected {len(gold_rows)} data rows; "
                f"received {len(prediction_table.rows)}"
            ),
        )

    matrix = precompute_prediction_columns(prediction_table, config)
    valid_permutation_count = 0
    for permutation in itertools.permutations(range(len(config.columns))):
        prediction_rows = rows_for_permutation(
            matrix,
            permutation,
            len(prediction_table.rows),
        )
        if prediction_rows is None:
            continue
        valid_permutation_count += 1
        if relations_equal(
            gold_rows,
            prediction_rows,
            order_sensitive=config.order_sensitive,
        ):
            mapping = [
                {
                    "gold_index": gold_index,
                    "prediction_index": prediction_index,
                }
                for gold_index, prediction_index in enumerate(permutation)
            ]
            return TaskEvaluationResult(
                task_id=config.task_id,
                passed=True,
                score=1,
                order_sensitive=config.order_sensitive,
                gold_column_count=len(config.columns),
                prediction_column_count=len(prediction_table.headers),
                gold_row_count=len(gold_rows),
                prediction_row_count=len(prediction_table.rows),
                column_mapping=mapping,
                error_code=None,
                error=None,
            )

    if valid_permutation_count == 0:
        error_code = "type_normalization_failed"
        error = "No one-to-one column mapping is compatible with the configured types"
    else:
        error_code = "relation_mismatch"
        relation_kind = (
            "ordered row sequence"
            if config.order_sensitive
            else "unordered row multiset"
        )
        error = f"Prediction does not equal the gold {relation_kind}"
    return failed_result(
        config=config,
        gold_columns=len(config.columns),
        prediction_columns=len(prediction_table.headers),
        gold_rows=len(gold_rows),
        prediction_rows=len(prediction_table.rows),
        error_code=error_code,
        error=error,
    )


def discover_unexpected_prediction_tasks(
    prediction_root: Path,
    expected_task_ids: set[str],
) -> list[str]:
    unexpected: list[str] = []
    for path in prediction_root.iterdir():
        if (
            path.is_dir()
            and TASK_ID_PATTERN.fullmatch(path.name)
            and path.name not in expected_task_ids
        ):
            unexpected.append(path.name)
    return sorted(unexpected, key=natural_task_key)


def evaluate_suite(
    *,
    prediction_root: Path,
    gold_root: Path,
    config_root: Path,
) -> dict[str, Any]:
    if not prediction_root.is_dir():
        raise EvaluationFatalError(
            f"Missing prediction root directory: {prediction_root}"
        )
    if not gold_root.is_dir():
        raise EvaluationFatalError(f"Missing gold root directory: {gold_root}")

    configs = load_configs(config_root)
    all_expected_task_ids = {
        path.stem
        for path in config_root.glob("task_*.json")
        if TASK_ID_PATTERN.fullmatch(path.stem)
    }
    results = [
        evaluate_task(
            config=config,
            prediction_path=(
                prediction_root / config.task_id / "prediction.csv"
            ),
            gold_path=gold_root / config.task_id / "gold.csv",
        )
        for config in configs
    ]

    passed_task_count = sum(result.passed for result in results)
    task_count = len(results)
    return {
        "evaluator": EVALUATOR_NAME,
        "evaluator_version": EVALUATOR_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "config_set_sha256": hash_config_set(config_root),
        "metric": "task_accuracy",
        "task_count": task_count,
        "passed_task_count": passed_task_count,
        "failed_task_count": task_count - passed_task_count,
        "task_accuracy": passed_task_count / task_count if task_count else 0.0,
        "unexpected_prediction_tasks": discover_unexpected_prediction_tasks(
            prediction_root,
            all_expected_task_ids,
        ),
        "tasks": [result.to_dict() for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EVALUATOR_NAME)
    parser.add_argument(
        "--prediction-root",
        required=True,
        type=Path,
        help="Directory containing task_N/prediction.csv files.",
    )
    parser.add_argument(
        "--gold-root",
        required=True,
        type=Path,
        help="Directory containing task_N/gold.csv files.",
    )
    parser.add_argument(
        "--config-root",
        required=True,
        type=Path,
        help="Frozen per-task config directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output evaluation_summary.json path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = evaluate_suite(
            prediction_root=args.prediction_root.resolve(),
            gold_root=args.gold_root.resolve(),
            config_root=args.config_root.resolve(),
        )
    except EvaluationFatalError as exc:
        print(f"Evaluator fatal error: {exc}", file=sys.stderr)
        return 2

    output_path = args.output
    if output_path is None:
        output_path = args.prediction_root.resolve() / "evaluation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    console_summary = {
        key: value for key, value in summary.items() if key != "tasks"
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
