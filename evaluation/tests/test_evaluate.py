from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("dataspace_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def text_column(index: int, name: str = "text") -> evaluator.ColumnSpec:
    return evaluator.ColumnSpec(
        gold_index=index,
        gold_name=name,
        type="text",
    )


def number_column(
    index: int,
    *,
    mode: str = "significant_digits",
    digits: int | None = 12,
    allow_percent_sign: bool = False,
    numeric_unit: str = "plain",
) -> evaluator.ColumnSpec:
    return evaluator.ColumnSpec(
        gold_index=index,
        gold_name="number",
        type="number",
        comparison_mode=mode,
        comparison_digits=digits,
        allow_percent_sign=allow_percent_sign,
        numeric_unit=numeric_unit,
    )


class EvaluatorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_csv(
        self,
        relative_path: str,
        rows: list[list[str]],
    ) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        return path

    def evaluate(
        self,
        *,
        gold: list[list[str]],
        prediction: list[list[str]],
        columns: tuple[evaluator.ColumnSpec, ...],
        order_sensitive: bool = False,
    ) -> evaluator.TaskEvaluationResult:
        gold_path = self.write_csv("gold/task_1/gold.csv", gold)
        prediction_path = self.write_csv(
            "prediction/task_1/prediction.csv",
            prediction,
        )
        config = evaluator.TaskConfig(
            task_id="task_1",
            order_sensitive=order_sensitive,
            columns=columns,
        )
        return evaluator.evaluate_task(
            config=config,
            prediction_path=prediction_path,
            gold_path=gold_path,
        )

    def test_headers_and_column_order_are_ignored(self) -> None:
        result = self.evaluate(
            gold=[
                ["company", "score"],
                ["A", "1"],
                ["B", "2"],
            ],
            prediction=[
                ["anything", "different"],
                ["2.0", "B"],
                ["1.00", "A"],
            ],
            columns=(text_column(0), number_column(1)),
        )
        self.assertTrue(result.passed)
        self.assertEqual(
            result.column_mapping,
            [
                {"gold_index": 0, "prediction_index": 1},
                {"gold_index": 1, "prediction_index": 0},
            ],
        )

    def test_cross_column_row_association_is_preserved(self) -> None:
        result = self.evaluate(
            gold=[
                ["company", "score"],
                ["A", "1"],
                ["B", "2"],
            ],
            prediction=[
                ["x", "y"],
                ["A", "2"],
                ["B", "1"],
            ],
            columns=(text_column(0), number_column(1)),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "relation_mismatch")

    def test_unordered_tasks_use_row_multiset_semantics(self) -> None:
        result = self.evaluate(
            gold=[
                ["company", "score"],
                ["A", "1"],
                ["A", "1"],
                ["B", "2"],
            ],
            prediction=[
                ["x", "y"],
                ["B", "2"],
                ["A", "1"],
                ["A", "1"],
            ],
            columns=(text_column(0), number_column(1)),
        )
        self.assertTrue(result.passed)

    def test_duplicate_row_multiplicity_is_required(self) -> None:
        result = self.evaluate(
            gold=[
                ["company", "score"],
                ["A", "1"],
                ["A", "1"],
            ],
            prediction=[
                ["x", "y"],
                ["A", "1"],
            ],
            columns=(text_column(0), number_column(1)),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "row_count_mismatch")

    def test_order_sensitive_task_checks_row_sequence(self) -> None:
        result = self.evaluate(
            gold=[
                ["rank", "company"],
                ["1", "A"],
                ["2", "B"],
            ],
            prediction=[
                ["rank", "company"],
                ["2", "B"],
                ["1", "A"],
            ],
            columns=(number_column(0, mode="integer", digits=None), text_column(1)),
            order_sensitive=True,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "relation_mismatch")

    def test_text_preserves_leading_zeroes(self) -> None:
        result = self.evaluate(
            gold=[["stock_code"], ["000001"]],
            prediction=[["anything"], ["1"]],
            columns=(text_column(0),),
        )
        self.assertFalse(result.passed)

    def test_significant_digits_remove_binary_float_noise(self) -> None:
        result = self.evaluate(
            gold=[["amount"], ["1443094.0000000002"]],
            prediction=[["amount"], ["1443094"]],
            columns=(number_column(0),),
        )
        self.assertTrue(result.passed)

    def test_nonstandard_numeric_syntax_is_rejected(self) -> None:
        result = self.evaluate(
            gold=[["amount"], ["1000"]],
            prediction=[["amount"], ["1_000"]],
            columns=(number_column(0),),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "type_normalization_failed")

    def test_invalid_prediction_date_fails_only_the_task(self) -> None:
        date_column = evaluator.ColumnSpec(
            gold_index=0,
            gold_name="report_date",
            type="date",
        )
        result = self.evaluate(
            gold=[["report_date"], ["2003-11-30"]],
            prediction=[["date"], ["2003-11-31"]],
            columns=(date_column,),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "type_normalization_failed")

    def test_configured_decimal_places_are_applied(self) -> None:
        result = self.evaluate(
            gold=[["average"], ["6.04"]],
            prediction=[["average"], ["6.039"]],
            columns=(
                number_column(0, mode="decimal_places", digits=2),
            ),
        )
        self.assertTrue(result.passed)

    def test_percentage_sign_is_optional_when_configured(self) -> None:
        result = self.evaluate(
            gold=[["rate"], ["15%"]],
            prediction=[["rate"], ["15.0"]],
            columns=(number_column(0, allow_percent_sign=True),),
        )
        self.assertTrue(result.passed)

    def test_fraction_column_converts_percent_sign(self) -> None:
        result = self.evaluate(
            gold=[["ratio"], ["0.0155"]],
            prediction=[["ratio"], ["1.55%"]],
            columns=(
                number_column(
                    0,
                    allow_percent_sign=True,
                    numeric_unit="fraction",
                ),
            ),
        )
        self.assertTrue(result.passed)

    def test_extra_columns_fail(self) -> None:
        result = self.evaluate(
            gold=[["company"], ["A"]],
            prediction=[["company", "extra"], ["A", "hidden"]],
            columns=(text_column(0),),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "column_count_mismatch")

    def test_ragged_prediction_fails(self) -> None:
        gold_path = self.write_csv(
            "gold/task_1/gold.csv",
            [["company", "score"], ["A", "1"]],
        )
        prediction_path = self.root / "prediction/task_1/prediction.csv"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text(
            "company,score\nA,1,unexpected\n",
            encoding="utf-8",
        )
        config = evaluator.TaskConfig(
            task_id="task_1",
            order_sensitive=False,
            columns=(text_column(0), number_column(1)),
        )
        result = evaluator.evaluate_task(
            config=config,
            prediction_path=prediction_path,
            gold_path=gold_path,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "invalid_prediction_csv")

    def test_boolean_equivalent_tokens_are_accepted(self) -> None:
        boolean_column = evaluator.ColumnSpec(
            gold_index=0,
            gold_name="has_result",
            type="boolean",
        )
        result = self.evaluate(
            gold=[["has_result"], ["1"]],
            prediction=[["answer"], ["yes"]],
            columns=(boolean_column,),
        )
        self.assertTrue(result.passed)

    def test_missing_prediction_has_specific_error(self) -> None:
        gold_path = self.write_csv(
            "gold/task_1/gold.csv",
            [["company"], ["A"]],
        )
        config = evaluator.TaskConfig(
            task_id="task_1",
            order_sensitive=False,
            columns=(text_column(0),),
        )
        result = evaluator.evaluate_task(
            config=config,
            prediction_path=self.root / "prediction/task_1/prediction.csv",
            gold_path=gold_path,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "missing_prediction")


if __name__ == "__main__":
    unittest.main()
