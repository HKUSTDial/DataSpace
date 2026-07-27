# DataSpace evaluation

This directory contains the official DataSpace evaluator. It compares a
predicted CSV with the corresponding gold tabular result under a frozen
per-task configuration. The evaluator uses only the Python standard library.

The benchmark release, rather than this code repository, is the source of
truth for official gold files and task configurations.

## Inputs

The evaluator reads three explicit directory roots:

```text
predictions/
└── task_N/
    └── prediction.csv

benchmark/output/
└── task_N/
    └── gold.csv

benchmark/evaluation/configs/
└── task_N.json
```

Run:

```bash
python3 evaluation/evaluate.py \
  --prediction-root /path/to/predictions \
  --gold-root /path/to/benchmark/output \
  --config-root /path/to/benchmark/evaluation/configs \
  --output /path/to/evaluation_summary.json
```

If `--output` is omitted, the summary is written to
`PREDICTION_ROOT/evaluation_summary.json`.

## Metric

The official metric is **Task Accuracy**:

```text
Task Accuracy = correct tasks / evaluated tasks
```

Each task receives a binary score. A task is correct only when one
one-to-one prediction-to-gold column mapping makes the complete predicted
table equal to the gold table under the task configuration.

Prediction header text and column order are not scored. Column alignment
preserves complete row associations. Unordered tasks compare row multisets,
including duplicate multiplicity; order-sensitive tasks compare the exact row
sequence.

## Task configurations

Each `task_N.json` specifies:

- the task identifier;
- whether row order is semantically significant;
- the type of each gold column;
- numeric precision and unit rules where applicable.

The evaluator supports text, number, date, datetime, and Boolean columns.
The JSON Schema is available at
[`schema/task_config.schema.json`](schema/task_config.schema.json).

Invalid official configs or gold files abort evaluation. Missing or malformed
participant predictions fail only the affected task. Every summary includes a
SHA-256 digest of the complete config set.

## Tests

Run the synthetic unit and CLI tests without benchmark data or model APIs:

```bash
python3 -m unittest discover -s evaluation/tests -v
```
