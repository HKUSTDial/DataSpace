# DataSpace

DataSpace is a benchmark for evaluating data agents that answer analytical
questions over self-contained heterogeneous workspaces and return verifiable
relations.

This repository will host the benchmark documentation, evaluation code,
reproducible baselines, experiment configurations, and analysis artifacts used
by the accompanying paper.

## Repository layout

- `evaluation/`: official Task Accuracy evaluator, configuration schema, and
  synthetic tests. Frozen gold files and per-task evaluation configurations
  are distributed with the benchmark data.
- `baseline/`: controlled DataSpace-Agent backbone experiments and the unified
  DataSpace-Agent/smolagents/Codex/Claude Code/Grok Build harness comparison,
  including the shared offline Data Workbench Runtime.

Additional release components will be added as they are finalized.

## License

The code in this repository is released under the MIT License. See
[`LICENSE`](LICENSE).
