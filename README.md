<div align="center">

# DataSpace

### Benchmarking Data Agents for Verifiable Analytics over Heterogeneous Workspaces

<p>
  <a href="https://arxiv.org/abs/2608.03451">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2608.03451-B31B1B?logo=arxiv&logoColor=white" alt="DataSpace paper on arXiv">
  </a>
  <a href="https://huggingface.co/datasets/HKUSTDial/DataSpace">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=111" alt="Hugging Face dataset">
  </a>
  <a href="https://dataspace-bench.github.io/">
    <img src="https://img.shields.io/badge/Leaderboard-View%20Results-216D68" alt="DataSpace leaderboard">
  </a>
  <a href="https://dataagent.top">
    <img src="https://img.shields.io/badge/KDD%20Cup%202026-Official%20Benchmark-8B5CF6" alt="KDD Cup 2026 official benchmark">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-22C55E" alt="MIT License">
  </a>
</p>

**[Paper](https://arxiv.org/abs/2608.03451) ·
[Dataset](https://huggingface.co/datasets/HKUSTDial/DataSpace) ·
[Leaderboard](https://dataspace-bench.github.io/) ·
[Evaluator](evaluation/) ·
[Baselines](baseline/) ·
[KDD Cup 2026](https://dataagent.top)**

</div>

---

DataSpace evaluates data agents on analytical questions over self-contained,
task-local workspaces. A workspace may combine CSV, JSON, SQLite, Markdown,
PDF, and video artifacts. The agent must discover and integrate the relevant
evidence, then return the complete result as a verifiable table.

DataSpace is also the official benchmark of the
[KDD Cup 2026 Data Agent Track](https://dataagent.top).
The accompanying paper is available on
[arXiv](https://arxiv.org/abs/2608.03451).

## At a glance

| | |
|---|---|
| **410 tasks** | All task questions and workspaces are publicly available |
| **6 modalities** | CSV, JSON, SQLite, Markdown, PDF, and video |
| **Cross-language workspaces** | Chinese and English may occur across both questions and data artifacts |
| **Exact, semantic evaluation** | Header-invariant column alignment with type-aware value and row comparison |
| **60 public references** | Gold results and frozen configs for local end-to-end evaluation |
| **15.0 GB of context** | 7,439 files spanning structured data, long documents, and multimedia |

Tasks cover evidence discovery, structured extraction, joins across
representations, filtering, aggregation, ranking, temporal reasoning, and
document or video understanding.

## Dataset

The complete release is hosted on
[Hugging Face](https://huggingface.co/datasets/HKUSTDial/DataSpace). Install
the Hub client and download the versioned archive:

```bash
pip install -U huggingface_hub

hf download HKUSTDial/DataSpace \
  release/DataSpace-Benchmark.zip \
  release/DataSpace-Benchmark.zip.sha256 \
  --repo-type dataset \
  --local-dir .
```

Verify the archive before extraction:

```bash
cd release
sha256sum --check DataSpace-Benchmark.zip.sha256
unzip DataSpace-Benchmark.zip
```

Every task exposes one question and one heterogeneous workspace:

```text
DataSpace-Benchmark/
├── input/
│   └── task_N/
│       ├── task.json
│       └── context/
├── output/                         # gold.csv for 60 public-reference tasks
│   └── task_N/gold.csv
└── evaluation/
    └── configs/task_N.json         # frozen configs for the same 60 tasks
```

An agent should write one prediction per task:

```text
predictions/
└── task_N/
    └── prediction.csv
```

All 410 inputs are public. The release includes reference answers and
evaluation configurations for 60 representative tasks; the remaining 350
references are withheld for official full-benchmark evaluation.

## Evaluation

The official evaluator uses only the Python standard library:

```bash
python3 evaluation/evaluate.py \
  --prediction-root /path/to/predictions \
  --gold-root /path/to/DataSpace-Benchmark/output \
  --config-root /path/to/DataSpace-Benchmark/evaluation/configs \
  --output /path/to/evaluation_summary.json
```

The primary metric is **Task Accuracy**. A task is correct only when the
predicted table matches the complete gold result under its frozen
configuration. Prediction headers and column order are not scored; values are
compared using task-specific types, numeric precision, units, and row-order
semantics.

See [`evaluation/README.md`](evaluation/README.md) for the complete protocol,
configuration schema, and test suite.

## Baselines

[`baseline/`](baseline/) contains:

- **DataSpace-Agent**, a controlled ReAct-style agent with three generic tools;
- a six-backbone comparison under the same agent implementation;
- a unified harness comparison across DataSpace-Agent, Smolagents, Codex,
  Claude Code, and Grok Build;
- the shared offline **Data Workbench Runtime** used for fair local execution.

Install the baseline package:

```bash
cd baseline
python3 -m venv .venv
.venv/bin/pip install -e .
```

The baseline documentation covers runtime construction, model configuration,
single-task and full-benchmark execution, sandboxing, resume semantics, and
reproducibility checks. See [`baseline/README.md`](baseline/README.md).

## Repository structure

```text
.
├── evaluation/     # official Task Accuracy evaluator and synthetic tests
├── baseline/       # agents, harness adapters, runtime, configs, and launchers
├── LICENSE
└── README.md
```

The benchmark data, public reference results, and frozen task configurations
are distributed separately through
[Hugging Face](https://huggingface.co/datasets/HKUSTDial/DataSpace).

## License

The code and dataset are released under the [MIT License](LICENSE).

## Citation

```bibtex
@article{li2026dataspace,
  title        = {DataSpace: Benchmarking Data Agents for Verifiable Analytics
                  over Heterogeneous Workspaces},
  author       = {Boyan Li and Zhuowen Liang and Yupeng Xie and Xiaotian Lin and
                  Tianqi Luo and Xinyu Liu and Yizhang Zhu and Zhangyang Peng and
                  Yuan Li and Zhengxuan Zhang and Jiayi Zhang and Nan Tang and
                  Guoliang Li and Yuyu Luo},
  journal      = {arXiv preprint arXiv:2608.03451},
  year         = {2026},
  url          = {https://arxiv.org/abs/2608.03451}
}
```
