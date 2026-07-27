# DataSpace baselines

This package contains two reproducible evaluation protocols:

1. **Controlled backbone comparison.** DataSpace-Agent gives every model the
   same terminal-style ReAct loop and the same three low-level tools.
2. **Harness comparison.** DataSpace-Agent, smolagents, Codex, Claude Code, and
   Grok Build use the same configuration-selected backbone, task inputs,
   output contract, wall-clock limit, and Data Workbench Runtime, while
   retaining their native planning and tool-use behavior.

The legacy `dataspace-agent` command remains available. Its prompt, model loop,
tool semantics, prediction validation, and run artifacts remain compatible;
its Docker sandbox now uses the shared Data Workbench Runtime. Common task,
result, and batch utilities are shared with the `dataspace-baselines` runner.

## Installation

Python 3.10 or newer, Docker, Bubblewrap, and `socat` are required. The latter
two support the fail-closed native CLI sandboxes on Linux. Install the package
in an isolated environment:

```bash
cd baseline
python3 -m venv .venv
.venv/bin/pip install -e .
```

Build the shared workbench and smolagents runtime images:

```bash
./scripts/build_harness_images.sh
```

The command builds `dataspace-workbench:1.0`, builds the smolagents executor on
top of it, and exports the workbench as
`.runtimes/data-workbench-1.0/rootfs`. Native harnesses mount this rootfs
read-only. Matching local artifacts are reused by default; pass `--rebuild`
when an intentional replacement is required. Sites using package mirrors can
set `DATASPACE_PYTHON_IMAGE`, `DATASPACE_DEBIAN_MIRROR`,
`DATASPACE_DEBIAN_SECURITY_MIRROR`, and `DATASPACE_PIP_INDEX_URL` for the build.

Install all pinned native harness runtimes:

```bash
./scripts/install_harness_runtimes.sh
```

The harness experiment uses these pinned project-local runtimes:

- smolagents 1.26.0, installed by the Python package;
- Codex 0.145.0;
- Claude Code 2.1.217;
- Grok 0.2.106.

`configs/harness.yaml` is the single version source of truth for the workbench,
installer, runtime paths, and doctor. Generated root filesystems and native
runtimes live under ignored `.runtimes/` directories and are mounted read-only;
the user's global Codex, Claude Code, and Grok installations are neither
modified nor used. Codex uses the Responses transport supported by the
checked-in MiMo/Vercel configuration.

Set the gateway credential in the shell that launches the run:

```bash
export AI_GATEWAY_API_KEY=...
```

Validate all dependencies, images, versions, and the filesystem jail before a
paid experiment:

```bash
.venv/bin/dataspace-baselines doctor \
  --config configs/harness.yaml
```

## Harness comparison

Each comparison uses the model selected under `experiment.model` in the
harness config and one common limit. The checked-in default is:

- backbone: `xiaomi/mimo-v2.5` through Vercel AI Gateway;
- wall-clock limit: 1,800 seconds per task;
- fresh harness session and task-local home directory for every task;
- rectangular UTF-8 `prediction.csv` as the only accepted answer;
- no reference trajectory or shared action budget;
- one shared set of generic local data-processing utilities, exposed through
  each harness's native tool interface;
- no Internet tools or network access from model-generated commands.

The model is not hard-coded in the runner. To use another backbone, edit the
`key`, `model`, endpoint, credential-variable, and output-token fields in
`configs/harness.yaml`, or copy that file and select the copy with
`--config /path/to/harness-other-model.yaml`. All harnesses launched in that
comparison receive the selected model, and `key` determines the default output
directory name. Set `context_window` to the selected provider model's actual
limit; the native CLI adapters use it to configure proactive context
compaction. For native harnesses with internal model roles, the adapter
also maps subagents, context compaction, summaries, classifiers, and media
description to that same backbone and disables model fallback. A completed
task is rejected if its available trace reports another model.

The available harness keys are:

```bash
.venv/bin/dataspace-baselines list \
  --config configs/harness.yaml
```

Run one task:

```bash
.venv/bin/dataspace-baselines run \
  --harness codex \
  --task-dir /path/to/dataspace/input/task_1 \
  --config configs/harness.yaml \
  --run-dir /path/to/runs/codex/task_1/run_1
```

Run all tasks in the foreground:

```bash
.venv/bin/dataspace-baselines batch \
  --harness codex \
  --input-root /path/to/dataspace/input \
  --config configs/harness.yaml \
  --run-root /path/to/runs/codex/run_1 \
  --concurrency 8
```

The unified launcher starts the batch command in tmux and closes the session
automatically when it finishes:

```bash
./scripts/run_benchmark.sh harness codex \
  --input-root /path/to/dataspace/input
./scripts/run_benchmark.sh harness claude-code \
  --input-root /path/to/dataspace/input --resume
./scripts/run_benchmark.sh harness smolagents \
  --input-root /path/to/dataspace/input --foreground
```

`run_harness_benchmark.sh` remains as a compatibility wrapper around the same
implementation.

The launcher requires an explicit `--input-root` and writes by default to
`runs/harness-comparison/MODEL_KEY/HARNESS/workbench-1.0`, where `MODEL_KEY`
comes from the selected config (and is `mimo-v2.5` in the checked-in default).
This default intentionally separates unified-runtime results from historical
pre-upgrade harness attempts. Use `--run-root` to select another output
location.

### Fairness and isolation boundary

All five harnesses execute data-processing code against Data Workbench Runtime
1.0. It provides the same generic CSV/JSON/SQLite, PDF, image/OCR, and video
utilities documented at `/opt/dataspace/TOOLS.md`; it contains no retrieval,
schema linking, document QA, video QA, NL2SQL, or benchmark-specific solver.
DataSpace-Agent runs the workbench image directly, smolagents derives its
executor image from it, and native CLIs use the exported read-only rootfs.
Harness-specific tool APIs, planners, memory, and command strategies remain
unchanged.

Each native CLI runs inside an outer Bubblewrap jail. It sees only a read-only
`/mnt/workspace`, writable `/mnt/output`, and fresh `/mnt/home`. Benchmark gold,
evaluation configuration, sibling tasks, the user's home, and persistent
harness memory are not visible. Codex, Claude Code, and Grok Build then apply
their own command sandbox inside that jail.

Search and command networking are disabled independently:

| Harness | Search control | Generated-code network control |
|---|---|---|
| DataSpace-Agent | no search tool | Docker network disabled |
| smolagents | no search tool | internal Docker network |
| Codex | `web_search = "disabled"` | workspace sandbox with network disabled |
| Claude Code | WebSearch/WebFetch denied | fail-closed sandbox with no allowed domains |
| Grok Build | `--disable-web-search` | strict native sandbox with read-only `/opt/dataspace` |

The host-side harness process retains only the network path needed to call the
configured model endpoint. API credentials are passed to that control process,
never to the task execution container. Raw stdout, stderr, harness traces, run
metadata, and the validated prediction are retained for auditing.

Native stdout remains harness-specific: Codex emits item lifecycle events,
Claude Code emits assistant/tool/result events, and Grok Build streams text and
thought chunks followed by an `end` event while recording tool executions in
its task-local unified log. `run.json` normalizes turns, tool actions, token
usage, status, and wall time without rewriting those raw audit traces.
For Grok Build through Vercel, a task-local relay maps only the gateway's
non-standard Chat Completions `finish_reason="other"` label to `"stop"` so the
pinned CLI can deserialize the completed turn. It does not alter message
content, reasoning, tool calls, prompts, model routing, or task execution.

### Output and resume behavior

A successful attempt contains:

```text
run_1/
├── events.jsonl
├── run.json
├── stdout.log                 # native harnesses
├── stderr.log                 # native harnesses
├── runtime/                   # task-local harness state and trace
└── output/
    └── prediction.csv
```

The batch directory additionally contains `batch_summary.json`, per-task
attempts under `tasks/`, and evaluator-ready copies under `predictions/`.
Only a cleanly exited harness that passes its execution-policy audit and
produces the exact file `output/prediction.csv` as a valid rectangular CSV
receives `submitted` status. A differently named CSV is reported but never
accepted. A file left behind after timeout, policy violation, or runtime
failure is a draft and is never copied into `predictions/`.

With `--resume`, finalized attempts are skipped. Infrastructure failures and
interrupted attempts receive a new `run_N` directory. Model output exhaustion,
context-window exhaustion, wall-time exhaustion, missing predictions, and
invalid predictions are finalized task outcomes and are not retried. Each
batch root
also stores `experiment.json`, whose fingerprint covers the model, harness,
limits, and workbench image. A run created by another configuration, or a
legacy run without this identity, cannot be resumed into the same directory;
select a new `--run-id` instead.

`Ctrl-C`, `SIGTERM`, and tmux termination cancel active model processes and
sandboxes before the batch exits. Interrupted attempts are recorded as
`stopped` and are retried by the next `--resume` run.

## Controlled DataSpace-Agent baseline

DataSpace-Agent exposes exactly three task-agnostic tools:

1. `bash(command, timeout)` executes commands in an isolated container;
2. `view_image(path)` sends one selected image to the multimodal backbone;
3. `submit_answer(path)` validates and submits a rectangular CSV.

Its sandbox provides ordinary local utilities for CSV, JSON, SQLite, Markdown,
PDF, image, and video inspection. It contains no semantic retrieval, schema
linking, document QA, video QA, NL2SQL, modality routing, or automatic evidence
selection. The workspace is read-only, `/output` is writable, the root
filesystem is read-only, and the container has no network or API credential.
The installed utilities and canonical alternatives are listed inside the
runtime at `/opt/dataspace/TOOLS.md`; package installation is disabled.

The existing single-task interface is unchanged:

```bash
.venv/bin/dataspace-agent run \
  --task-dir /path/to/dataspace/input/task_1 \
  --config configs/vercel.yaml \
  --model mimo-v2.5 \
  --run-dir /path/to/runs/mimo-v2.5/task_1/run_1
```

The existing full-benchmark interface and launcher are also unchanged:

```bash
.venv/bin/dataspace-agent batch \
  --input-root /path/to/dataspace/input \
  --config configs/vercel.yaml \
  --model mimo-v2.5 \
  --run-root /path/to/runs/mimo-v2.5/run_1 \
  --concurrency 8

./scripts/run_benchmark.sh model mimo-v2.5 \
  --input-root /path/to/dataspace/input --resume
```

The previous `run_full_benchmark.sh MODEL ...` command remains available as a
compatibility wrapper and produces the same run-directory layout.

Within the harness comparison only, DataSpace-Agent's turn and action ceilings
are set above any reachable count so that the shared wall-clock deadline is the
active termination condition. The DataSpace-Agent prompt, tools, observation
format, model backend, and Docker execution path remain the same.

Controlled six-backbone runs produced with the predecessor DataSpace-Agent
image remain a valid within-agent comparison because every backbone used the
same prompt, tools, and image. Harness-comparison runs use the unified 1.0
runtime and should not mix attempts created before and after that upgrade.

## Task contract

The runner expects released tasks in this form:

```text
task_N/
├── task.json
└── context/
```

The question is loaded from `task.json`; only `context/` is exposed as the data
workspace. The harness must write `prediction.csv` with a header and a
rectangular relation. A header-only CSV is permitted for an empty relation.
Correctness feedback is never returned during a run.

## Tests

The unit suite does not call a model API:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

It covers the original DataSpace-Agent loop and tools, common batch/resume
behavior, configuration loading, native-process deadlines, prediction
acceptance, filesystem confinement, and harness adapter safety settings.
