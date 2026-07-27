#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_benchmark.sh model MODEL [options]
  run_benchmark.sh harness HARNESS [options]

Modes:
  model    Compare backbone models with the fixed DataSpace-Agent harness.
  harness  Compare harnesses using the model selected by the harness config.

Common options:
  --config PATH      Override the mode's default config
  --input-root PATH  Benchmark input root
  --run-root PATH    Explicit batch run root
  --concurrency N    Concurrent tasks
  --run-id ID        Derived run directory name (mode-specific default)
  --session NAME     tmux session name
  --resume           Resume retryable or interrupted tasks
  --foreground       Run in the current shell instead of tmux
  -h, --help         Show this help

Examples:
  ./scripts/run_benchmark.sh model grok-4.5 --resume
  ./scripts/run_benchmark.sh model mimo-v2.5 --concurrency 4
  ./scripts/run_benchmark.sh harness codex
  ./scripts/run_benchmark.sh harness claude-code --resume
EOF
}

mode_usage() {
  case "$1" in
    model)
      cat <<'EOF'
Usage:
  run_benchmark.sh model MODEL [options]

Runs the controlled DataSpace-Agent baseline with one model from
configs/vercel.yaml. Concurrency defaults to 8.
EOF
      ;;
    harness)
      cat <<'EOF'
Usage:
  run_benchmark.sh harness HARNESS [options]

Harnesses:
  dataspace-agent | smolagents | codex | claude-code | grok-build

Uses configs/harness.yaml by default. The checked-in config selects MiMo-V2.5;
edit it or pass another file with --config to choose a different model.
Concurrency comes from the experiment config.
EOF
      ;;
  esac
  cat <<'EOF'

Options:
  --config PATH      Override the default config
  --input-root PATH  Benchmark input root
  --run-root PATH    Explicit batch run root
  --concurrency N    Concurrent tasks
  --run-id ID        Derived run directory name
                       model default: run_1
                       harness default: workbench-1.0
  --session NAME     tmux session name
  --resume           Resume retryable or interrupted tasks
  --foreground       Run in the current shell instead of tmux
  -h, --help         Show help
EOF
}

if [[ $# -eq 0 || $1 == -h || $1 == --help ]]; then
  usage
  exit 0
fi

mode=$1
shift
case "$mode" in
  model|harness) ;;
  *)
    echo "error: mode must be 'model' or 'harness': $mode" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ $# -eq 0 ]]; then
  mode_usage "$mode" >&2
  exit 2
fi
if [[ $1 == -h || $1 == --help ]]; then
  mode_usage "$mode"
  exit 0
fi

subject=$1
shift
[[ $subject =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "error: invalid $mode key: $subject" >&2
  exit 2
}
if [[ $mode == harness ]]; then
  case "$subject" in
    dataspace-agent|smolagents|codex|claude-code|grok-build) ;;
    *)
      echo "error: unsupported harness: $subject" >&2
      mode_usage harness >&2
      exit 2
      ;;
  esac
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
baseline_dir=$(cd -- "$script_dir/.." && pwd)
repo_dir=$(cd -- "$baseline_dir/.." && pwd)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
source_dir="$baseline_dir/src"
python_bin="$baseline_dir/.venv/bin/python"
runner_bin="$baseline_dir/.venv/bin/dataspace-baselines"

if [[ $mode == model ]]; then
  config_path="$baseline_dir/configs/vercel.yaml"
  concurrency=8
else
  config_path="$baseline_dir/configs/harness.yaml"
  concurrency=
fi
input_root="$workspace_dir/DataAgentBenchmark/input"
run_root=
if [[ $mode == model ]]; then
  run_id=run_1
else
  # Keep official unified-runtime runs separate from pre-upgrade harness
  # attempts that used the historical run_1 directory.
  run_id=workbench-1.0
fi
session_name=
resume=false
foreground=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "error: --config requires a value" >&2; exit 2; }
      config_path=$2
      shift 2
      ;;
    --input-root)
      [[ $# -ge 2 ]] || { echo "error: --input-root requires a value" >&2; exit 2; }
      input_root=$2
      shift 2
      ;;
    --run-root)
      [[ $# -ge 2 ]] || { echo "error: --run-root requires a value" >&2; exit 2; }
      run_root=$2
      shift 2
      ;;
    --concurrency)
      [[ $# -ge 2 ]] || { echo "error: --concurrency requires a value" >&2; exit 2; }
      concurrency=$2
      shift 2
      ;;
    --run-id)
      [[ $# -ge 2 ]] || { echo "error: --run-id requires a value" >&2; exit 2; }
      run_id=$2
      shift 2
      ;;
    --session)
      [[ $# -ge 2 ]] || { echo "error: --session requires a value" >&2; exit 2; }
      session_name=$2
      shift 2
      ;;
    --resume)
      resume=true
      shift
      ;;
    --foreground)
      foreground=true
      shift
      ;;
    -h|--help)
      mode_usage "$mode"
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      mode_usage "$mode" >&2
      exit 2
      ;;
  esac
done

[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "error: invalid run id: $run_id" >&2
  exit 2
}
if [[ -n $concurrency ]] && [[ ! $concurrency =~ ^[1-9][0-9]*$ ]]; then
  echo "error: concurrency must be a positive integer" >&2
  exit 2
fi
if [[ -n $session_name ]] && [[ ! $session_name =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "error: invalid tmux session name: $session_name" >&2
  exit 2
fi

[[ -x $python_bin ]] || {
  echo "error: project Python environment not found: $python_bin" >&2
  echo "create it and install the baseline package before running" >&2
  exit 1
}
[[ -f $config_path ]] || { echo "error: config not found: $config_path" >&2; exit 1; }
[[ -d $input_root ]] || { echo "error: benchmark input not found: $input_root" >&2; exit 1; }

if [[ $mode == model ]]; then
  resolved=$(
    PYTHONPATH="$source_dir" "$python_bin" - "$config_path" "$subject" <<'PY'
import importlib.util
import sys
from dataspace_agent.config import load_run_config

if importlib.util.find_spec("openai") is None:
    raise SystemExit("the openai package is unavailable")
config = load_run_config(sys.argv[1], sys.argv[2])
print(config.model.key)
print(config.model.api_key_env)
print(8)
PY
  )
else
  [[ -x $runner_bin ]] || {
    echo "error: unified baseline entry point not found: $runner_bin" >&2
    exit 1
  }
  resolved=$(
    "$python_bin" - "$config_path" "$subject" <<'PY'
import sys
from dataspace_baselines.config import load_harness_experiment_config

config = load_harness_experiment_config(sys.argv[1])
harness = config.harnesses.get(sys.argv[2])
if harness is None:
    raise SystemExit(f"harness is not configured: {sys.argv[2]}")
if not harness.enabled:
    raise SystemExit(f"harness is disabled: {sys.argv[2]}")
print(config.model.key)
print(config.model.api_key_env)
print(config.concurrency)
PY
  )
fi
readarray -t experiment_values <<< "$resolved"
if [[ ${#experiment_values[@]} -ne 3 ]]; then
  echo "error: could not resolve experiment configuration" >&2
  exit 1
fi
model_key=${experiment_values[0]}
api_key_env=${experiment_values[1]}
default_concurrency=${experiment_values[2]}

if [[ -z ${!api_key_env:-} ]]; then
  echo "error: $api_key_env is not set" >&2
  exit 1
fi
if [[ -z $concurrency ]]; then
  concurrency=$default_concurrency
fi

if [[ -z $run_root ]]; then
  if [[ $mode == model ]]; then
    run_root="$repo_dir/runs/${subject}-default-full/$run_id"
  else
    run_root="$repo_dir/runs/harness-comparison/$model_key/$subject/$run_id"
  fi
fi
log_path="${run_root%/}.console.log"

if [[ $mode == model ]]; then
  runner=(
    "$python_bin" -m dataspace_agent batch
    --input-root "$input_root"
    --config "$config_path"
    --model "$subject"
    --run-root "$run_root"
    --concurrency "$concurrency"
  )
else
  runner=(
    "$runner_bin" batch
    --harness "$subject"
    --input-root "$input_root"
    --config "$config_path"
    --run-root "$run_root"
    --concurrency "$concurrency"
  )
fi
if $resume; then
  runner+=(--resume)
fi

if [[ -d $run_root ]] && [[ -n $(find "$run_root" -mindepth 1 -print -quit) ]] && ! $resume; then
  echo "error: run directory is not empty: $run_root" >&2
  echo "use --resume, --run-id, or --run-root" >&2
  exit 1
fi

export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
if $foreground; then
  exec "${runner[@]}"
fi

command -v tmux >/dev/null || { echo "error: tmux is not installed" >&2; exit 1; }
if [[ -z $session_name ]]; then
  session_subject=${subject//./_}
  session_run_id=${run_id//./_}
  if [[ $mode == model ]]; then
    session_name="dataspace-${session_subject}-${session_run_id}"
  else
    session_name="dataspace-harness-${session_subject}-${session_run_id}"
  fi
fi
[[ $session_name =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "error: invalid tmux session name: $session_name" >&2
  exit 2
}
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "error: tmux session already exists: $session_name" >&2
  echo "attach with: tmux attach -t $session_name" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$run_root")"
printf -v runner_command '%q ' "${runner[@]}"
runner_command="sleep 1; exec ${runner_command% }"

pane_id=$(tmux new-session -d -P -F '#{pane_id}' \
  -s "$session_name" \
  -c "$baseline_dir" \
  -e "$api_key_env=${!api_key_env}" \
  -e "PYTHONPATH=$source_dir" \
  "$runner_command")
[[ $pane_id =~ ^%[0-9]+$ ]] || {
  echo "error: tmux did not return a valid pane id: $pane_id" >&2
  exit 1
}
tmux set-option -w -t "$session_name:0" remain-on-exit off >/dev/null
printf -v pipe_command 'cat >> %q' "$log_path"
tmux pipe-pane -o -t "$pane_id" "$pipe_command"

printf '[launcher] %s mode=%s subject=%s model=%s concurrency=%s resume=%s\n' \
  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$mode" "$subject" "$model_key" \
  "$concurrency" "$resume" >> "$log_path"

echo "Started benchmark."
echo "  mode:    $mode"
echo "  subject: $subject"
echo "  model:   $model_key"
echo "  session: $session_name (closes automatically when the run ends)"
echo "  run:     $run_root"
echo "  log:     $log_path"
echo "Attach: tmux attach -t $session_name"
