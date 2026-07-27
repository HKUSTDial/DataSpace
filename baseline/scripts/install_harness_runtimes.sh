#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
baseline_root="$(cd "$script_dir/.." && pwd)"
config_path="$baseline_root/configs/harness.yaml"
python_bin="$baseline_root/.venv/bin/python"
runtime_root="$baseline_root/.runtimes"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Native harness runtimes currently support Linux x86_64 only." >&2
  exit 1
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Install the baseline Python environment before harness runtimes." >&2
  exit 1
fi
for command_name in curl tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "$command_name is required to install harness runtimes." >&2
    exit 1
  fi
done

resolved_versions="$($python_bin - "$config_path" <<'PY'
import sys
from dataspace_baselines.config import load_harness_experiment_config

config = load_harness_experiment_config(sys.argv[1])
for key in ("codex", "claude-code", "grok-build"):
    harness = config.harnesses[key]
    print(harness.options["version"])
PY
)"
readarray -t versions <<< "$resolved_versions"
if [[ ${#versions[@]} -ne 3 ]]; then
  echo "Could not resolve pinned harness versions from $config_path." >&2
  exit 1
fi
codex_version=${versions[0]}
claude_version=${versions[1]}
grok_version=${versions[2]}
mkdir -p "$runtime_root"
work_root="$(mktemp -d)"
trap 'rm -rf "$work_root"' EXIT

download_parallel() {
  local url=$1
  local output=$2
  local header size connections chunk_size chunk_root
  header="$(curl --fail --silent --show-error --location \
    --range 0-0 --dump-header - --output /dev/null "$url")"
  size="$(printf '%s\n' "$header" | awk -F/ \
    'tolower($1) ~ /^content-range:/ {gsub("\\r", "", $2); print $2; exit}')"
  if [[ ! $size =~ ^[1-9][0-9]*$ ]] || (( size < 16777216 )); then
    curl --fail --silent --show-error --location "$url" --output "$output"
    return
  fi

  connections=8
  chunk_size=$(( (size + connections - 1) / connections ))
  chunk_root="$work_root/chunks-$(basename "$output")"
  mkdir -p "$chunk_root"
  local index start end
  local -a pids=()
  for ((index = 0; index < connections; index++)); do
    start=$(( index * chunk_size ))
    end=$(( start + chunk_size - 1 ))
    (( end >= size )) && end=$(( size - 1 ))
    curl --fail --silent --show-error --location \
      --range "$start-$end" \
      "$url" \
      --output "$chunk_root/$(printf '%03d' "$index")" &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  cat "$chunk_root"/* > "$output"
  [[ "$(stat -c '%s' "$output")" == "$size" ]] || {
    echo "Parallel download size verification failed for $url." >&2
    exit 1
  }
}

codex_root="$runtime_root/codex-$codex_version"
codex_package="$codex_root/platform-package"
codex_arch="$codex_package/vendor/x86_64-unknown-linux-musl"
codex_binary="$codex_arch/bin/codex"
if [[ ! -x "$codex_binary" ]] || [[ "$($codex_binary --version 2>/dev/null || true)" != "codex-cli $codex_version" ]]; then
  echo "Installing Codex $codex_version..."
  codex_archive="$work_root/codex-$codex_version.tgz"
  download_parallel \
    "https://registry.npmjs.org/@openai/codex/-/codex-$codex_version-linux-x64.tgz" \
    "$codex_archive"
  mkdir -p "$codex_package"
  tar --extract --gzip --file "$codex_archive" \
    --directory "$codex_package" --strip-components=1
fi
[[ "$($codex_binary --version 2>/dev/null)" == "codex-cli $codex_version" ]] || {
  echo "Codex runtime verification failed." >&2
  exit 1
}

claude_root="$runtime_root/claude-code-$claude_version"
claude_runtime="$claude_root/runtime"
claude_binary="$claude_runtime/bin/claude"
if [[ ! -x "$claude_binary" ]] || [[ "$($claude_binary --version 2>/dev/null || true)" != "$claude_version (Claude Code)" ]]; then
  echo "Installing Claude Code $claude_version..."
  claude_archive="$work_root/claude-code-$claude_version.tgz"
  claude_package="$work_root/claude-code-package"
  download_parallel \
    "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-$claude_version.tgz" \
    "$claude_archive"
  mkdir -p "$claude_package" "$claude_runtime/bin"
  tar --extract --gzip --file "$claude_archive" \
    --directory "$claude_package" --strip-components=1
  install -m755 "$claude_package/claude" "$claude_binary"
fi
# Claude Code's command sandbox uses socat from the shared Data Workbench.
# A host-linked copy may depend on libraries that are absent from the rootfs.
rm -f "$claude_runtime/bin/socat"
[[ "$($claude_binary --version)" == "$claude_version (Claude Code)" ]] || {
  echo "Claude Code runtime verification failed." >&2
  exit 1
}

grok_root="$runtime_root/grok-build-$grok_version"
grok_binary="$grok_root/.grok/bin/grok"
if [[ ! -x "$grok_binary" ]] || ! "$grok_binary" --version 2>/dev/null | grep -Fq "grok $grok_version "; then
  echo "Installing Grok Build $grok_version..."
  installer="$work_root/grok-installer.sh"
  curl --fail --silent --show-error --location \
    https://x.ai/cli/install.sh --output "$installer"
  HOME="$grok_root" \
    GROK_BIN_DIR="$grok_root/.grok/bin" \
    GROK_DISABLE_AUTO_UPDATE=1 \
    SHELL=/bin/false \
    bash "$installer" "$grok_version"
fi
"$grok_binary" --version | grep -Fq "grok $grok_version " || {
  echo "Grok Build runtime verification failed." >&2
  exit 1
}

echo "Installed pinned native harness runtimes:"
echo "  $($codex_binary --version 2>/dev/null)"
echo "  $($claude_binary --version)"
echo "  $($grok_binary --version)"
