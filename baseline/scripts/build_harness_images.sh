#!/usr/bin/env bash

set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  echo "Usage: build_harness_images.sh [--rebuild]"
  echo "Build the shared Data Workbench Runtime and smolagents image."
  echo "The workbench image is also exported as a read-only native rootfs."
  echo "Use --rebuild to replace existing images and the exported rootfs."
  exit 0
fi
rebuild=false
if [[ ${1:-} == --rebuild ]]; then
  rebuild=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "error: expected no arguments or --rebuild" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
baseline_dir=$(cd -- "$script_dir/.." && pwd)

ensure_image() {
  local image=$1
  local dockerfile=$2
  shift 2
  if ! $rebuild && docker image inspect "$image" >/dev/null 2>&1; then
    echo "Reusing existing image: $image"
    return
  fi
  docker build \
    --tag "$image" \
    --file "$dockerfile" \
    "$@" \
    "$baseline_dir"
}

materialize_rootfs() {
  local image=$1
  local runtime_dir=$2
  local rootfs_dir="$runtime_dir/rootfs"
  local manifest="$runtime_dir/workbench-manifest.json"
  local image_id
  image_id=$(docker image inspect "$image" --format '{{.Id}}')

  if ! $rebuild && [[ -d $rootfs_dir && -f $manifest ]]; then
    local recorded_id
    recorded_id=$(sed -n 's/^[[:space:]]*"image_id": "\([^"]*\)",*$/\1/p' "$manifest")
    if [[ $recorded_id == "$image_id" ]]; then
      echo "Reusing exported Data Workbench rootfs: $rootfs_dir"
      return
    fi
  fi

  mkdir -p "$(dirname -- "$runtime_dir")"
  local temporary
  temporary=$(mktemp -d "$(dirname -- "$runtime_dir")/.data-workbench.XXXXXX")
  local container_id=
  cleanup_export() {
    if [[ -n $container_id ]]; then
      docker rm --force "$container_id" >/dev/null 2>&1 || true
    fi
    rm -rf "$temporary"
  }
  trap cleanup_export RETURN

  container_id=$(docker create "$image")
  mkdir -p "$temporary/rootfs"
  docker export --output "$temporary/rootfs.tar" "$container_id"
  tar -xf "$temporary/rootfs.tar" -C "$temporary/rootfs"
  docker rm --force "$container_id" >/dev/null
  container_id=
  rm -f "$temporary/rootfs.tar"

  local tools_sha256
  tools_sha256=$(sha256sum "$temporary/rootfs/opt/dataspace/TOOLS.md" | cut -d' ' -f1)
  local exported_at
  exported_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{\n  "runtime_version": "1.0",\n  "image": "%s",\n  "image_id": "%s",\n  "tools_sha256": "%s",\n  "exported_at": "%s"\n}\n' \
    "$image" "$image_id" "$tools_sha256" "$exported_at" \
    > "$temporary/workbench-manifest.json"

  rm -rf "$runtime_dir"
  mv "$temporary" "$runtime_dir"
  trap - RETURN
  echo "Exported Data Workbench rootfs: $rootfs_dir"
}

workbench_image=dataspace-workbench:1.0
smolagents_image=dataspace-smolagents:1.0
workbench_runtime="$baseline_dir/.runtimes/data-workbench-1.0"
python_image=${DATASPACE_PYTHON_IMAGE:-python:3.11-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba}
debian_mirror=${DATASPACE_DEBIAN_MIRROR:-http://deb.debian.org/debian}
debian_security_mirror=${DATASPACE_DEBIAN_SECURITY_MIRROR:-http://deb.debian.org/debian-security}
pip_index_url=${DATASPACE_PIP_INDEX_URL:-https://pypi.org/simple}

ensure_image \
  "$workbench_image" \
  "$baseline_dir/docker/Dockerfile.workbench" \
  --build-arg "PYTHON_IMAGE=$python_image" \
  --build-arg "DEBIAN_MIRROR=$debian_mirror" \
  --build-arg "DEBIAN_SECURITY_MIRROR=$debian_security_mirror" \
  --build-arg "PIP_INDEX_URL=$pip_index_url"
ensure_image \
  "$smolagents_image" \
  "$baseline_dir/docker/Dockerfile.smolagents" \
  --build-arg "BASE_IMAGE=$workbench_image" \
  --build-arg "PIP_INDEX_URL=$pip_index_url"

materialize_rootfs "$workbench_image" "$workbench_runtime"

echo "Harness runtimes are ready:"
echo "  $workbench_image"
echo "  $smolagents_image"
echo "  $workbench_runtime/rootfs"
