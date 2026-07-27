from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import DataWorkbenchConfig


CONTAINER_TOOLS_MANIFEST = "/opt/dataspace/TOOLS.md"
CONTAINER_CAPABILITIES_MANIFEST = "/opt/dataspace/capabilities.json"
EXPORT_MANIFEST_NAME = "workbench-manifest.json"


def export_manifest_path(workbench: DataWorkbenchConfig) -> Path:
    return workbench.rootfs_path.parent / EXPORT_MANIFEST_NAME


def load_export_manifest(workbench: DataWorkbenchConfig) -> dict[str, Any]:
    path = export_manifest_path(workbench)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def runtime_metadata(workbench: DataWorkbenchConfig) -> dict[str, Any]:
    manifest_path = export_manifest_path(workbench)
    manifest = load_export_manifest(workbench)
    metadata: dict[str, Any] = {
        "image": workbench.image,
        "rootfs_path": str(workbench.rootfs_path),
        "tools_manifest": CONTAINER_TOOLS_MANIFEST,
        "capabilities_manifest": CONTAINER_CAPABILITIES_MANIFEST,
    }
    for key in ("runtime_version", "image_id", "tools_sha256", "exported_at"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    if manifest_path.is_file():
        metadata["export_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    return metadata


def runtime_identity(workbench: DataWorkbenchConfig) -> dict[str, str]:
    metadata = runtime_metadata(workbench)
    identity: dict[str, str] = {"image": workbench.image}
    for key in ("runtime_version", "image_id", "tools_sha256"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            identity[key] = value
    return identity
