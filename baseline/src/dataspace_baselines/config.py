from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentModelConfig:
    key: str
    model: str
    api_key_env: str
    openai_base_url: str
    anthropic_base_url: str
    max_output_tokens: int = 32768
    context_window: int = 200000


@dataclass(frozen=True)
class HarnessConfig:
    key: str
    enabled: bool = True
    executable: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataWorkbenchConfig:
    image: str
    rootfs_path: Path


@dataclass(frozen=True)
class HarnessExperimentConfig:
    model: ExperimentModelConfig
    wall_time_seconds: int
    concurrency: int
    workbench: DataWorkbenchConfig
    harnesses: dict[str, HarnessConfig]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def load_harness_experiment_config(
    path: str | Path,
) -> HarnessExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "config")

    experiment = _mapping(raw.get("experiment"), "experiment")
    model_raw = _mapping(experiment.get("model"), "experiment.model")
    model_key = str(model_raw.get("key", "")).strip()
    model_name = str(model_raw.get("model", "")).strip()
    api_key_env = str(model_raw.get("api_key_env", "")).strip()
    openai_base_url = str(model_raw.get("openai_base_url", "")).rstrip("/")
    anthropic_base_url = str(
        model_raw.get("anthropic_base_url", "")
    ).rstrip("/")
    for name, value in (
        ("key", model_key),
        ("model", model_name),
        ("api_key_env", api_key_env),
        ("openai_base_url", openai_base_url),
        ("anthropic_base_url", anthropic_base_url),
    ):
        if not value:
            raise ValueError(f"experiment.model.{name} is required")

    model = ExperimentModelConfig(
        key=model_key,
        model=model_name,
        api_key_env=api_key_env,
        openai_base_url=openai_base_url,
        anthropic_base_url=anthropic_base_url,
        max_output_tokens=_positive_int(
            model_raw.get("max_output_tokens", 32768),
            "experiment.model.max_output_tokens",
        ),
        context_window=_positive_int(
            model_raw.get("context_window", 200000),
            "experiment.model.context_window",
        ),
    )
    if model.max_output_tokens >= model.context_window:
        raise ValueError(
            "experiment.model.max_output_tokens must be smaller than "
            "experiment.model.context_window"
        )

    harness_map = _mapping(raw.get("harnesses"), "harnesses")
    harnesses: dict[str, HarnessConfig] = {}
    for key, value in harness_map.items():
        harness_key = str(key).strip()
        if not harness_key:
            raise ValueError("harness key must not be empty")
        harness_raw = _mapping(value, f"harnesses.{harness_key}")
        executable_raw = harness_raw.get("executable")
        executable = (
            str(executable_raw).strip() if executable_raw is not None else None
        )
        reserved = {"enabled", "executable"}
        harnesses[harness_key] = HarnessConfig(
            key=harness_key,
            enabled=bool(harness_raw.get("enabled", True)),
            executable=executable or None,
            options={
                str(option): option_value
                for option, option_value in harness_raw.items()
                if option not in reserved
            },
        )
    if not harnesses:
        raise ValueError("at least one harness must be configured")

    workbench_raw = _mapping(
        experiment.get("workbench", {}), "experiment.workbench"
    )
    workbench_image = str(
        workbench_raw.get(
            "image",
            experiment.get("sandbox_image", "dataspace-workbench:1.0"),
        )
    ).strip()
    if not workbench_image:
        raise ValueError("experiment.workbench.image is required")
    rootfs_raw = str(
        workbench_raw.get(
            "rootfs_path", "../.runtimes/data-workbench-1.0/rootfs"
        )
    ).strip()
    if not rootfs_raw:
        raise ValueError("experiment.workbench.rootfs_path is required")
    rootfs_path = Path(rootfs_raw).expanduser()
    if not rootfs_path.is_absolute():
        rootfs_path = config_path.resolve().parent / rootfs_path
    workbench = DataWorkbenchConfig(
        image=workbench_image,
        rootfs_path=rootfs_path.resolve(),
    )

    return HarnessExperimentConfig(
        model=model,
        wall_time_seconds=_positive_int(
            experiment.get("wall_time_seconds", 1800),
            "experiment.wall_time_seconds",
        ),
        concurrency=_positive_int(
            experiment.get("concurrency", 8), "experiment.concurrency"
        ),
        workbench=workbench,
        harnesses=harnesses,
    )
