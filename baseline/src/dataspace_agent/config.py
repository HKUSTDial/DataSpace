from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model: str
    api_key_env: str
    base_url: str | None = None
    max_output_tokens: int = 32768
    temperature: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    default_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    max_model_turns: int = 60
    max_tool_actions: int = 50
    max_wall_time_seconds: int = 1800
    max_command_seconds: int = 180
    max_tool_output_chars: int = 20000
    image_max_edge: int = 1600
    image_jpeg_quality: int = 90


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "dataspace-workbench:1.0"
    cpus: float = 4
    memory: str = "16g"
    pids_limit: int = 256
    tmpfs_size: str = "2g"


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    agent: AgentConfig
    sandbox: SandboxConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def load_run_config(path: str | Path, model_key: str) -> RunConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "config")

    model_map = _mapping(raw.get("models"), "models")
    if model_key not in model_map:
        available = ", ".join(sorted(model_map))
        raise KeyError(f"unknown model key {model_key!r}; available: {available}")
    model_raw = _mapping(model_map[model_key], f"models.{model_key}")

    model_name = str(model_raw.get("model", "")).strip()
    api_key_env = str(model_raw.get("api_key_env", "")).strip()
    if not model_name:
        raise ValueError(f"models.{model_key}.model is required")
    if not api_key_env:
        raise ValueError(f"models.{model_key}.api_key_env is required")

    model = ModelConfig(
        key=model_key,
        model=model_name,
        api_key_env=api_key_env,
        base_url=model_raw.get("base_url"),
        max_output_tokens=_positive_int(
            model_raw.get("max_output_tokens", 32768),
            f"models.{model_key}.max_output_tokens",
        ),
        temperature=model_raw.get("temperature"),
        extra_body=_mapping(model_raw.get("extra_body", {}), "extra_body"),
        default_headers={
            str(key): str(value)
            for key, value in _mapping(
                model_raw.get("default_headers", {}), "default_headers"
            ).items()
        },
    )

    agent_raw = _mapping(raw.get("agent", {}), "agent")
    agent = AgentConfig(
        max_model_turns=_positive_int(
            agent_raw.get("max_model_turns", 60), "agent.max_model_turns"
        ),
        max_tool_actions=_positive_int(
            agent_raw.get("max_tool_actions", 50), "agent.max_tool_actions"
        ),
        max_wall_time_seconds=_positive_int(
            agent_raw.get("max_wall_time_seconds", 1800),
            "agent.max_wall_time_seconds",
        ),
        max_command_seconds=_positive_int(
            agent_raw.get("max_command_seconds", 180),
            "agent.max_command_seconds",
        ),
        max_tool_output_chars=_positive_int(
            agent_raw.get("max_tool_output_chars", 20000),
            "agent.max_tool_output_chars",
        ),
        image_max_edge=_positive_int(
            agent_raw.get("image_max_edge", 1600), "agent.image_max_edge"
        ),
        image_jpeg_quality=_positive_int(
            agent_raw.get("image_jpeg_quality", 90),
            "agent.image_jpeg_quality",
        ),
    )
    if not 1 <= agent.image_jpeg_quality <= 100:
        raise ValueError("agent.image_jpeg_quality must be between 1 and 100")

    sandbox_raw = _mapping(raw.get("sandbox", {}), "sandbox")
    sandbox = SandboxConfig(
        image=str(
            sandbox_raw.get("image", "dataspace-workbench:1.0")
        ),
        cpus=float(sandbox_raw.get("cpus", 4)),
        memory=str(sandbox_raw.get("memory", "16g")),
        pids_limit=_positive_int(
            sandbox_raw.get("pids_limit", 256), "sandbox.pids_limit"
        ),
        tmpfs_size=str(sandbox_raw.get("tmpfs_size", "2g")),
    )
    if sandbox.cpus <= 0:
        raise ValueError("sandbox.cpus must be positive")

    return RunConfig(model=model, agent=agent, sandbox=sandbox)
