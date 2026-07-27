from __future__ import annotations

from ..config import HarnessExperimentConfig
from ..core.interfaces import HarnessRunner
from .claude_code import build_claude_code
from .codex import build_codex
from .dataspace_agent import build_dataspace_agent
from .grok_build import build_grok_build


_HARNESS_KEYS = (
    "dataspace-agent",
    "smolagents",
    "codex",
    "claude-code",
    "grok-build",
)


def available_harnesses() -> tuple[str, ...]:
    return _HARNESS_KEYS


def build_harness(
    harness_key: str, config: HarnessExperimentConfig
) -> HarnessRunner:
    if harness_key not in config.harnesses:
        available = ", ".join(sorted(config.harnesses))
        raise KeyError(
            f"harness {harness_key!r} is not configured; available: {available}"
        )
    harness = config.harnesses[harness_key]
    if not harness.enabled:
        raise ValueError(f"harness {harness_key!r} is disabled")
    if harness_key == "dataspace-agent":
        return build_dataspace_agent(config)
    if harness_key == "codex":
        return build_codex(config, harness)
    if harness_key == "claude-code":
        return build_claude_code(config, harness)
    if harness_key == "grok-build":
        return build_grok_build(config, harness)
    if harness_key == "smolagents":
        from .smolagents import build_smolagents

        return build_smolagents(config, harness)
    raise KeyError(f"unsupported harness: {harness_key}")
