from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from dataspace_agent.agent import DataSpaceAgent, PROMPT_CONTRACT_VERSION
from dataspace_agent.backend import OpenAICompatibleBackend
from dataspace_agent.config import AgentConfig, ModelConfig, SandboxConfig
from dataspace_agent.sandbox import DockerSandbox

from ..config import HarnessExperimentConfig
from ..workbench import runtime_identity, runtime_metadata


def build_dataspace_agent(config: HarnessExperimentConfig) -> DataSpaceAgent:
    model = ModelConfig(
        key=config.model.key,
        model=config.model.model,
        api_key_env=config.model.api_key_env,
        base_url=config.model.openai_base_url,
        max_output_tokens=config.model.max_output_tokens,
    )
    agent_config = AgentConfig(
        max_model_turns=1_000_000,
        max_tool_actions=1_000_000,
        max_wall_time_seconds=config.wall_time_seconds,
        max_command_seconds=config.wall_time_seconds,
        max_tool_output_chars=20_000,
        image_max_edge=1600,
        image_jpeg_quality=90,
    )
    sandbox_config = SandboxConfig(image=config.workbench.image)
    backend = OpenAICompatibleBackend(model)

    def sandbox_factory(workspace: Path, output: Path) -> DockerSandbox:
        return DockerSandbox(
            workspace_root=workspace,
            output_root=output,
            config=sandbox_config,
            max_output_chars=agent_config.max_tool_output_chars,
        )

    return DataSpaceAgent(
        backend=backend,
        agent_config=agent_config,
        sandbox_factory=sandbox_factory,
        run_metadata={
            "harness": "dataspace-agent",
            "model": {
                "key": model.key,
                "requested_model": model.model,
                "base_url": model.base_url,
                "max_output_tokens": model.max_output_tokens,
                "api_key_env": model.api_key_env,
            },
            "sandbox": asdict(sandbox_config),
            "data_workbench": runtime_metadata(config.workbench),
            "common_limit": {
                "max_wall_time_seconds": config.wall_time_seconds,
            },
        },
        enforce_remaining_wall_time=True,
        resume_identity={
            "runner": "dataspace-agent",
            "model": {
                "key": model.key,
                "model": model.model,
                "base_url": model.base_url,
                "max_output_tokens": model.max_output_tokens,
            },
            "wall_time_seconds": config.wall_time_seconds,
            "workbench": runtime_identity(config.workbench),
            "agent": asdict(agent_config),
            "prompt_contract": PROMPT_CONTRACT_VERSION,
        },
    )
