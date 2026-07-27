"""Minimal terminal-style ReAct baseline for DataSpace."""

from .agent import DataSpaceAgent
from .config import RunConfig, load_run_config
from .task import TaskSpec

__all__ = ["DataSpaceAgent", "RunConfig", "TaskSpec", "load_run_config"]
__version__ = "0.1.0"
