from __future__ import annotations

from pathlib import Path

from .task import TaskSpec


PROMPT_CONTRACT_VERSION = "relation-v2"


def relation_task_prompt(
    task: TaskSpec,
    *,
    workspace_path: str | Path,
    output_path: str | Path,
) -> str:
    """The common user-level contract used in native-harness evaluation."""

    return f"""Task ID: {task.task_id}

Question:
{task.question}

All task input artifacts are located under:
{workspace_path}

Solve the task using only those artifacts. Internet access is unavailable.
The preinstalled generic data-processing utilities are documented at
/opt/dataspace/TOOLS.md. Package installation is unavailable.
Write the complete requested relation to:
{output_path}

The answer must be a rectangular UTF-8 CSV with one header row. A header-only
CSV is valid when the requested result is empty. Return all requested rows and
columns; do not replace the CSV with a prose answer.

Do not finish the task until the exact file path above exists. Before ending,
verify that its filename is exactly prediction.csv, that it is readable as a
rectangular CSV, and that it contains the complete final relation. A prose
response or a differently named file does not count as submission. Once the
verified prediction.csv is ready, terminate the session."""
