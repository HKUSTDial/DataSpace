from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .backend import ModelBackend
from .config import AgentConfig
from .media import ImageObserver
from .runlog import RunLogger, utc_now
from .sandbox import Sandbox
from .task import TaskPaths, TaskSpec
from .tools import TOOL_SCHEMAS, ToolExecutor
from .types import RunResult


SYSTEM_PROMPT = """You are a data agent operating in an isolated task workspace.

Solve the user's analytical question using only the files under /workspace.
The workspace is read-only. Store scripts, intermediate files, rendered PDF
pages, extracted video frames, and the final answer under /output.

You have exactly three tools:
1. bash for low-level filesystem inspection and data processing;
2. view_image for inspecting one image that you selected or rendered;
3. submit_answer for submitting a final rectangular CSV under /output.

Use bash with standard command-line tools and Python for CSV, JSON, SQLite,
Markdown, PDF, and video processing. Decide which files, pages, and timestamps
to inspect. The tools do not perform semantic retrieval or task-specific
reasoning for you.

The preinstalled generic utilities are documented at
/opt/dataspace/TOOLS.md. Package installation and Internet access are
unavailable.

Return the complete requested relation, not an explanation. CSV header names
need not match hidden reference names, but the data columns and rows must have
the intended semantics. Do not end the run until the exact file
/output/prediction.csv exists and you have called submit_answer on that file.
Verify the CSV is readable and rectangular before submitting it. A prose
answer or a differently named file is not a submission."""

PROMPT_CONTRACT_VERSION = "dataspace-agent-v2"


def _usage_totals(current: dict[str, int], usage: dict[str, Any]) -> None:
    aliases = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "total_tokens": "total_tokens",
        "reasoning_tokens": "reasoning_tokens",
    }
    for source, destination in aliases.items():
        value = usage.get(source)
        if isinstance(value, int):
            current[destination] = current.get(destination, 0) + value
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
        if isinstance(reasoning, int):
            current["reasoning_tokens"] = (
                current.get("reasoning_tokens", 0) + reasoning
            )


class DataSpaceAgent:
    def __init__(
        self,
        backend: ModelBackend,
        agent_config: AgentConfig,
        sandbox_factory: Callable[[Path, Path], Sandbox],
        run_metadata: dict[str, Any] | None = None,
        enforce_remaining_wall_time: bool = False,
        resume_identity: dict[str, Any] | None = None,
    ):
        self.backend = backend
        self.config = agent_config
        self.sandbox_factory = sandbox_factory
        self.run_metadata = run_metadata or {}
        self.resume_identity = resume_identity or {
            "runner": "dataspace-agent",
            "agent": {
                "max_model_turns": agent_config.max_model_turns,
                "max_tool_actions": agent_config.max_tool_actions,
                "max_wall_time_seconds": agent_config.max_wall_time_seconds,
                "max_command_seconds": agent_config.max_command_seconds,
                "max_tool_output_chars": agent_config.max_tool_output_chars,
                "image_max_edge": agent_config.image_max_edge,
                "image_jpeg_quality": agent_config.image_jpeg_quality,
            },
            "run_metadata": self.run_metadata,
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
        }
        # Kept off by default so the established dataspace-agent CLI retains
        # its exact command-timeout behavior. The common harness protocol
        # enables it to make the shared wall deadline active within a command.
        self.enforce_remaining_wall_time = enforce_remaining_wall_time
        self._cancel_event = threading.Event()
        self._sandbox_lock = threading.Lock()
        self._active_sandboxes: dict[int, Sandbox] = {}

    def cancel(self) -> None:
        """Request cancellation of every active run using this agent."""

        self._cancel_event.set()
        cancel_backend = getattr(self.backend, "cancel", None)
        if callable(cancel_backend):
            try:
                cancel_backend()
            except Exception:
                pass
        with self._sandbox_lock:
            sandboxes = list(self._active_sandboxes.values())
        for sandbox in sandboxes:
            try:
                sandbox.close()
            except Exception:
                pass

    @property
    def resume_fingerprint(self) -> str:
        serialized = json.dumps(
            self.resume_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def run(self, task: TaskSpec, run_dir: str | Path) -> RunResult:
        run_root = Path(run_dir).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        output_root = run_root / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        logger = RunLogger(run_root)
        paths = TaskPaths(task.context_dir, output_root)
        sandbox = self.sandbox_factory(task.context_dir, output_root)
        image_observer = ImageObserver(
            paths,
            max_edge=self.config.image_max_edge,
            jpeg_quality=self.config.image_jpeg_quality,
        )
        tool_executor = ToolExecutor(
            sandbox=sandbox,
            paths=paths,
            image_observer=image_observer,
            max_command_seconds=self.config.max_command_seconds,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task ID: {task.task_id}\n"
                    f"Question: {task.question}\n\n"
                    "Inspect /workspace. Before ending, write the complete "
                    "relation to the exact path /output/prediction.csv, "
                    "verify it, and call submit_answer on that file."
                ),
            },
        ]
        started = time.monotonic()
        model_turns = 0
        tool_actions = 0
        usage_totals: dict[str, int] = {}
        prediction_path: Path | None = None
        status = "runtime_error"
        error: str | None = None

        logger.event(
            "run_started",
            {
                "task_id": task.task_id,
                "question": task.question,
                "workspace": str(task.context_dir),
                "started_at": utc_now(),
                "run_metadata": self.run_metadata,
                "experiment_fingerprint": self.resume_fingerprint,
                "system_prompt_sha256": hashlib.sha256(
                    SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
                "limits": {
                    "max_model_turns": self.config.max_model_turns,
                    "max_tool_actions": self.config.max_tool_actions,
                    "max_wall_time_seconds": (
                        self.config.max_wall_time_seconds
                    ),
                    "max_command_seconds": self.config.max_command_seconds,
                    "image_max_edge": self.config.image_max_edge,
                },
            },
        )

        try:
            with self._sandbox_lock:
                self._active_sandboxes[id(sandbox)] = sandbox
            if self._cancel_event.is_set():
                status = "stopped"
            else:
                sandbox.start()
            while True:
                if self._cancel_event.is_set():
                    status = "stopped"
                    break
                elapsed = time.monotonic() - started
                if elapsed >= self.config.max_wall_time_seconds:
                    status = "wall_time_exhausted"
                    break
                if model_turns >= self.config.max_model_turns:
                    status = "model_turns_exhausted"
                    break
                if tool_actions >= self.config.max_tool_actions:
                    status = "tool_actions_exhausted"
                    break

                remaining_seconds = max(
                    1.0, self.config.max_wall_time_seconds - elapsed
                )
                turn = self.backend.complete(
                    messages,
                    TOOL_SCHEMAS,
                    timeout_seconds=remaining_seconds,
                )
                model_turns += 1
                _usage_totals(usage_totals, turn.usage)
                logger.event(
                    "model_turn",
                    {
                        "turn": model_turns,
                        "assistant": turn.assistant_message,
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "name": call.name,
                                "arguments": call.arguments_json,
                            }
                            for call in turn.tool_calls
                        ],
                        "usage": turn.usage,
                        "finish_reason": turn.finish_reason,
                        "response_id": turn.response_id,
                        "resolved_model": turn.resolved_model,
                    },
                )
                messages.append(turn.assistant_message)

                if not turn.tool_calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue by calling one of the available "
                                "tools. The run ends only after submit_answer."
                            ),
                        }
                    )
                    logger.event(
                        "protocol_reminder",
                        {"reason": "assistant made no tool call"},
                    )
                    continue

                pending_image_messages: list[dict[str, Any]] = []
                for call in turn.tool_calls:
                    if self._cancel_event.is_set():
                        status = "stopped"
                        break
                    if tool_actions >= self.config.max_tool_actions:
                        break
                    remaining_wall_time: float | None = None
                    if self.enforce_remaining_wall_time:
                        remaining_wall_time = (
                            self.config.max_wall_time_seconds
                            - (time.monotonic() - started)
                        )
                        if remaining_wall_time <= 0:
                            status = "wall_time_exhausted"
                            break
                    tool_actions += 1
                    command_timeout_cap = None
                    if remaining_wall_time is not None:
                        command_timeout_cap = max(
                            1, math.ceil(remaining_wall_time)
                        )
                    outcome = tool_executor.execute(
                        call,
                        command_timeout_cap=command_timeout_cap,
                    )
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": json.dumps(
                            outcome.result,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                    messages.append(tool_message)
                    logger.event(
                        "tool_result",
                        {
                            "action": tool_actions,
                            "tool_call_id": call.call_id,
                            "tool": call.name,
                            "result": outcome.result,
                        },
                    )
                    if outcome.image_message is not None:
                        pending_image_messages.append(outcome.image_message)
                    if outcome.submitted_path is not None:
                        prediction_path = outcome.submitted_path
                        status = "submitted"
                        break

                if prediction_path is not None:
                    break
                messages.extend(pending_image_messages)

            if status == "runtime_error":
                status = "stopped"
        except Exception as exc:
            if self._cancel_event.is_set():
                status = "stopped"
            else:
                status = "runtime_error"
                error = f"{type(exc).__name__}: {exc}"
                logger.event(
                    "run_error",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
        finally:
            try:
                sandbox.close()
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                logger.event(
                    "sandbox_cleanup_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if error is None:
                    error = cleanup_error
            with self._sandbox_lock:
                self._active_sandboxes.pop(id(sandbox), None)

        wall_time = time.monotonic() - started
        result = RunResult(
            task_id=task.task_id,
            status=status,
            run_dir=run_root,
            prediction_path=prediction_path,
            model_turns=model_turns,
            tool_actions=tool_actions,
            wall_time_seconds=wall_time,
            usage=usage_totals,
            error=error,
        )
        summary = {
            "task_id": result.task_id,
            "status": result.status,
            "prediction_path": (
                str(result.prediction_path)
                if result.prediction_path is not None
                else None
            ),
            "model_turns": result.model_turns,
            "tool_actions": result.tool_actions,
            "wall_time_seconds": round(result.wall_time_seconds, 6),
            "usage": result.usage,
            "run_metadata": self.run_metadata,
            "error": result.error,
            "finished_at": utc_now(),
        }
        logger.event("run_finished", summary)
        logger.write_summary(summary)
        return result
