from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Protocol

from .config import SandboxConfig
from .types import CommandResult


def docker_image_metadata(image: str) -> dict[str, object]:
    metadata: dict[str, object] = {"image": image}
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        payload = json.loads(result.stdout) if result.returncode == 0 else []
        inspected = payload[0] if isinstance(payload, list) and payload else {}
        if isinstance(inspected, dict):
            image_id = inspected.get("Id")
            created = inspected.get("Created")
            repo_digests = inspected.get("RepoDigests")
            if isinstance(image_id, str):
                metadata["image_id"] = image_id
            if isinstance(created, str):
                metadata["created"] = created
            if isinstance(repo_digests, list):
                metadata["repo_digests"] = [
                    value for value in repo_digests if isinstance(value, str)
                ]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return metadata


class Sandbox(Protocol):
    def start(self) -> None:
        ...

    def execute(self, command: str, timeout_seconds: int) -> CommandResult:
        ...

    def close(self) -> None:
        ...


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head_size = max(1, (limit * 2) // 3)
    tail_size = max(1, limit - head_size)
    removed = len(text) - head_size - tail_size
    marker = f"\n... [{removed} characters truncated] ...\n"
    return text[:head_size] + marker + text[-tail_size:], True


class DockerSandbox:
    def __init__(
        self,
        workspace_root: Path,
        output_root: Path,
        config: SandboxConfig,
        max_output_chars: int,
    ):
        self.workspace_root = workspace_root.resolve()
        self.output_root = output_root.resolve()
        self.config = config
        self.max_output_chars = max_output_chars
        self.container_name = f"dataspace-agent-{uuid.uuid4().hex[:12]}"
        self._started = False

    def start(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--init",
            "--name",
            self.container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.config.pids_limit),
            "--cpus",
            str(self.config.cpus),
            "--memory",
            self.config.memory,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "MKL_NUM_THREADS=1",
            "--env",
            "NUMEXPR_NUM_THREADS=1",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.config.tmpfs_size}",
            "--mount",
            f"type=bind,src={self.workspace_root},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={self.output_root},dst=/output",
            "--workdir",
            "/workspace",
            self.config.image,
            "sleep",
            "infinity",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "failed to start Docker sandbox: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        self._started = True

    def execute(self, command: str, timeout_seconds: int) -> CommandResult:
        if not self._started:
            raise RuntimeError("sandbox is not running")
        started = time.monotonic()
        docker_command = [
            "docker",
            "exec",
            "--workdir",
            "/workspace",
            self.container_name,
            "timeout",
            "--signal=KILL",
            f"{timeout_seconds}s",
            "bash",
            "-lc",
            command,
        ]
        try:
            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds + 10,
                check=False,
            )
            timed_out = result.returncode in {124, 137}
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout or ""
            )
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr or ""
            )
            stderr += "\nHost-side command timeout expired."

        stdout, stdout_truncated = _truncate(
            stdout, self.max_output_chars
        )
        stderr, stderr_truncated = _truncate(
            stderr, self.max_output_chars
        )
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def close(self) -> None:
        if not self._started:
            return
        subprocess.run(
            ["docker", "rm", "--force", self.container_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self._started = False
