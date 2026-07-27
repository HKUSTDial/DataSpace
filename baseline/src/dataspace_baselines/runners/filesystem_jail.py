from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


JAIL_WORKSPACE = "/mnt/workspace"
JAIL_OUTPUT = "/mnt/output"
JAIL_HOME = "/mnt/home"
JAIL_RUNTIME = "/mnt/runtime"


@dataclass(frozen=True)
class BubblewrapJail:
    """Expose one read-only workspace and one writable output directory.

    Network namespaces are intentionally not changed here: the harness process
    must reach its model endpoint, while each harness's native command sandbox
    is responsible for blocking network access from model-generated code.
    """

    executable: str = "bwrap"
    user_runtime_root: Path = Path.home() / ".local"

    def command(
        self,
        *,
        task_context: Path,
        output_root: Path,
        home_root: Path,
        inner_command: tuple[str, ...],
        working_directory: str = JAIL_OUTPUT,
        runtime_binds: tuple[tuple[Path, str], ...] = (),
        rootfs_root: Path | None = None,
    ) -> tuple[str, ...]:
        executable = shutil.which(self.executable)
        if executable is None:
            raise FileNotFoundError(
                f"filesystem jail executable is unavailable: {self.executable}"
            )
        context = task_context.resolve()
        output = output_root.resolve()
        home = home_root.resolve()
        if not context.is_dir():
            raise FileNotFoundError(f"task context does not exist: {context}")
        output.mkdir(parents=True, exist_ok=True)
        home.mkdir(parents=True, exist_ok=True)
        (home / ".local").mkdir(parents=True, exist_ok=True)
        (home / ".grok" / "bin").mkdir(parents=True, exist_ok=True)
        (home / ".grok" / "downloads").mkdir(parents=True, exist_ok=True)

        root_source = Path("/")
        if rootfs_root is not None:
            root_source = rootfs_root.resolve()
            if not root_source.is_dir():
                raise FileNotFoundError(
                    f"data workbench rootfs does not exist: {root_source}"
                )
            tools_manifest = root_source / "opt" / "dataspace" / "TOOLS.md"
            if not tools_manifest.is_file():
                raise FileNotFoundError(
                    "data workbench tools manifest does not exist: "
                    f"{tools_manifest}"
                )

        command: list[str] = [
            executable,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind",
            str(root_source),
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            # Headless native sandboxes may require an openable /dev/tty when
            # constructing their Landlock rules. A null device satisfies that
            # requirement without exposing the host's controlling terminal.
            "--dev-bind",
            "/dev/null",
            "/dev/tty",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--tmpfs",
            "/mnt",
            "--dir",
            JAIL_WORKSPACE,
            "--dir",
            JAIL_OUTPUT,
            "--dir",
            JAIL_HOME,
        ]
        resolver = Path("/etc/resolv.conf").resolve()
        run_root = Path("/run")
        if rootfs_root is not None and resolver.is_file():
            command.extend(("--ro-bind", str(resolver), "/etc/resolv.conf"))
        elif resolver.is_file() and resolver.is_relative_to(run_root):
            # /etc/resolv.conf commonly points into /run/systemd/resolve.
            # Recreate only that path after masking /run so the harness
            # control process can resolve its model endpoint.
            relative_parent = resolver.parent.relative_to(run_root)
            current_parent = run_root
            for part in relative_parent.parts:
                current_parent /= part
                command.extend(("--dir", str(current_parent)))
            command.extend(("--ro-bind", str(resolver), str(resolver)))
        if runtime_binds:
            command.extend(("--dir", JAIL_RUNTIME))
        if rootfs_root is None:
            for hidden in ("/home", "/root", "/data"):
                if Path(hidden).exists():
                    command.extend(("--tmpfs", hidden))

        command.extend(
            (
                "--ro-bind",
                str(context),
                JAIL_WORKSPACE,
                "--bind",
                str(output),
                JAIL_OUTPUT,
                "--bind",
                str(home),
                JAIL_HOME,
            )
        )

        for source_path, target_path in runtime_binds:
            source = source_path.resolve()
            if not source.exists():
                raise FileNotFoundError(f"configured runtime does not exist: {source}")
            target = PurePosixPath(target_path)
            runtime_root = PurePosixPath(JAIL_RUNTIME)
            if (
                not target.is_absolute()
                or target == runtime_root
                or runtime_root not in target.parents
                or ".." in target.parts
            ):
                raise ValueError(
                    f"runtime mount must be a child of {JAIL_RUNTIME}: {target_path}"
                )
            command.extend(("--ro-bind", str(source), str(target)))

        if not runtime_binds and rootfs_root is None:
            runtime = self.user_runtime_root.resolve()
            if runtime.is_dir():
                command.extend(("--ro-bind", str(runtime), f"{JAIL_HOME}/.local"))
                # Backward-compatible fallback for explicitly configured
                # host runtimes. Released configs use project-local mounts.
                command.extend(
                    (
                        "--dir",
                        str(self.user_runtime_root),
                        "--ro-bind",
                        str(runtime),
                        str(self.user_runtime_root),
                    )
                )
            grok_binary_root = Path.home() / ".grok" / "bin"
            if grok_binary_root.is_dir():
                command.extend(
                    (
                        "--ro-bind",
                        str(grok_binary_root.resolve()),
                        f"{JAIL_HOME}/.grok/bin",
                    )
                )
            grok_download_root = Path.home() / ".grok" / "downloads"
            if grok_download_root.is_dir():
                command.extend(
                    (
                        "--ro-bind",
                        str(grok_download_root.resolve()),
                        f"{JAIL_HOME}/.grok/downloads",
                    )
                )

        command.extend(
            (
                "--setenv",
                "HOME",
                JAIL_HOME,
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "DATASPACE_TOOLS_MANIFEST",
                "/opt/dataspace/TOOLS.md",
                "--setenv",
                "PYTHONDONTWRITEBYTECODE",
                "1",
                "--setenv",
                "PYTHONUNBUFFERED",
                "1",
                "--setenv",
                "OPENBLAS_NUM_THREADS",
                "1",
                "--setenv",
                "OMP_NUM_THREADS",
                "1",
                "--setenv",
                "MKL_NUM_THREADS",
                "1",
                "--setenv",
                "NUMEXPR_NUM_THREADS",
                "1",
                "--setenv",
                "PATH",
                f"{JAIL_RUNTIME}/codex/bin:"
                f"{JAIL_RUNTIME}/claude-code/bin:"
                f"{JAIL_RUNTIME}/grok-build/bin:"
                f"{JAIL_HOME}/.local/node/bin:{JAIL_HOME}/.local/bin:"
                f"{JAIL_HOME}/.grok/bin:"
                "/usr/local/bin:/usr/bin:/bin",
                "--chdir",
                working_directory,
                "--",
                *inner_command,
            )
        )
        return tuple(command)


def minimal_harness_environment(api_key_env: str) -> dict[str, str]:
    """Pass only inference credentials and deterministic process basics."""

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"environment variable {api_key_env} is not set")
    return {
        api_key_env: api_key,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TERM": "dumb",
        "NO_COLOR": "1",
    }
