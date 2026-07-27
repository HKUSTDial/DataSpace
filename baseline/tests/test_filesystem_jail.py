from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from dataspace_baselines.runners.filesystem_jail import (
    JAIL_OUTPUT,
    JAIL_WORKSPACE,
    BubblewrapJail,
)


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap is unavailable")
class BubblewrapJailTests(unittest.TestCase):
    def test_exposes_tty_path_for_nested_native_sandboxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context"
            context.mkdir()
            command = BubblewrapJail().command(
                task_context=context,
                output_root=root / "output",
                home_root=root / "home",
                inner_command=(
                    "/usr/bin/python3",
                    "-c",
                    "import os; fd = os.open('/dev/tty', os.O_RDWR); os.close(fd)",
                ),
            )
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", errors="replace"),
        )
        tty_mount = ("--dev-bind", "/dev/null", "/dev/tty")
        self.assertTrue(
            any(
                command[index : index + 3] == tty_mount
                for index in range(len(command) - 2)
            )
        )

    def test_exposes_only_read_only_workspace_and_writable_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context"
            output = root / "output"
            home = root / "home"
            context.mkdir()
            output.mkdir()
            (context / "data.txt").write_text("visible\n", encoding="utf-8")
            secret = root / "gold.csv"
            secret.write_text("secret\n", encoding="utf-8")

            inner = (
                "/bin/sh",
                "-c",
                " && ".join(
                    (
                        "test ! -e " + str(secret),
                        f"cat {JAIL_WORKSPACE}/data.txt > {JAIL_OUTPUT}/copied.txt",
                        f"! touch {JAIL_WORKSPACE}/forbidden.txt",
                        f"touch {JAIL_OUTPUT}/allowed.txt",
                    )
                ),
            )
            command = BubblewrapJail().command(
                task_context=context,
                output_root=output,
                home_root=home,
                inner_command=inner,
            )
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr.decode("utf-8", errors="replace"),
            )
            self.assertEqual(
                (output / "copied.txt").read_text(encoding="utf-8"),
                "visible\n",
            )
            self.assertTrue((output / "allowed.txt").is_file())
            self.assertFalse((context / "forbidden.txt").exists())

    def test_exposes_configured_runtime_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context"
            output = root / "output"
            home = root / "home"
            runtime = root / "runtime"
            for directory in (context, output, runtime):
                directory.mkdir()
            (runtime / "version.txt").write_text("0.80.0\n", encoding="utf-8")

            command = BubblewrapJail().command(
                task_context=context,
                output_root=output,
                home_root=home,
                runtime_binds=((runtime, "/mnt/runtime/codex-test"),),
                inner_command=(
                    "/bin/sh",
                    "-c",
                    "cat /mnt/runtime/codex-test/version.txt > "
                    f"{JAIL_OUTPUT}/runtime-version.txt && "
                    "! touch /mnt/runtime/codex-test/forbidden.txt",
                ),
            )
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr.decode("utf-8", errors="replace"),
            )
            self.assertEqual(
                (output / "runtime-version.txt").read_text(encoding="utf-8"),
                "0.80.0\n",
            )
            self.assertFalse((runtime / "forbidden.txt").exists())
            host_runtime_mount = (
                "--ro-bind",
                str(BubblewrapJail().user_runtime_root.resolve()),
                "/mnt/home/.local",
            )
            self.assertFalse(
                any(
                    command[index : index + 3] == host_runtime_mount
                    for index in range(len(command) - 2)
                )
            )

    def test_masks_common_host_data_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context"
            context.mkdir()
            command = BubblewrapJail().command(
                task_context=context,
                output_root=root / "output",
                home_root=root / "home",
                inner_command=("/bin/true",),
            )

        for hidden in ("/home", "/root", "/data"):
            mount = ("--tmpfs", hidden)
            if Path(hidden).exists():
                self.assertTrue(
                    any(
                        command[index : index + 2] == mount
                        for index in range(len(command) - 1)
                    )
                )

    def test_can_replace_host_root_with_data_workbench_rootfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context"
            rootfs = root / "rootfs"
            context.mkdir()
            tools = rootfs / "opt/dataspace/TOOLS.md"
            tools.parent.mkdir(parents=True)
            tools.write_text("test tools\n", encoding="utf-8")
            command = BubblewrapJail().command(
                task_context=context,
                output_root=root / "output",
                home_root=root / "home",
                rootfs_root=rootfs,
                inner_command=("/bin/true",),
            )

        root_mount = ("--ro-bind", str(rootfs.resolve()), "/")
        self.assertTrue(
            any(
                command[index : index + 3] == root_mount
                for index in range(len(command) - 2)
            )
        )
        self.assertIn("DATASPACE_TOOLS_MANIFEST", command)
        self.assertIn("OPENBLAS_NUM_THREADS", command)
        self.assertNotIn(str(BubblewrapJail().user_runtime_root), command)

    def test_preserves_resolver_file_when_run_is_masked(self) -> None:
        resolver = Path("/etc/resolv.conf").resolve()
        if not resolver.is_relative_to(Path("/run")):
            self.skipTest("host resolver is not stored below /run")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context"
            context.mkdir()
            command = BubblewrapJail().command(
                task_context=context,
                output_root=root / "output",
                home_root=root / "home",
                inner_command=("/bin/true",),
            )

        resolver_mount = ("--ro-bind", str(resolver), str(resolver))
        self.assertTrue(
            any(
                command[index : index + 3] == resolver_mount
                for index in range(len(command) - 2)
            )
        )


if __name__ == "__main__":
    unittest.main()
