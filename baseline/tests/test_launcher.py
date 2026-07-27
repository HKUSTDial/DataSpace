from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


class LauncherTests(unittest.TestCase):
    def test_requires_explicit_input_root(self) -> None:
        baseline_root = Path(__file__).resolve().parents[1]
        launcher = baseline_root / "scripts" / "run_benchmark.sh"
        completed = subprocess.run(
            (str(launcher), "model", "mimo-v2.5", "--foreground"),
            cwd=baseline_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--input-root PATH is required", completed.stderr)


if __name__ == "__main__":
    unittest.main()
