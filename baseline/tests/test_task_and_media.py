from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataspace_agent.media import ImageObserver
from dataspace_agent.task import TaskPaths, TaskSpec


class TaskAndMediaTests(unittest.TestCase):
    def test_loads_task_and_confines_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "task_7"
            context = task_dir / "context"
            output = root / "output"
            context.mkdir(parents=True)
            output.mkdir()
            (task_dir / "task.json").write_text(
                json.dumps(
                    {"task_id": "task_7", "question": "Compute the result."}
                ),
                encoding="utf-8",
            )
            (context / "data.txt").write_text("value", encoding="utf-8")

            task = TaskSpec.load(task_dir)
            paths = TaskPaths(context, output)
            resolved = paths.resolve_agent_path("/workspace/data.txt")

            self.assertEqual(task.task_id, "task_7")
            self.assertEqual(resolved, (context / "data.txt").resolve())
            with self.assertRaises(ValueError):
                paths.resolve_agent_path("/workspace/../../etc/passwd")
            with self.assertRaises(ValueError):
                paths.resolve_agent_path(
                    "/workspace/data.txt", output_only=True
                )

    def test_normalizes_image_to_common_max_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            output = root / "output"
            workspace.mkdir()
            output.mkdir()
            image_path = workspace / "large.png"
            Image.new("RGB", (2400, 1200), color=(20, 40, 60)).save(
                image_path
            )

            observer = ImageObserver(
                TaskPaths(workspace, output),
                max_edge=800,
                jpeg_quality=90,
            )
            observation = observer.observe("/workspace/large.png")

            self.assertTrue(
                observation.data_url.startswith(
                    "data:image/jpeg;base64,"
                )
            )
            self.assertEqual(
                observation.metadata["normalized_width"], 800
            )
            self.assertEqual(
                observation.metadata["normalized_height"], 400
            )


if __name__ == "__main__":
    unittest.main()
