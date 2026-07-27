from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps

from .task import TaskPaths
from .types import ImageObservation


class ImageObserver:
    def __init__(
        self,
        paths: TaskPaths,
        max_edge: int,
        jpeg_quality: int,
    ):
        self.paths = paths
        self.max_edge = max_edge
        self.jpeg_quality = jpeg_quality

    def observe(self, agent_path: str) -> ImageObservation:
        host_path = self.paths.resolve_agent_path(agent_path)
        if not host_path.is_file():
            raise ValueError(f"image path is not a file: {agent_path}")
        source_bytes = host_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()

        with Image.open(io.BytesIO(source_bytes)) as opened:
            image = ImageOps.exif_transpose(opened)
            original_size = image.size
            image = image.convert("RGB")
            image.thumbnail(
                (self.max_edge, self.max_edge),
                Image.Resampling.LANCZOS,
            )
            encoded = io.BytesIO()
            image.save(
                encoded,
                format="JPEG",
                quality=self.jpeg_quality,
                optimize=True,
            )
            normalized_size = image.size

        normalized_bytes = encoded.getvalue()
        data_url = (
            "data:image/jpeg;base64,"
            + base64.b64encode(normalized_bytes).decode("ascii")
        )
        metadata = {
            "agent_path": agent_path,
            "source_sha256": source_sha256,
            "source_bytes": len(source_bytes),
            "original_width": original_size[0],
            "original_height": original_size[1],
            "normalized_width": normalized_size[0],
            "normalized_height": normalized_size[1],
            "normalized_bytes": len(normalized_bytes),
            "encoding": "image/jpeg",
            "jpeg_quality": self.jpeg_quality,
        }
        return ImageObservation(
            agent_path=agent_path,
            data_url=data_url,
            metadata=metadata,
        )
