"""
pipeline/video_utils.py

Thin wrappers around OpenCV VideoCapture / VideoWriter with:
  - Context-manager interface (no leaked file handles)
  - FPS negotiation (caps to target_fps without duplicating frames)
  - Resolution / codec helpers
  - Frame iterator with progress reporting
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple
import time


class VideoReader:
    """Context-manager wrapper around cv2.VideoCapture."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        return self

    def __exit__(self, *_):
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def shape(self) -> Tuple[int, int]:
        """Returns (height, width)."""
        return self.height, self.width

    def frames(
        self,
        target_fps: Optional[float] = None,
        verbose: bool = True,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_index, frame_bgr) for every frame we should process.

        If target_fps < source_fps, frames are uniformly sub-sampled so the
        output video runs at target_fps without temporal aliasing.
        """
        src_fps    = self.fps
        total      = self.frame_count
        target_fps = target_fps or src_fps

        # Sub-sampling: only process every Nth frame
        step = max(1, round(src_fps / target_fps))

        frame_idx  = 0
        kept_idx   = 0
        t_start    = time.time()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                if verbose and kept_idx % 30 == 0:
                    elapsed = time.time() - t_start + 1e-6
                    print(f"  frame {frame_idx:5d}/{total}  "
                          f"({100*frame_idx/max(total,1):.1f}%)  "
                          f"{kept_idx/elapsed:.1f} fps processed")
                yield kept_idx, frame
                kept_idx += 1

            frame_idx += 1

    def read_first_n(self, n: int) -> list[np.ndarray]:
        """Read the first N frames (for calibration / bootstrap)."""
        frames = []
        for _ in range(n):
            ret, frame = self._cap.read()
            if not ret:
                break
            frames.append(frame)
        # Rewind
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frames


class VideoWriter:
    """Context-manager wrapper around cv2.VideoWriter."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        fourcc: str = "mp4v",
        grayscale: bool = False,
    ):
        self.path      = Path(path)
        self._fps      = fps
        self._width    = width
        self._height   = height
        self._fourcc   = fourcc
        self._gray     = grayscale
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
        # OpenCV VideoWriter always writes colour; we convert inside write()
        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, self._fps,
            (self._width, self._height),
            isColor=True,
        )
        if not self._writer.isOpened():
            raise IOError(f"Cannot open video writer: {self.path}")
        return self

    def __exit__(self, *_):
        if self._writer:
            self._writer.release()
            self._writer = None

    def write(self, frame: np.ndarray):
        """Write one frame.  Accepts both grayscale (H,W) and BGR (H,W,3)."""
        if frame.ndim == 2:
            # Grayscale → BGR so VideoWriter is happy
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self._writer.write(frame)


def build_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    suffix: str = "_mask",
) -> Path:
    """Derive output filename from input filename."""
    p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{p.stem}{suffix}{p.suffix}"
