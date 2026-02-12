"""
video_loader.py
---------------
Handles loading of video input from webcam streams or uploaded .mp4 files.
Standardizes frame rate to 25–30 FPS and resizes frames to 224x224.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Union


TARGET_FPS = 25
TARGET_SIZE = (224, 224)


class VideoLoader:
    """
    Loads video from a file path or webcam index.
    Yields normalized, resized frames at a standardized frame rate.
    """

    def __init__(
        self,
        source: Union[str, int, Path],
        target_fps: int = TARGET_FPS,
        target_size: tuple = TARGET_SIZE,
        normalize: bool = True,
    ):
        """
        Args:
            source:      Path to .mp4 file, or integer webcam device index (e.g. 0).
            target_fps:  Desired output frame rate (default 25).
            target_size: (width, height) to resize each frame to (default 224x224).
            normalize:   If True, pixel values are scaled to [0.0, 1.0].
        """
        self.source = source
        self.target_fps = target_fps
        self.target_size = target_size
        self.normalize = normalize

        self._cap: cv2.VideoCapture = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video capture device or file."""
        self._cap = cv2.VideoCapture(
            str(self.source) if not isinstance(self.source, int) else self.source
        )
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source!r}")

    def close(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def source_fps(self) -> float:
        """Return the native FPS of the source (0.0 if unknown)."""
        self._require_open()
        return self._cap.get(cv2.CAP_PROP_FPS) or 0.0

    @property
    def frame_count(self) -> int:
        """Total frames in a file source (-1 for live webcam)."""
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_seconds(self) -> float:
        """Estimated duration in seconds (file sources only)."""
        fps = self.source_fps
        count = self.frame_count
        return count / fps if fps > 0 and count > 0 else -1.0

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield preprocessed frames one at a time.

        Each frame is:
          - Resized to target_size (width x height)
          - Color-converted from BGR → RGB
          - Optionally normalized to [0.0, 1.0] float32

        Frame-rate thinning is applied when the source FPS exceeds target_fps,
        so that the yielded sequence approximates target_fps.
        """
        self._require_open()

        src_fps = self.source_fps
        # Determine frame step: how many source frames to skip per output frame.
        # Falls back to 1 (no thinning) when source FPS is unknown or below target.
        if src_fps > 0 and src_fps > self.target_fps:
            frame_step = round(src_fps / self.target_fps)
        else:
            frame_step = 1

        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                yield self._preprocess(frame)

            frame_idx += 1

    def load_all(self) -> np.ndarray:
        """
        Load and return all frames as a single NumPy array.

        Returns:
            Array of shape (N, H, W, C) where N is the number of frames.
        """
        frame_list = list(self.frames())
        if not frame_list:
            raise ValueError("No frames could be read from source.")
        return np.stack(frame_list, axis=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, recolor, and optionally normalize a single BGR frame."""
        # Resize
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame

    def _require_open(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoLoader is not open. Call open() first.")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_video(
    source: Union[str, int, Path],
    target_fps: int = TARGET_FPS,
    target_size: tuple = TARGET_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load all frames from a video file or webcam into a NumPy array.

    Args:
        source:      File path or webcam index.
        target_fps:  Desired output FPS.
        target_size: (width, height) resize target.
        normalize:   Scale pixels to [0.0, 1.0] if True.

    Returns:
        Array of shape (N, H, W, 3).
    """
    with VideoLoader(source, target_fps, target_size, normalize) as loader:
        return loader.load_all()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else 0  # default: webcam
    print(f"Opening source: {src!r}")

    with VideoLoader(src) as vl:
        print(f"  Source FPS   : {vl.source_fps:.2f}")
        print(f"  Frame count  : {vl.frame_count}")
        print(f"  Duration (s) : {vl.duration_seconds:.2f}")
        for i, frame in enumerate(vl.frames()):
            if i == 0:
                print(f"  Frame shape  : {frame.shape}")
                print(f"  Dtype        : {frame.dtype}")
                print(f"  Value range  : [{frame.min():.3f}, {frame.max():.3f}]")
            if i >= 9:  # inspect first 10 frames only
                break
    print("Done.")