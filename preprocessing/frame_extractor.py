"""
frame_extractor.py
------------------
Extracts frames from video sources and saves them to disk as part of
the preprocessing pipeline for the railway sign language dataset.

Designed to work with VideoLoader for consistent frame standardization.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List

from video_loader import VideoLoader, TARGET_FPS, TARGET_SIZE


class FrameExtractor:
    """
    Extracts and saves frames from a video file or webcam stream.

    Output structure per video:
        output_dir/
        └── <video_stem>/
            ├── frame_0000.jpg
            ├── frame_0001.jpg
            └── ...

    Frames are saved as uint8 JPEGs (denormalized if needed).
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        target_fps: int = TARGET_FPS,
        target_size: tuple = TARGET_SIZE,
        max_frames: Optional[int] = None,
        image_format: str = "jpg",
        jpeg_quality: int = 95,
    ):
        """
        Args:
            output_dir:    Root directory where frame folders are saved.
            target_fps:    FPS used when loading the source video.
            target_size:   (width, height) for each frame.
            max_frames:    Cap on the number of frames to extract per video.
                           None means extract all.
            image_format:  'jpg' or 'png'.
            jpeg_quality:  JPEG compression quality (1–100, ignored for PNG).
        """
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.target_size = target_size
        self.max_frames = max_frames
        self.image_format = image_format.lower().lstrip(".")
        self.jpeg_quality = jpeg_quality

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_file(self, video_path: Union[str, Path]) -> List[Path]:
        """
        Extract frames from a single video file and save them to disk.

        Args:
            video_path: Path to the .mp4 (or any OpenCV-supported) video.

        Returns:
            List of paths to saved frame images.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        frame_dir = self._make_frame_dir(video_path.stem)
        return self._extract_and_save(str(video_path), frame_dir)

    def extract_from_dataset(
        self,
        dataset_root: Union[str, Path],
        label_subdirs: bool = True,
    ) -> dict:
        """
        Batch-extract frames from an entire dataset directory.

        Expected layout when label_subdirs=True:
            dataset_root/
            ├── train/
            │   ├── video_01.mp4
            │   └── video_02.mp4
            ├── platform/
            └── ...

        Args:
            dataset_root:   Root of the raw dataset.
            label_subdirs:  If True, mirrors the label subdirectory structure
                            in the output directory.

        Returns:
            Dict mapping label → list of frame directories.
        """
        dataset_root = Path(dataset_root)
        results = {}

        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

        if label_subdirs:
            for label_dir in sorted(dataset_root.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = label_dir.name
                results[label] = []
                label_out = self.output_dir / label
                label_out.mkdir(parents=True, exist_ok=True)

                for video_file in sorted(label_dir.iterdir()):
                    if video_file.suffix.lower() not in video_extensions:
                        continue
                    frame_dir = label_out / video_file.stem
                    saved = self._extract_and_save(str(video_file), frame_dir)
                    if saved:
                        results[label].append(frame_dir)
                        print(
                            f"  [{label}] {video_file.name} → {len(saved)} frames"
                        )
        else:
            results["all"] = []
            for video_file in sorted(dataset_root.rglob("*")):
                if video_file.suffix.lower() not in video_extensions:
                    continue
                frame_dir = self.output_dir / video_file.stem
                saved = self._extract_and_save(str(video_file), frame_dir)
                if saved:
                    results["all"].append(frame_dir)

        return results

    def extract_from_webcam(
        self,
        device_index: int = 0,
        duration_seconds: float = 3.0,
        clip_name: str = "webcam_clip",
    ) -> List[Path]:
        """
        Capture a fixed-duration clip from a webcam and save its frames.

        Args:
            device_index:     OpenCV device index (default 0).
            duration_seconds: How many seconds to capture.
            clip_name:        Subfolder name for the saved frames.

        Returns:
            List of paths to saved frame images.
        """
        frame_dir = self._make_frame_dir(clip_name)
        max_frames = int(duration_seconds * self.target_fps)
        return self._extract_and_save(device_index, frame_dir, max_frames=max_frames)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_frame_dir(self, name: str) -> Path:
        """Create and return a subdirectory for a video's frames."""
        frame_dir = self.output_dir / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir

    def _extract_and_save(
        self,
        source: Union[str, int],
        frame_dir: Path,
        max_frames: Optional[int] = None,
    ) -> List[Path]:
        """
        Core extraction loop: reads frames via VideoLoader and saves to disk.

        Args:
            source:     File path or webcam index passed to VideoLoader.
            frame_dir:  Directory where frames will be written.
            max_frames: Override for self.max_frames (used for webcam capture).

        Returns:
            List of saved frame paths.
        """
        limit = max_frames if max_frames is not None else self.max_frames
        saved_paths: List[Path] = []

        encode_param = (
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            if self.image_format == "jpg"
            else []
        )

        with VideoLoader(
            source,
            target_fps=self.target_fps,
            target_size=self.target_size,
            normalize=False,  # Save raw uint8 to disk
        ) as loader:
            for idx, frame in enumerate(loader.frames()):
                if limit is not None and idx >= limit:
                    break

                # frame is RGB uint8; convert to BGR for OpenCV saving
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                filename = frame_dir / f"frame_{idx:04d}.{self.image_format}"
                cv2.imwrite(str(filename), bgr_frame, encode_param)
                saved_paths.append(filename)

        return saved_paths

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_frames_from_dir(
        frame_dir: Union[str, Path],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Load previously saved frame images back into a NumPy array.

        Args:
            frame_dir:  Directory containing frame_XXXX.jpg/png files.
            normalize:  Scale to [0.0, 1.0] float32 if True.

        Returns:
            Array of shape (N, H, W, 3) in RGB order.
        """
        frame_dir = Path(frame_dir)
        image_paths = sorted(frame_dir.glob("frame_*.jpg")) + sorted(
            frame_dir.glob("frame_*.png")
        )
        if not image_paths:
            raise ValueError(f"No frame images found in {frame_dir}")

        frames = []
        for path in image_paths:
            bgr = cv2.imread(str(path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if normalize:
                rgb = rgb.astype(np.float32) / 255.0
            frames.append(rgb)

        return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    target_fps: int = TARGET_FPS,
    target_size: tuple = TARGET_SIZE,
    max_frames: Optional[int] = None,
) -> List[Path]:
    """
    One-liner: extract frames from a video and save them to output_dir.

    Returns:
        List of saved frame paths.
    """
    extractor = FrameExtractor(
        output_dir=output_dir,
        target_fps=target_fps,
        target_size=target_size,
        max_frames=max_frames,
    )
    return extractor.extract_from_file(video_path)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python frame_extractor.py <video_path> <output_dir>")
        sys.exit(1)

    video_src = sys.argv[1]
    out_dir = sys.argv[2]

    print(f"Extracting frames from: {video_src}")
    print(f"Output directory      : {out_dir}")

    saved = extract_frames(video_src, out_dir, max_frames=50)
    print(f"Saved {len(saved)} frames.")

    if saved:
        frames = FrameExtractor.load_frames_from_dir(Path(out_dir) / Path(video_src).stem)
        print(f"Reloaded array shape : {frames.shape}")
        print(f"Dtype / range        : {frames.dtype}, [{frames.min():.3f}, {frames.max():.3f}]")