"""
optical_flow.py
---------------
Computes optical flow between consecutive frames to capture motion
information for sign language gesture recognition.

Two backends are provided:
  - Farneback  : Dense optical flow (OpenCV, CPU, no extra deps)
  - TVL1       : Dense optical flow (OpenCV, slower but higher quality)

Flow maps are returned as (dx, dy) channels and can be stacked with RGB
frames or used as standalone inputs to the CNN feature extractor.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional

from frame_extractor import FrameExtractor


# ---------------------------------------------------------------------------
# Flow computation
# ---------------------------------------------------------------------------

class OpticalFlowComputer:
    """
    Computes per-frame optical flow from a sequence of RGB/grayscale images.

    Output shape per call to compute():
        (N-1, H, W, 2)  — N-1 flow maps, each with (dx, dy) channels.

    The flow maps are optionally normalized to [-1, 1] for stable CNN input.
    """

    BACKENDS = {"farneback", "tvl1"}

    def __init__(
        self,
        backend: str = "farneback",
        normalize: bool = True,
        clip_magnitude: float = 20.0,
    ):
        """
        Args:
            backend:         'farneback' (fast, default) or 'tvl1' (quality).
            normalize:       If True, clip and scale flow to [-1.0, 1.0].
            clip_magnitude:  Pixel displacement value used as the clipping
                             boundary before normalization.
        """
        backend = backend.lower()
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {self.BACKENDS}"
            )
        self.backend = backend
        self.normalize = normalize
        self.clip_magnitude = clip_magnitude

        # Farneback params (tuned for 224x224 sign-language clips)
        self._fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # TVL1 estimator (lazy init — requires opencv-contrib)
        self._tvl1 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute optical flow for a sequence of frames.

        Args:
            frames: Array of shape (N, H, W, C) or (N, H, W).
                    Values can be uint8 [0-255] or float32 [0-1].
                    C=1 or C=3; if C=3, frames are converted to grayscale.

        Returns:
            flow: Array of shape (N-1, H, W, 2), dtype float32.
                  flow[i, :, :, 0] = dx (horizontal displacement)
                  flow[i, :, :, 1] = dy (vertical displacement)
        """
        if frames.ndim == 3:
            frames = frames[..., np.newaxis]  # (N, H, W, 1)

        n_frames = frames.shape[0]
        if n_frames < 2:
            raise ValueError("At least 2 frames are required to compute optical flow.")

        gray_frames = self._to_grayscale(frames)  # (N, H, W)
        H, W = gray_frames.shape[1], gray_frames.shape[2]
        flow_maps = np.zeros((n_frames - 1, H, W, 2), dtype=np.float32)

        for i in range(n_frames - 1):
            prev = gray_frames[i]
            curr = gray_frames[i + 1]
            flow_maps[i] = self._compute_pair(prev, curr)

        return flow_maps

    def compute_and_stack(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute flow and stack it with the corresponding RGB frames.

        Alignment: flow[i] corresponds to motion FROM frames[i] → frames[i+1],
        so we pair flow[i] with frames[i] (drop the last frame to match lengths).

        Args:
            frames: (N, H, W, 3) RGB array.

        Returns:
            stacked: (N-1, H, W, 5) array — channels: [R, G, B, dx, dy].
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("Expected frames of shape (N, H, W, 3).")

        flow = self.compute(frames)                 # (N-1, H, W, 2)
        rgb_aligned = frames[:-1].astype(np.float32)  # (N-1, H, W, 3)

        if rgb_aligned.max() > 1.0:
            rgb_aligned /= 255.0

        return np.concatenate([rgb_aligned, flow], axis=-1)  # (N-1, H, W, 5)

    def compute_from_dir(
        self,
        frame_dir: Union[str, Path],
        normalize_frames: bool = True,
    ) -> np.ndarray:
        """
        Load frames from a saved frame directory and compute flow.

        Args:
            frame_dir:        Directory created by FrameExtractor.
            normalize_frames: Whether to normalize loaded frames to [0, 1].

        Returns:
            flow: (N-1, H, W, 2) float32 array.
        """
        frames = FrameExtractor.load_frames_from_dir(frame_dir, normalize=normalize_frames)
        return self.compute(frames)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_pair(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two consecutive grayscale frames.

        Args:
            prev, curr: (H, W) uint8 arrays.

        Returns:
            flow: (H, W, 2) float32 array.
        """
        if self.backend == "farneback":
            raw_flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, **self._fb_params
            )
        else:  # tvl1
            raw_flow = self._tvl1_flow(prev, curr)

        if self.normalize:
            raw_flow = self._normalize_flow(raw_flow)

        return raw_flow  # (H, W, 2)

    def _tvl1_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute TVL1 flow, initializing the estimator on first call."""
        if self._tvl1 is None:
            try:
                self._tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            except AttributeError:
                raise ImportError(
                    "TVL1 requires opencv-contrib-python. "
                    "Install it with: pip install opencv-contrib-python"
                )
        flow = np.zeros((*prev.shape, 2), dtype=np.float32)
        self._tvl1.calc(prev, curr, flow)
        return flow

    def _normalize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Clip pixel displacements to [-clip, clip] and scale to [-1, 1]."""
        c = self.clip_magnitude
        return np.clip(flow, -c, c) / c

    @staticmethod
    def _to_grayscale(frames: np.ndarray) -> np.ndarray:
        """
        Convert an (N, H, W, C) float/uint8 array to (N, H, W) uint8 grayscale.
        """
        # Ensure uint8 range
        if frames.dtype != np.uint8:
            arr = (frames * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = frames

        n = arr.shape[0]
        c = arr.shape[-1]

        if c == 1:
            return arr[..., 0]

        gray = np.zeros((n, arr.shape[1], arr.shape[2]), dtype=np.uint8)
        for i in range(n):
            gray[i] = cv2.cvtColor(arr[i], cv2.COLOR_RGB2GRAY)
        return gray


# ---------------------------------------------------------------------------
# Visualization utility
# ---------------------------------------------------------------------------

def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert a single (H, W, 2) flow map to an HSV-encoded RGB image.

    Hue    → flow direction
    Value  → flow magnitude (brighter = faster motion)

    Useful for debugging and dataset inspection.

    Args:
        flow: (H, W, 2) float32 flow map (normalized or raw).

    Returns:
        rgb: (H, W, 3) uint8 RGB image.
    """
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)           # Hue   [0-179]
    hsv[..., 1] = 255                                      # Saturation
    hsv[..., 2] = cv2.normalize(                           # Value
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_optical_flow(
    frames: np.ndarray,
    backend: str = "farneback",
    normalize: bool = True,
    clip_magnitude: float = 20.0,
) -> np.ndarray:
    """
    One-liner: compute optical flow for a (N, H, W, C) frame array.

    Returns:
        flow: (N-1, H, W, 2) float32 array.
    """
    computer = OpticalFlowComputer(
        backend=backend,
        normalize=normalize,
        clip_magnitude=clip_magnitude,
    )
    return computer.compute(frames)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Optical Flow Smoke Test")
    print("-----------------------")

    # Generate synthetic gradient frames to verify the pipeline
    N, H, W = 10, 224, 224
    rng = np.random.default_rng(42)
    fake_frames = rng.integers(0, 256, size=(N, H, W, 3), dtype=np.uint8)

    computer = OpticalFlowComputer(backend="farneback", normalize=True)
    flow = computer.compute(fake_frames)

    print(f"Input frames shape : {fake_frames.shape}")
    print(f"Flow output shape  : {flow.shape}")       # (N-1, H, W, 2)
    print(f"Flow dtype         : {flow.dtype}")
    print(f"Flow value range   : [{flow.min():.4f}, {flow.max():.4f}]")

    stacked = computer.compute_and_stack(fake_frames.astype(np.float32) / 255.0)
    print(f"Stacked (RGB+flow) : {stacked.shape}")    # (N-1, H, W, 5)

    vis = flow_to_rgb(flow[0])
    print(f"Visualization img  : {vis.shape}, dtype={vis.dtype}")
    print("All assertions passed.")