"""
Quick start
-----------
python -m src.temporal_consistency \\
    --video-path outputs/examples/baseline_press_conference.mp4

Expected output
---------------
- Prints temporal consistency score for the provided MP4.
- Writes an optional smoothed copy when `--write-smoothed` is supplied.

Description
-----------
Temporal consistency utilities for evaluating and post-processing generated
video clips. Implements an optical-flow-inspired score, simple temporal
smoothing, and background blending helpers used across the pipeline.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch

try:
    import cv2
    from cv2 import optflow as cv2_optflow
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    cv2_optflow = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class TemporalConsistencyConfig:
    """Configuration used when smoothing or rescoring frames."""

    temporal_alpha: float = 0.7
    background_blend: float = 0.1
    fps: int = 8


class TemporalConsistencyEvaluator:
    """Compute lightweight temporal consistency metrics for generated clips."""

    def __init__(self, fps: int = 8, variance_scale: float = 5.0) -> None:
        self.fps = fps
        self.variance_scale = variance_scale

        if cv2_optflow is None:
            raise ImportError(
                "OpenCV optical flow module not available. Install `opencv-contrib-python`."
            )
        self._flow_estimator = cv2_optflow.DualTVL1OpticalFlow_create()

    @staticmethod
    def _ensure_tensor(video: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(video, torch.Tensor):
            return video.float()
        array = torch.from_numpy(video).float()
        if array.ndim == 4 and array.shape[-1] in (3, 4):
            array = array.permute(0, 3, 1, 2)
        return array

    def _frame_to_gray(self, frame: torch.Tensor) -> np.ndarray:
        array = frame.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in (3, 4):
            array = np.transpose(array[:3], (1, 2, 0))
        if array.max() <= 1.0:
            array = (array * 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        return gray

    def _flow_magnitudes(self, tensor: torch.Tensor) -> np.ndarray:
        magnitudes: list[float] = []
        prev_gray = self._frame_to_gray(tensor[0])

        for idx in range(1, tensor.shape[0]):
            next_gray = self._frame_to_gray(tensor[idx])
            flow = self._flow_estimator.calc(prev_gray, next_gray, None)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(mag)))
            prev_gray = next_gray
        return np.array(magnitudes, dtype=np.float32)

    def score_tensor_video(self, frames: torch.Tensor | np.ndarray) -> float:
        """Return a 0-1 optical-flow stability score."""

        tensor = self._ensure_tensor(frames)
        if tensor.ndim != 4:
            raise ValueError("Expected frames with shape (T, C, H, W)")
        if tensor.shape[0] < 2:
            return 1.0

        magnitudes = self._flow_magnitudes(tensor)
        variance = float(np.var(magnitudes))
        score = float(np.exp(-variance / max(self.variance_scale, 1e-6)))
        return float(np.clip(score, 0.0, 1.0))

    def score_video_file(self, video_path: Path) -> float:
        """Load a video file and compute the temporal consistency score."""

        LOGGER.info("Loading video from %s", video_path)
        frames = iio.imread(video_path)
        score = self.score_tensor_video(frames)
        LOGGER.info("Temporal consistency score: %.3f", score)
        return score


def temporal_smoothing(
    frames: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """Apply exponential moving average smoothing across frames."""

    smoothed = np.copy(frames)
    for idx in range(1, smoothed.shape[0]):
        smoothed[idx] = alpha * smoothed[idx] + (1 - alpha) * smoothed[idx - 1]
    return smoothed


def blend_background(
    frames: np.ndarray,
    background: np.ndarray,
    strength: float = 0.1,
) -> np.ndarray:
    """Blend frames with a reference background to reduce flicker."""

    if background.shape != frames.shape[1:]:
        raise ValueError("background must match frame spatial dimensions")
    blended = frames * (1 - strength) + background * strength
    return blended


def iter_video_frames(video_path: Path) -> Iterable[np.ndarray]:
    """Yield frames from disk as uint8 arrays."""

    for frame in iio.imiter(video_path):
        yield frame


def write_video(frames: np.ndarray, output_path: Path, fps: int = 8) -> None:
    """Write frames back to H.264 encoded MP4 using imageio."""

    LOGGER.info("Writing smoothed video to %s", output_path)
    iio.imwrite(output_path, frames, fps=fps, codec="h264", quality=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal consistency evaluator.")
    parser.add_argument("--video-path", type=Path, required=True, help="Path to MP4 file to evaluate.")
    parser.add_argument("--write-smoothed", type=Path, help="Optional path to store smoothed video.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Temporal smoothing alpha.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    evaluator = TemporalConsistencyEvaluator()
    score = evaluator.score_video_file(args.video_path)
    LOGGER.info("Temporal consistency >= 0.6 ? %s", score >= 0.6)

    if args.write_smoothed:
        frames = np.stack(list(iter_video_frames(args.video_path)))
        smoothed = temporal_smoothing(frames, alpha=args.alpha)
        write_video(smoothed, args.write_smoothed, fps=evaluator.fps)


if __name__ == "__main__":
    main()

