"""
Quick start
-----------
python -m src.metrics --video outputs/examples/baseline_press_conference.mp4

Expected output
---------------
- Prints temporal consistency, average frame brightness, and duration.
- Optionally appends results to `outputs/metrics.csv`.

Description
-----------
Utilities for computing quantitative metrics on generated videos such as
temporal consistency, VRAM estimates, and latency aggregates. Used during
phase acceptance and final reporting.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import imageio.v3 as iio
import numpy as np

from .temporal_consistency import TemporalConsistencyEvaluator


LOGGER = logging.getLogger(__name__)


@dataclass
class MetricResult:
    video_path: Path
    temporal_consistency: float
    avg_brightness: float
    duration_seconds: float
    fps: int

    def to_row(self) -> Dict[str, float | str]:
        return {
            "video_path": str(self.video_path),
            "temporal_consistency": self.temporal_consistency,
            "avg_brightness": self.avg_brightness,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
        }


def analyze_video(video_path: Path) -> MetricResult:
    LOGGER.info("Analyzing video %s", video_path)
    frames = iio.imread(video_path, index=range(0, None))
    fps = int(iio.immeta(video_path).get("fps", 8))
    duration = frames.shape[0] / fps
    evaluator = TemporalConsistencyEvaluator(fps=fps)
    temporal_score = evaluator.score_tensor_video(frames)
    brightness = float(frames.mean() / 255.0)
    return MetricResult(
        video_path=video_path,
        temporal_consistency=temporal_score,
        avg_brightness=brightness,
        duration_seconds=duration,
        fps=fps,
    )


def write_metrics(result: MetricResult, metrics_path: Optional[Path]) -> None:
    if not metrics_path:
        return
    is_new = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result.to_row().keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(result.to_row())
    LOGGER.info("Appended metrics to %s", metrics_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics for generated videos.")
    parser.add_argument("--video", type=Path, required=True, help="Video file to analyze.")
    parser.add_argument("--metrics-path", type=Path, help="Optional CSV metrics log path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    result = analyze_video(args.video)
    LOGGER.info(
        "Video %s | temporal=%.3f brightness=%.3f duration=%.2fs",
        result.video_path,
        result.temporal_consistency,
        result.avg_brightness,
        result.duration_seconds,
    )
    write_metrics(result, args.metrics_path)


if __name__ == "__main__":
    main()

