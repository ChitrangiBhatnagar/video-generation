"""
Quick start
-----------
python -m src.evaluation.benchmark \\
    --input-image data/samples/sample_image_1.png \\
    --prompt "Reporters preparing for a media briefing" \\
    --mode optimized

Expected output
---------------
- Runs baseline and/or optimized inference for three repetitions.
- Reports average latency, temporal consistency, and VRAM metrics.

Description
-----------
Benchmark harness for comparing baseline versus optimized inference settings.
Supports CSV export and seed control for reproducible measurements.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..baseline_inference import BaselineInferenceConfig, generate_video as baseline_generate, load_pipeline as load_baseline
from ..optimized_inference import (
    OptimizedInferenceConfig,
    benchmark_generation as optimized_generate,
    load_pipeline as load_optimized,
)
from ..temporal_consistency import TemporalConsistencyEvaluator
from ..utils.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    mode: str
    latency_avg: float
    latency_std: float
    temporal_avg: float
    temporal_std: float
    max_vram_gb: float


def run_iterations(
    mode: str,
    image: Path,
    prompt: str,
    repetitions: int,
    lora: Optional[Path] = None,
) -> BenchmarkResult:
    evaluator = TemporalConsistencyEvaluator()
    latencies: List[float] = []
    temporal_scores: List[float] = []

    torch.cuda.reset_peak_memory_stats()

    if mode == "baseline":
        config = BaselineInferenceConfig()
        pipeline = load_baseline(config)
        generator = lambda: baseline_generate(pipeline, config, image, prompt)  # noqa: E731
    else:
        config = OptimizedInferenceConfig()
        pipeline = load_optimized(config)
        generator = lambda: optimized_generate(pipeline, config, image, prompt, lora)  # noqa: E731

    for _ in range(repetitions):
        start = time.time()
        video_path = generator()
        latency = time.time() - start
        score = evaluator.score_video_file(video_path)
        latencies.append(latency)
        temporal_scores.append(score)

    max_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    return BenchmarkResult(
        mode=mode,
        latency_avg=statistics.mean(latencies),
        latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        temporal_avg=statistics.mean(temporal_scores),
        temporal_std=statistics.stdev(temporal_scores) if len(temporal_scores) > 1 else 0.0,
        max_vram_gb=max_vram,
    )


def write_csv(result: BenchmarkResult, csv_path: Path) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "mode",
                "latency_avg",
                "latency_std",
                "temporal_avg",
                "temporal_std",
                "max_vram_gb",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(result.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs optimized inference.")
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--mode", choices=["baseline", "optimized", "both"], default="both")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--lora", type=Path, help="Optional LoRA adapter for optimized mode.")
    parser.add_argument("--csv", type=Path, help="Optional path to append benchmark results.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    modes = ["baseline", "optimized"] if args.mode == "both" else [args.mode]
    for mode in modes:
        result = run_iterations(mode, args.input_image, args.prompt, args.repetitions, args.lora)
        LOGGER.info(
            "%s | latency %.2fs±%.2f | temp %.3f±%.3f | VRAM %.2f GB",
            mode,
            result.latency_avg,
            result.latency_std,
            result.temporal_avg,
            result.temporal_std,
            result.max_vram_gb,
        )
        if args.csv:
            write_csv(result, args.csv)


if __name__ == "__main__":
    main()

