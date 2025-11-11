"""
Quick start
-----------
python -m src.batch_processor \\
    --jobs-file data/metadata/batch_jobs.json \\
    --mode optimized \\
    --output-dir outputs/batch_runs

Expected output
---------------
- Iterates over job definitions and generates videos with progress reporting.
- Writes a run summary JSON capturing status, latency, and consistency scores.

Description
-----------
Batch generation utility supporting baseline or optimized inference modes with
robust error handling. Designed for processing curated JSON job lists for
press/publication review workflows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from .baseline_inference import BaselineInferenceConfig, generate_video as baseline_generate, load_pipeline as load_baseline
from .optimized_inference import (
    OptimizedInferenceConfig,
    benchmark_generation as optimized_generate,
    load_pipeline as load_optimized,
)
from .temporal_consistency import TemporalConsistencyEvaluator
from .utils.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)


@dataclass
class BatchJob:
    image: Path
    prompt: str
    output_path: Path
    lora: Optional[Path] = None


@dataclass
class BatchProcessorConfig:
    mode: str = "baseline"
    output_dir: Path = field(default_factory=lambda: Path("outputs/batch_runs"))
    record_metrics: bool = True

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def parse_jobs(jobs_file: Path, default_output_dir: Path) -> List[BatchJob]:
    with jobs_file.open("r", encoding="utf-8") as fh:
        jobs_data = json.load(fh)
    jobs: List[BatchJob] = []
    for entry in jobs_data:
        output_path = default_output_dir / entry.get("output_name", f"{len(jobs)}.mp4")
        jobs.append(
            BatchJob(
                image=Path(entry["image"]),
                prompt=entry["caption"],
                output_path=entry.get("output_path") and Path(entry["output_path"]) or output_path,
                lora=entry.get("lora_path") and Path(entry["lora_path"]),
            )
        )
    return jobs


def run_batch(jobs: List[BatchJob], config: BatchProcessorConfig) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    evaluator = TemporalConsistencyEvaluator()

    if config.mode == "baseline":
        inference_config = BaselineInferenceConfig()
        pipeline = load_baseline(inference_config)
        generator = lambda job: baseline_generate(  # noqa: E731
            pipeline,
            inference_config,
            job.image,
            job.prompt,
            job.output_path,
        )
    else:
        inference_config = OptimizedInferenceConfig()
        pipeline = load_optimized(inference_config)
        generator = lambda job: optimized_generate(  # noqa: E731
            pipeline,
            inference_config,
            job.image,
            job.prompt,
            job.lora,
            job.output_path,
        )

    for job in tqdm(jobs, desc=f"Processing jobs ({config.mode})"):
        start = time.time()
        try:
            video_path = generator(job)
            latency = time.time() - start
            score = evaluator.score_video_file(video_path)
            metrics[str(job.output_path)] = {"latency_sec": latency, "temporal_consistency": score}
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Job failed for %s: %s", job.output_path, exc)
            metrics[str(job.output_path)] = {"error": str(exc)}
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch process image+prompt jobs into videos.")
    parser.add_argument("--jobs-file", type=Path, required=True, help="JSON list of {image, caption, [lora_path]} items.")
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch_runs"))
    parser.add_argument("--metrics-path", type=Path, help="Optional path to write per-job metrics.")
    return parser.parse_args()


def main() -> None:
    configure_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args()
    config = BatchProcessorConfig(mode=args.mode, output_dir=args.output_dir)
    jobs = parse_jobs(args.jobs_file, config.output_dir)
    metrics = run_batch(jobs, config)
    if args.metrics_path:
        with args.metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        LOGGER.info("Wrote metrics to %s", args.metrics_path)


if __name__ == "__main__":
    main()

