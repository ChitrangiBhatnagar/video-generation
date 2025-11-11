"""
Quick start
-----------
python -m src.validation.validate_lora \\
    --lora checkpoints/lora_cogvideox_press/adapter_final \\
    --scenarios configs/training/validation_prompts.yaml

Expected output
---------------
- Generates comparison videos for baseline vs LoRA-enhanced pipeline.
- Prints temporal consistency deltas and qualitative remarks to stdout.

Description
-----------
Validation harness evaluating LoRA adapters against a fixed scenario set for
alignment and stability improvements.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from ..baseline_inference import BaselineInferenceConfig, generate_video as baseline_generate, load_pipeline as load_baseline
from ..optimized_inference import (
    OptimizedInferenceConfig,
    benchmark_generation as optimized_generate,
    load_pipeline as load_optimized,
)
from ..temporal_consistency import TemporalConsistencyEvaluator
from ..utils.logging_config import configure_logging
from ..utils.pipeline_utils import load_lora_into_pipeline


LOGGER = logging.getLogger(__name__)


def load_scenarios(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["scenarios"]


def validate_lora(lora_path: Path, scenarios: List[Dict[str, str]]) -> None:
    baseline_config = BaselineInferenceConfig()
    optimized_config = OptimizedInferenceConfig()
    baseline_pipeline = load_baseline(baseline_config)
    optimized_pipeline = load_optimized(optimized_config)
    load_lora_into_pipeline(optimized_pipeline, lora_path)

    evaluator = TemporalConsistencyEvaluator()

    for scenario in scenarios:
        image_path = Path(scenario["image_path"])
        prompt = scenario["prompt"]
        name = scenario["name"]
        LOGGER.info("Validating scenario: %s", name)

        baseline_video = baseline_generate(
            pipeline=baseline_pipeline,
            config=baseline_config,
            input_image=image_path,
            prompt=prompt,
            output_path=baseline_config.output_dir / f"{name}_baseline.mp4",
        )
        optimized_video = optimized_generate(
            pipeline=optimized_pipeline,
            config=optimized_config,
            input_image=image_path,
            prompt=prompt,
            lora_path=None,
            output_path=optimized_config.output_dir / f"{name}_lora.mp4",
        )

        baseline_score = evaluator.score_video_file(baseline_video)
        optimized_score = evaluator.score_video_file(optimized_video)
        LOGGER.info(
            "%s | baseline=%.3f | lora=%.3f | delta=%.3f",
            name,
            baseline_score,
            optimized_score,
            optimized_score - baseline_score,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LoRA adapter quality.")
    parser.add_argument("--lora", type=Path, required=True, help="Path to LoRA adapter directory.")
    parser.add_argument("--scenarios", type=Path, required=True, help="YAML file listing validation scenarios.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)
    validate_lora(args.lora, scenarios)


if __name__ == "__main__":
    main()

