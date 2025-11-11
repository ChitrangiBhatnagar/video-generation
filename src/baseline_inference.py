"""
Quick start
-----------
python -m src.baseline_inference \\
    --input-image data/samples/sample_image_1.png \\
    --prompt "A press conference in a modern newsroom" \\
    --output-path outputs/examples/baseline_press_conference.mp4

Expected output
---------------
- Writes 49-frame (≈6 s @ 8 fps) MP4 to the specified path.
- Prints timing, VRAM estimate, and temporal consistency placeholder metrics.

Description
-----------
Baseline inference entry-point for generating 6-second 720×480 video clips from
an input image and scenario description using CogVideoX-5B-I2V with image
conditioning. Designed to be the default flow before LoRA fine-tuning or
runtime optimizations are applied.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

try:
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
except Exception as exc:  # pragma: no cover - dependency guard
    CogVideoXImageToVideoPipeline = None  # type: ignore
    export_to_video = None  # type: ignore
    load_image = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .temporal_consistency import TemporalConsistencyEvaluator
from .utils.logging_config import configure_logging
from .utils.pipeline_utils import enable_offload_for_pipeline
from .utils.seed_utils import seed_everything


LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineInferenceConfig:
    """Runtime configuration for baseline inference."""

    model_id: str = "THUDM/CogVideoX-5B-I2V"
    num_frames: int = 49
    fps: int = 8
    height: int = 480
    width: int = 720
    guidance_scale: float = 6.5
    num_inference_steps: int = 40
    seed: Optional[int] = 42
    use_fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    offload: bool = True
    output_dir: Path = field(default_factory=lambda: Path("outputs/examples"))

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_pipeline(config: BaselineInferenceConfig) -> CogVideoXImageToVideoPipeline:
    """Load CogVideoX pipeline with safe defaults."""

    if CogVideoXImageToVideoPipeline is None or load_image is None:
        raise ImportError(
            "diffusers not available; install per requirements.txt"
        ) from _IMPORT_ERROR

    torch_dtype = torch.float16 if config.use_fp16 and config.device != "cpu" else torch.float32
    LOGGER.info("Loading pipeline %s (dtype=%s)", config.model_id, torch_dtype)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
    )
    if config.device == "cuda":
        pipe.to("cuda")
    if config.offload:
        enable_offload_for_pipeline(pipe)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_video(
    pipeline: CogVideoXImageToVideoPipeline,
    config: BaselineInferenceConfig,
    input_image: Path,
    prompt: str,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a video clip and persist it to disk."""

    seed_everything(config.seed)
    output_path = output_path or (config.output_dir / "baseline_output.mp4")
    LOGGER.info("Loading conditioning image from %s", input_image)
    init_image = load_image(str(input_image))

    LOGGER.info(
        "Generating video with steps=%d guidance=%.2f frames=%d",
        config.num_inference_steps,
        config.guidance_scale,
        config.num_frames,
    )
    start_time = time.time()
    result = pipeline(
        prompt=prompt,
        image=init_image,
        num_frames=config.num_frames,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        output_type="pt",
    )
    inference_seconds = time.time() - start_time
    frames = result.frames[0] if hasattr(result, "frames") else result.videos[0]  # type: ignore[attr-defined]

    LOGGER.info("Exporting video to %s", output_path)
    export_to_video(frames, str(output_path), fps=config.fps)

    vram_usage = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    LOGGER.info("Baseline generation finished in %.2fs (max VRAM %.2f GB)", inference_seconds, vram_usage)

    evaluator = TemporalConsistencyEvaluator(fps=config.fps)
    score = evaluator.score_tensor_video(frames)
    LOGGER.info("Temporal consistency score: %.3f", score)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline CogVideoX inference runner.")
    parser.add_argument("--input-image", type=Path, required=True, help="Path to the conditioning image.")
    parser.add_argument("--prompt", type=str, required=True, help="Scenario description/text prompt.")
    parser.add_argument("--output-path", type=Path, help="Where to save the resulting MP4.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Override default denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override guidance scale.")
    parser.add_argument("--num-frames", type=int, default=None, help="Override number of frames.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    configure_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args()
    config = BaselineInferenceConfig()
    if args.num_inference_steps:
        config.num_inference_steps = args.num_inference_steps
    if args.guidance_scale:
        config.guidance_scale = args.guidance_scale
    if args.num_frames:
        config.num_frames = args.num_frames
    if args.seed is not None:
        config.seed = args.seed

    pipeline = load_pipeline(config)
    generate_video(
        pipeline=pipeline,
        config=config,
        input_image=args.input_image,
        prompt=args.prompt,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

