"""
Quick start
-----------
python -m src.optimized_inference \\
    --input-image data/samples/sample_image_1.png \\
    --prompt "An anchor delivering breaking news about market updates" \\
    --lora checkpoints/lora_cogvideox_press/adapter_final \\
    --output-path outputs/examples/optimized_markets.mp4

Expected output
---------------
- Generates a 6-second MP4 using optimized settings with latency and VRAM stats.
- Reports temporal consistency and writes benchmark metrics to stdout.

Description
-----------
Optimized inference runner featuring selective CPU/GPU offload, VAE tiling,
fast schedulers, and optional LoRA adapters for domain-specialized generation.
Use to evaluate latency gains versus the baseline pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

try:
    from diffusers import (
        CogVideoXImageToVideoPipeline,
        DPMSolverMultistepScheduler,
    )
    from diffusers.utils import export_to_video, load_image
except Exception as exc:  # pragma: no cover
    CogVideoXImageToVideoPipeline = None  # type: ignore
    DPMSolverMultistepScheduler = None  # type: ignore
    export_to_video = None  # type: ignore
    load_image = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .temporal_consistency import TemporalConsistencyEvaluator
from .utils.logging_config import configure_logging
from .utils.pipeline_utils import (
    enable_offload_for_pipeline,
    load_lora_into_pipeline,
    set_lora_scale,
)
from .utils.seed_utils import seed_everything


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/model/optimized_inference.yaml")


@dataclass
class OptimizedInferenceConfig:
    model_id: str = "THUDM/CogVideoX-5B-I2V"
    num_frames: int = 49
    fps: int = 8
    height: int = 480
    width: int = 720
    guidance_scale: float = 6.0
    num_inference_steps: int = 28
    seed: Optional[int] = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_flash_attention: bool = True
    vae_slicing: bool = True
    vae_tiling: bool = True
    use_fp16: bool = True
    offload: bool = True
    output_dir: Path = field(default_factory=lambda: Path("outputs/examples"))
    lora_scale: float = 1.0
    lora_presets: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_config_from_file(config_path: Optional[Path]) -> Dict[str, object]:
    if not config_path:
        config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        LOGGER.warning("Config file %s not found; using defaults.", config_path)
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def build_config(overrides: Optional[Dict[str, object]] = None) -> OptimizedInferenceConfig:
    overrides = overrides or {}
    kwargs = dict(overrides)
    if "output_dir" in kwargs:
        kwargs["output_dir"] = Path(kwargs["output_dir"])
    if "lora_presets" in kwargs and kwargs["lora_presets"] is None:
        kwargs["lora_presets"] = {}
    return OptimizedInferenceConfig(**kwargs)


def load_pipeline(config: OptimizedInferenceConfig) -> CogVideoXImageToVideoPipeline:
    if CogVideoXImageToVideoPipeline is None or load_image is None:
        raise ImportError("diffusers not available") from _IMPORT_ERROR

    torch_dtype = torch.float16 if config.use_fp16 and config.device != "cpu" else torch.float32
    pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
        config.model_id, torch_dtype=torch_dtype
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    if config.device == "cuda":
        pipeline.to("cuda")
    if config.offload:
        enable_offload_for_pipeline(pipeline)
    if config.vae_slicing:
        pipeline.vae.enable_slicing()
    if config.vae_tiling:
        pipeline.vae.enable_tiling()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def benchmark_generation(
    pipeline: CogVideoXImageToVideoPipeline,
    config: OptimizedInferenceConfig,
    input_image: Path,
    prompt: str,
    lora_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    lora_scale: Optional[float] = None,
) -> Path:
    seed_everything(config.seed)
    lora_scale = lora_scale if lora_scale is not None else config.lora_scale
    if lora_path:
        load_lora_into_pipeline(pipeline, lora_path, scale=lora_scale)
        LOGGER.info("Applied LoRA adapter %s with scale %.2f", lora_path, lora_scale)
    else:
        set_lora_scale(pipeline, lora_scale)
        LOGGER.info("Running with existing LoRA scale %.2f (path unchanged).", lora_scale)

    conditioning_image = load_image(str(input_image))
    output_path = output_path or config.output_dir / "optimized_output.mp4"

    start_time = time.time()
    result = pipeline(
        prompt=prompt,
        image=conditioning_image,
        num_frames=config.num_frames,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        output_type="pt",
    )
    total_time = time.time() - start_time
    frames = result.frames[0] if hasattr(result, "frames") else result.videos[0]  # type: ignore[attr-defined]
    export_to_video(frames, str(output_path), fps=config.fps)

    vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    throughput = 3600.0 / total_time if total_time > 0 else float("inf")
    LOGGER.info(
        "Optimized inference completed in %.2fs (VRAM %.2f GB, throughput %.1f videos/hour)",
        total_time,
        vram,
        throughput,
    )
    evaluator = TemporalConsistencyEvaluator(fps=config.fps)
    score = evaluator.score_tensor_video(frames)
    LOGGER.info("Temporal consistency score: %.3f", score)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized CogVideoX inference runner.")
    parser.add_argument("--config", type=Path, help="Optional YAML config path.")
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--scenario", type=str, help="Scenario preset key defined in config lora_presets.")
    parser.add_argument("--lora", type=Path, help="Override LoRA adapter directory.")
    parser.add_argument("--disable-lora", action="store_true", help="Force disable LoRA even if scenario preset exists.")
    parser.add_argument("--lora-scale", type=float, help="Override LoRA scaling factor.")
    parser.add_argument("--output-path", type=Path, help="Target video output path.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenario presets and exit.")
    return parser.parse_args()


def resolve_lora_path(
    config: OptimizedInferenceConfig,
    scenario: Optional[str],
    explicit_path: Optional[Path],
    disable_lora: bool,
) -> Optional[Path]:
    if disable_lora:
        return None
    if explicit_path:
        return explicit_path
    if scenario:
        preset = config.lora_presets.get(scenario)
        if not preset:
            raise ValueError(f"Scenario '{scenario}' not found in config lora_presets.")
        return Path(preset)
    return None


def main() -> None:
    configure_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args()
    config_overrides = load_config_from_file(args.config)
    config = build_config(config_overrides)

    if args.list_scenarios:
        if config.lora_presets:
            for key, value in sorted(config.lora_presets.items()):
                print(f"{key}: {value}")
        else:
            print("No scenario presets defined.")
        return

    if args.num_inference_steps:
        config.num_inference_steps = args.num_inference_steps
    if args.guidance_scale:
        config.guidance_scale = args.guidance_scale
    if args.num_frames:
        config.num_frames = args.num_frames
    if args.seed is not None:
        config.seed = args.seed
    if args.lora_scale is not None:
        config.lora_scale = args.lora_scale

    lora_path = resolve_lora_path(
        config=config,
        scenario=args.scenario,
        explicit_path=args.lora,
        disable_lora=args.disable_lora,
    )

    pipeline = load_pipeline(config)
    benchmark_generation(
        pipeline=pipeline,
        config=config,
        input_image=args.input_image,
        prompt=args.prompt,
        lora_path=lora_path,
        output_path=args.output_path,
        lora_scale=args.lora_scale,
    )


if __name__ == "__main__":
    main()

