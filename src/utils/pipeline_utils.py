"""Helpers for managing Diffusers pipelines and related utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

try:
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video
except ImportError:  # pragma: no cover
    CogVideoXImageToVideoPipeline = Any  # type: ignore
    export_to_video = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def enable_offload_for_pipeline(pipeline: CogVideoXImageToVideoPipeline) -> None:
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        LOGGER.info("Enabled model CPU offload.")
    elif hasattr(pipeline, "enable_sequential_cpu_offload"):
        pipeline.enable_sequential_cpu_offload()
        LOGGER.info("Enabled sequential CPU offload.")


def load_lora_into_pipeline(pipeline: CogVideoXImageToVideoPipeline, lora_path: Path) -> None:
    LOGGER.info("Loading LoRA adapter from %s", lora_path)
    pipeline.unet.load_attn_procs(lora_path)


def save_video_sample(frames: torch.Tensor, output_path: Path, fps: int) -> None:
    if export_to_video is None:
        raise RuntimeError("diffusers export_to_video unavailable.")
    export_to_video(frames, str(output_path), fps=fps)
    LOGGER.info("Saved video sample to %s", output_path)

