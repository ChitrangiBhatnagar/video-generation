"""
Quick start
-----------
python -m src.data_prep.preprocess_video_latents \
    --metadata data/metadata/scenario_metadata.json \
    --split train \
    --output-dir data/prepared/latents \
    --fps 8 \
    --size 720x480 \
    --augment

Expected output
---------------
- Writes `.npz` latent/clip files under `data/prepared/latents/<scenario>/`.
- Ensures every clip is resampled to 8 fps at 720x480 with normalized lighting.
- Optionally encodes CogVideoX VAE latents for faster training.

Description
-----------
Preprocesses curated newsroom videos into a consistent cache for LoRA
fine-tuning. The pipeline standardizes temporal sampling, frame resolution,
and brightness, while offering lightweight augmentations (pans/zooms) that
preserve scene identity. Encoded caches feed directly into the training loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    fps: int = 8
    width: int = 720
    height: int = 480
    augment: bool = False
    encode_latents: bool = False
    model_id: str = "THUDM/CogVideoX-5B-I2V"
    dtype: str = "bf16"


def load_metadata(metadata_path: Path, split: str) -> List[Dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        content = json.load(fh)
    if isinstance(content, dict):
        if split not in content:
            raise KeyError(f"Split '{split}' not found in metadata file {metadata_path}.")
        return content[split]
    if split != "train":
        raise ValueError("Flat metadata files only support the 'train' split.")
    return content


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float32) / 255.0
    mean = frames.mean(axis=(0, 1, 2), keepdims=True)
    std = frames.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    normalized = (frames - mean) / std
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-6)
    return np.clip(normalized, 0.0, 1.0)


def random_pan_zoom(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(frame.astype(np.uint8))
    tw, th = target_size
    scale = random.uniform(1.0, 1.08)
    new_w = int(pil.width * scale)
    new_h = int(pil.height * scale)
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    max_x = max(0, new_w - tw)
    max_y = max(0, new_h - th)
    offset_x = random.randint(0, max(0, max_x))
    offset_y = random.randint(0, max(0, max_y))
    pil = pil.crop((offset_x, offset_y, offset_x + tw, offset_y + th))
    return np.array(pil)


def resample_frames(frames: np.ndarray, target_count: int) -> np.ndarray:
    if frames.shape[0] == target_count:
        return frames
    indices = np.linspace(0, frames.shape[0] - 1, target_count, dtype=int)
    return frames[indices]


def resize_frames(frames: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    resized = []
    for frame in frames:
        pil = Image.fromarray(frame).resize((width, height), Image.BICUBIC)
        resized.append(np.array(pil))
    return np.stack(resized, axis=0)


def maybe_encode_latents(frames: np.ndarray, config: PreprocessConfig, device: Optional[str]) -> np.ndarray:
    if not config.encode_latents:
        return frames

    import torch
    from diffusers import CogVideoXImageToVideoPipeline

    dtype = torch.float16 if config.dtype == "fp16" else torch.bfloat16
    pipeline = CogVideoXImageToVideoPipeline.from_pretrained(config.model_id, torch_dtype=dtype)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
    tensor = (tensor * 2.0) - 1.0
    with torch.no_grad():
        latents_dist = pipeline.vae.encode(tensor)
        latents = latents_dist.latent_dist.sample()
        latents = latents * pipeline.vae.config.scaling_factor
    return latents.squeeze(0).cpu().numpy()


def process_record(
    record: Dict[str, str],
    output_dir: Path,
    config: PreprocessConfig,
    device: Optional[str] = None,
) -> Path:
    video_path = Path(record["video_path"])
    scenario = record.get("scenario", "unspecified")
    target_dir = output_dir / scenario
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{video_path.stem}.npz"
    if target_path.exists():
        LOGGER.debug("Skipping %s (already cached)", target_path)
        return target_path

    LOGGER.info("Processing %s -> %s", video_path, target_path)
    reader = iio.imiter(video_path)
    frames: List[np.ndarray] = [frame for frame in reader]
    if not frames:
        raise ValueError(f"No frames decoded for video {video_path}")

    source_fps = int(math.ceil(iio.immeta(video_path).get("fps", config.fps)))
    target_frame_count = max(1, int((len(frames) / source_fps) * config.fps))

    frames = np.stack(frames, axis=0)
    frames = resample_frames(frames, target_frame_count)
    frames = resize_frames(frames, (config.width, config.height))

    if config.augment:
        augmented = []
        for frame in frames:
            augmented.append(random_pan_zoom(frame, (config.width, config.height)))
        frames = np.stack(augmented, axis=0)

    frames = normalize_frames(frames)
    data = maybe_encode_latents(frames, config, device)

    np.savez_compressed(
        target_path,
        data=data,
        fps=config.fps,
        width=config.width,
        height=config.height,
        normalized=np.array([not config.encode_latents], dtype=np.bool_),
        scenario=np.array([scenario]),
        data_format=np.array(["latents" if config.encode_latents else "frames"]),
    )
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess video clips into scenario-aligned latent caches.")
    parser.add_argument("--metadata", type=Path, required=True, help="Scenario metadata JSON file.")
    parser.add_argument("--split", type=str, default="train", help="Which split to preprocess (train/validation).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for cached latents or frames.")
    parser.add_argument("--fps", type=int, default=8, help="Target frames per second.")
    parser.add_argument("--size", type=str, default="720x480", help="Target WxH resolution.")
    parser.add_argument("--augment", action="store_true", help="Apply subtle pan/zoom augmentations.")
    parser.add_argument("--encode-latents", action="store_true", help="Encode CogVideoX VAE latents instead of RGB.")
    parser.add_argument("--model-id", type=str, default="THUDM/CogVideoX-5B-I2V", help="Model used for latent encoding.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Precision for latent encoding.")
    parser.add_argument("--device", type=str, help="Torch device when encoding latents (default: auto).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for augmentations.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    random.seed(args.seed)

    try:
        width_str, height_str = args.size.lower().split("x")
        width, height = int(width_str), int(height_str)
    except ValueError as exc:
        raise ValueError(f"Invalid size '{args.size}'. Expected format 'WIDTHxHEIGHT'.") from exc

    metadata = load_metadata(args.metadata, args.split)
    config = PreprocessConfig(
        fps=args.fps,
        width=width,
        height=height,
        augment=args.augment,
        encode_latents=args.encode_latents,
        model_id=args.model_id,
        dtype=args.dtype,
    )

    for record in metadata:
        try:
            cache_path = process_record(
                record,
                output_dir=args.output_dir,
                config=config,
                device=args.device,
            )
            record["latent_path"] = str(cache_path.resolve())
        except Exception as exc:  # pragma: no cover - logging informational
            LOGGER.warning("Failed to process %s: %s", record.get("video_path"), exc)

    # Optionally persist updated metadata with latent paths resolved.
    metadata_path = args.output_dir / f"{args.split}_metadata_with_latents.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    LOGGER.info("Wrote updated metadata (split=%s) with latent paths to %s", args.split, metadata_path)


if __name__ == "__main__":
    main()

