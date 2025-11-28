"""
Quick start
-----------
accelerate launch src/train_lora.py \\
    --metadata data/metadata/training_metadata.json \\
    --output-dir checkpoints/lora_cogvideox_press \\
    --config configs/training/lora_default.yaml

Expected output
---------------
- Saves epoch checkpoints (`adapter_epoch_{n}`) under the output directory.
- Logs training/loss metrics to console and optional Weights & Biases run.
- Generates sample clips to `outputs/examples/lora_epoch_{n}.mp4`.

Description
-----------
Implements LoRA fine-tuning for the attention processors of the CogVideoX
video UNet, targeting press/publication scenarios while training on prepared
image+prompt metadata. Optimized for small GPU memory footprints with gradient
checkpointing and mixed precision support.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import random
import shutil

import csv
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerator_logger
from diffusers import (
    CogVideoXImageToVideoPipeline,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from peft import LoraConfig
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install peft to run LoRA training.") from exc

try:
    from .evaluation.clip_similarity import ClipScorerConfig, ClipSimilarityScorer, select_keyframes
    from .temporal_consistency import TemporalConsistencyEvaluator
    from .utils.logging_config import configure_logging
    from .utils.pipeline_utils import enable_offload_for_pipeline, save_video_sample
    from .utils.seed_utils import seed_everything
except ImportError:
    # Fallback for when running as script directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.evaluation.clip_similarity import ClipScorerConfig, ClipSimilarityScorer, select_keyframes
    from src.temporal_consistency import TemporalConsistencyEvaluator
    from src.utils.logging_config import configure_logging
    from src.utils.pipeline_utils import enable_offload_for_pipeline, save_video_sample
    from src.utils.seed_utils import seed_everything


LOGGER = logging.getLogger(__name__)


@dataclass
class LoraTrainingConfig:
    model_id: str = "THUDM/CogVideoX-5B-I2V"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    num_train_epochs: int = 3
    max_train_steps: Optional[int] = None
    mixed_precision: str = "bf16"
    seed: int = 42
    fp16: bool = True
    checkpointing_steps: int = 100
    sample_prompts: List[str] = None  # type: ignore[assignment]
    image_column: str = "image_path"
    prompt_column: str = "caption"
    scenario_column: str = "scenario"
    adapter_rank: int = 64
    adapter_alpha: int = 128
    adapter_dropout: float = 0.05
    validation_every_n_epochs: int = 1
    num_video_frames: int = 32
    frame_sampling_strategy: str = "uniform"
    metrics_path: Optional[Path] = None
    metadata_split: str = "train"
    scenario_filter: Optional[List[str]] = None
    use_latent_cache: bool = True
    cfg_dropout: float = 0.0
    validation_prompts_path: Optional[Path] = None
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_device: Optional[str] = None
    clip_batch_size: int = 4
    keyframes_for_clip: int = 4
    guidance_scale_validation: float = 6.0
    keep_top_k_adapters: int = 3
    sample_images: List[Path] = field(default_factory=lambda: [
        Path("data/samples/sample_image_1.png"),
        Path("data/samples/sample_image_2.png"),
    ])

    def __post_init__(self) -> None:
        if self.sample_prompts is None:
            self.sample_prompts = [
                "A bustling newsroom preparing for a breaking announcement",
                "A panel discussion on economic policy in a studio",
            ]
        if isinstance(self.validation_prompts_path, str):
            self.validation_prompts_path = Path(self.validation_prompts_path)
        if self.validation_prompts_path is None:
            default_validation = Path("configs/training/validation_prompts.yaml")
            if default_validation.exists():
                self.validation_prompts_path = default_validation


class MetadataDataset(Dataset):
    """Dataset reading conditioning images, prompts, and associated video clips."""

    def __init__(
        self,
        metadata_path: Path,
        image_column: str,
        prompt_column: str,
        video_column: str,
        num_frames: int,
        split: str = "train",
        scenario_filter: Optional[Iterable[str]] = None,
        use_latent_cache: bool = False,
    ) -> None:
        self.metadata_path = metadata_path
        with metadata_path.open("r", encoding="utf-8") as fh:
            content = json.load(fh)
        if isinstance(content, dict):
            records: List[Dict[str, Any]] = content.get(split, [])
        else:
            if split != "train":
                LOGGER.warning("Metadata file %s is flat; ignoring split '%s'.", metadata_path, split)
            records = content

        if scenario_filter:
            scenario_set = {scenario for scenario in scenario_filter}
            records = [record for record in records if record.get("scenario") in scenario_set]

        if not records:
            raise ValueError(f"No records found for split '{split}' in metadata {metadata_path}.")

        self.records = records
        self.image_column = image_column
        self.prompt_column = prompt_column
        self.video_column = video_column
        self.num_frames = num_frames
        self.use_latent_cache = use_latent_cache

    def __len__(self) -> int:
        return len(self.records)

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        frames = iio.imread(video_path)
        if frames.ndim == 3:
            frames = frames[None, ...]

        if frames.shape[0] >= self.num_frames:
            indices = np.linspace(0, frames.shape[0] - 1, self.num_frames, dtype=int)
            frames = frames[indices]
        else:
            pad = np.repeat(frames[-1][None, ...], self.num_frames - frames.shape[0], axis=0)
            frames = np.concatenate([frames, pad], axis=0)

        tensor = torch.from_numpy(frames).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2)  # (F, C, H, W)
        return tensor

    def _load_from_cache(self, record: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        if not self.use_latent_cache:
            return None
        latent_path = record.get("latent_path")
        if not latent_path:
            return None
        cache_path = Path(latent_path)
        if not cache_path.exists():
            LOGGER.debug("Latent cache missing for %s", cache_path)
            return None
        with np.load(cache_path, allow_pickle=False) as cache:
            data = cache["data"]
            data_format = cache["data_format"].item() if "data_format" in cache else "frames"
            if data_format == "latents":
                tensor = torch.from_numpy(data).float()
                if tensor.ndim == 4 and tensor.shape[0] != self.num_frames:
                    tensor = tensor.permute(1, 0, 2, 3)  # (F, C, H, W)
                return {"latents": tensor}
            tensor = torch.from_numpy(data).float()  # (F, H, W, C)
            if tensor.ndim == 4:
                tensor = tensor.permute(0, 3, 1, 2)
            return {"video": tensor}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        prompt_candidates = record.get("prompt_variants") or [record.get(self.prompt_column, "")]
        prompt = random.choice(prompt_candidates)
        image = Image.open(record[self.image_column]).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

        cache = self._load_from_cache(record)
        if cache is None:
            video_frames = self._load_video_frames(Path(record[self.video_column]))
            cache = {"video": video_frames}

        cache.update({"conditioning_image": image_tensor, "prompt": prompt, "scenario": record.get("scenario", "")})
        return cache


def read_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path:
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_validation_items(config: LoraTrainingConfig) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    if config.validation_prompts_path and config.validation_prompts_path.exists():
        with config.validation_prompts_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        for entry in payload.get("scenarios", []):
            prompt = entry.get("prompt")
            image_path = entry.get("image_path")
            if not prompt or not image_path:
                continue
            items.append((prompt, Path(image_path)))
    if not items:
        items = list(zip(config.sample_prompts, config.sample_images))
    return items


def prune_adapters(output_dir: Path, keep_top_k: int) -> None:
    if keep_top_k <= 0:
        return
    metrics_log = output_dir / "validation_metrics.jsonl"
    if not metrics_log.exists():
        LOGGER.info("No validation metrics log found at %s; skipping adapter pruning.", metrics_log)
        return
    epoch_scores: Dict[int, Dict[str, List[float]]] = {}
    with metrics_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = record.get("epoch")
            if epoch is None:
                continue
            entry = epoch_scores.setdefault(int(epoch), {"temporal": [], "clip": []})
            temporal_score = record.get("temporal_score")
            clip_score = record.get("clip_score")
            if temporal_score is not None:
                entry["temporal"].append(float(temporal_score))
            if clip_score is not None:
                entry["clip"].append(float(clip_score))

    if not epoch_scores:
        LOGGER.info("No epoch metrics available; skipping adapter pruning.")
        return

    aggregates: List[Dict[str, float | int]] = []
    for epoch, values in epoch_scores.items():
        temporal_avg = float(np.mean(values["temporal"])) if values["temporal"] else 0.0
        clip_avg = float(np.mean(values["clip"])) if values["clip"] else 0.0
        combined = temporal_avg + clip_avg
        aggregates.append(
            {
                "epoch": epoch,
                "temporal_avg": temporal_avg,
                "clip_avg": clip_avg,
                "combined": combined,
            }
        )

    aggregates.sort(key=lambda item: item["combined"], reverse=True)
    keep = {entry["epoch"] for entry in aggregates[:keep_top_k]}
    prune_candidates = aggregates[keep_top_k:]
    if not prune_candidates:
        LOGGER.info("All adapters retained (%d <= keep_top_k=%d).", len(aggregates), keep_top_k)
        return

    prune_dir = output_dir / "pruned"
    prune_dir.mkdir(parents=True, exist_ok=True)
    for entry in prune_candidates:
        epoch = int(entry["epoch"])
        adapter_dir = output_dir / f"adapter_epoch_{epoch}"
        if adapter_dir.exists():
            target = prune_dir / adapter_dir.name
            shutil.move(str(adapter_dir), target)
            LOGGER.info(
                "Pruned adapter_epoch_%d (combined %.3f -> moved to %s)",
                epoch,
                entry["combined"],
                target,
            )


def apply_lora_adapters(
    pipeline: CogVideoXImageToVideoPipeline,
    config: LoraTrainingConfig,
) -> List[torch.nn.Parameter]:
    LOGGER.info("Attaching LoRA adapters: rank=%d alpha=%d dropout=%.2f", config.adapter_rank, config.adapter_alpha, config.adapter_dropout)
    lora_config = LoraConfig(
        r=config.adapter_rank,
        lora_alpha=config.adapter_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config.adapter_dropout,
    )
    pipeline.unet.requires_grad_(False)
    pipeline.unet.add_adapter(lora_config, adapter_name="press_lora")
    pipeline.unet.set_attn_processor("press_lora")
    pipeline.unet.enable_gradient_checkpointing()
    if is_xformers_available():
        pipeline.unet.enable_xformers_memory_efficient_attention()
    trainable_params: List[torch.nn.Parameter] = []
    for processor in pipeline.unet.attn_processors.values():
        if hasattr(processor, "parameters"):
            for param in processor.parameters():
                param.requires_grad = True
                trainable_params.append(param)
    LOGGER.info("Trainable LoRA parameters: %d", sum(param.numel() for param in trainable_params))
    return trainable_params


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    videos = [item["video"] for item in batch if "video" in item]
    latents = [item["latents"] for item in batch if "latents" in item]
    conditioning_images = torch.stack([item["conditioning_image"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    scenarios = [item.get("scenario", "") for item in batch]

    if videos:
        result["videos"] = torch.stack(videos)
    if latents:
        result["latents"] = torch.stack(latents)
    result["conditioning_images"] = conditioning_images
    result["prompts"] = prompts
    result["scenarios"] = scenarios
    return result


def train_lora(
    metadata_path: Path,
    output_dir: Path,
    config: LoraTrainingConfig,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    seed_everything(config.seed)
    configure_logging()
    accel_logger = get_accelerator_logger(__name__)
    metrics_path = config.metrics_path
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def record_metric(step: int, loss_value: float) -> None:
        if not metrics_path:
            return
        is_new = not metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if is_new:
                writer.writerow(["step", "loss"])
            writer.writerow([step, f"{loss_value:.6f}"])

    LOGGER.info("Loading dataset metadata from %s", metadata_path)
    dataset = MetadataDataset(
        metadata_path=metadata_path,
        image_column=config.image_column,
        prompt_column=config.prompt_column,
        video_column="video_path",
        num_frames=config.num_video_frames,
        split=config.metadata_split,
        scenario_filter=config.scenario_filter,
        use_latent_cache=config.use_latent_cache,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    LOGGER.info("Loading CogVideoX pipeline: %s", config.model_id)
    try:
        # Load with memory-efficient settings
        pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except MemoryError:
        LOGGER.warning("MemoryError during model loading. Trying with CPU offloading...")
        pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            config.model_id,
            torch_dtype=torch.float32,  # Use FP32 for CPU
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    
    if torch.cuda.is_available():
        pipeline.to(accelerator.device)
    enable_offload_for_pipeline(pipeline)
    lora_params = apply_lora_adapters(pipeline, config)

    validation_items = load_validation_items(config)
    clip_scorer: Optional[ClipSimilarityScorer] = None
    if accelerator.is_main_process and config.clip_model_name:
        clip_config = ClipScorerConfig(
            model_name=config.clip_model_name,
            device=config.clip_device,
            batch_size=config.clip_batch_size,
        )
        clip_scorer = ClipSimilarityScorer(clip_config)

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config.learning_rate,
    )
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps or (len(dataloader) * config.num_train_epochs),
    )

    ema_unet = EMAModel(pipeline.unet.parameters(), model_cls=pipeline.unet.__class__)

    pipeline.unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        pipeline.unet, optimizer, lr_scheduler, dataloader
    )
    trainable_params = [param for param in pipeline.unet.parameters() if param.requires_grad]

    total_steps = config.max_train_steps or (len(dataloader) * config.num_train_epochs)
    global_step = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    def encode_videos(videos: torch.Tensor) -> torch.Tensor:
        videos = videos.to(device=accelerator.device, dtype=pipeline.vae.dtype)
        videos = (videos * 2.0) - 1.0
        videos = videos.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
        with torch.no_grad():
            latents_dist = pipeline.vae.encode(videos)
            latents = latents_dist.latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor
        return latents

    def encode_prompts(prompts: List[str]) -> torch.Tensor:
        text_inputs = pipeline.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = pipeline.text_encoder(
                text_inputs.input_ids.to(accelerator.device)
            )[0]
        return encoder_hidden_states

    def encode_conditioning_images(images: torch.Tensor, target_frames: int) -> torch.Tensor:
        images = images.to(device=accelerator.device, dtype=pipeline.vae.dtype)
        images = (images * 2.0) - 1.0
        images = images.unsqueeze(2)  # (B, C, 1, H, W)
        with torch.no_grad():
            latents_dist = pipeline.vae.encode(images)
            latents = latents_dist.latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor
        if latents.shape[2] != target_frames:
            latents = latents.repeat(1, 1, target_frames, 1, 1)
        return latents

    for epoch in range(config.num_train_epochs):
        pipeline.unet.train()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}")
        for step, batch in progress_bar:
            with accelerator.accumulate(pipeline.unet):
                latents: torch.Tensor
                if "videos" in batch:
                    videos = batch["videos"]
                    latents = encode_videos(videos)
                elif "latents" in batch:
                    latents = batch["latents"].to(device=accelerator.device, dtype=pipeline.vae.dtype)
                    if latents.ndim == 5:
                        latents = latents.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
                else:
                    raise ValueError("Batch is missing both 'videos' and 'latents' tensors.")

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompts = batch["prompts"]
                if config.cfg_dropout > 0.0:
                    mask = torch.rand(len(prompts)) < config.cfg_dropout
                    prompts = [" " if drop else prompt for prompt, drop in zip(prompts, mask)]
                encoder_hidden_states = encode_prompts(prompts)
                conditioning_latents = encode_conditioning_images(batch["conditioning_images"], latents.shape[2]).to(latents.dtype)
                added_cond_kwargs = {"image_latents": conditioning_latents}

                try:
                    model_output = pipeline.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                    )
                except TypeError:
                    model_output = pipeline.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                    )
                model_pred = model_output.sample if hasattr(model_output, "sample") else model_output[0]
                loss = F.mse_loss(model_pred.float(), noise.float())
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            ema_unet.step(pipeline.unet.parameters())
            global_step += 1
            if accelerator.is_main_process:
                progress_bar.set_postfix({"loss": loss.item()})

            if accelerator.is_main_process and global_step % 10 == 0:
                LOGGER.info("Epoch %d | Step %d/%d | Loss %.4f", epoch + 1, global_step, total_steps, loss.item())
                accel_logger.log({"train/loss": loss.item(), "train/step": global_step})
                record_metric(global_step, loss.item())

            if accelerator.is_main_process and config.checkpointing_steps and global_step % config.checkpointing_steps == 0:
                save_path = output_dir / f"adapter_step_{global_step}"
                accelerator.wait_for_everyone()
                pipeline.unet.save_attn_procs(save_path)
                LOGGER.info("Saved LoRA attn procs to %s", save_path)

            if config.max_train_steps and global_step >= config.max_train_steps:
                break

        if accelerator.is_main_process and (epoch + 1) % config.validation_every_n_epochs == 0:
            validate_and_sample(
                pipeline,
                output_dir,
                epoch + 1,
                config,
                validation_items,
                clip_scorer=clip_scorer,
            )

        if accelerator.is_main_process:
            epoch_path = output_dir / f"adapter_epoch_{epoch + 1}"
            pipeline.unet.save_attn_procs(epoch_path)
            LOGGER.info("Saved epoch checkpoint to %s", epoch_path)

        if config.max_train_steps and global_step >= config.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = output_dir / "adapter_final"
        pipeline.unet.save_attn_procs(final_path)
        LOGGER.info("Training complete. Final adapter saved to %s", final_path)
        prune_adapters(output_dir, config.keep_top_k_adapters)


def validate_and_sample(
    pipeline: CogVideoXImageToVideoPipeline,
    output_dir: Path,
    epoch: int,
    config: LoraTrainingConfig,
    validation_items: List[Tuple[str, Path]],
    clip_scorer: Optional[ClipSimilarityScorer] = None,
) -> None:
    pipeline.unet.eval()
    evaluator = TemporalConsistencyEvaluator()
    device = pipeline.device if isinstance(pipeline.device, torch.device) else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if config.fp16 else torch.bfloat16
    examples_dir = Path("outputs/examples")
    examples_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = output_dir / "validation_metrics.jsonl"

    for idx, (prompt, image_path) in enumerate(validation_items):
        output_path = output_dir / f"sample_epoch_{epoch}_{idx}.mp4"
        conditioning_image = Image.open(image_path).convert("RGB")
        with torch.autocast(device_type=device.type, dtype=dtype):
            result = pipeline(
                prompt=prompt,
                image=conditioning_image,
                num_frames=config.num_video_frames,
                guidance_scale=config.guidance_scale_validation,
                num_inference_steps=30,
            )
        frames = result.frames[0] if hasattr(result, "frames") else result.videos[0]  # type: ignore[attr-defined]
        save_video_sample(frames, output_path, fps=8)
        example_copy = examples_dir / f"lora_epoch_{epoch}_{idx}.mp4"
        if example_copy != output_path:
            example_copy.write_bytes(output_path.read_bytes())
        score = evaluator.score_tensor_video(frames)
        clip_score: Optional[float] = None
        if clip_scorer:
            try:
                frame_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
                keyframes = select_keyframes(frame_tensor.unsqueeze(0), config.keyframes_for_clip)
                clip_score = clip_scorer.score(keyframes, prompt)
            except Exception as exc:  # pragma: no cover - best effort metric
                LOGGER.warning("CLIP scoring failed for prompt '%s': %s", prompt, exc)
        LOGGER.info(
            "Validation prompt '%s' | temporal=%.3f | clip=%s | sample=%s",
            prompt,
            score,
            f"{clip_score:.3f}" if clip_score is not None else "n/a",
            output_path,
        )
        record = {
            "epoch": epoch,
            "prompt": prompt,
            "output_path": str(output_path),
            "reference_image": str(image_path),
            "temporal_score": float(score),
            "clip_score": float(clip_score) if clip_score is not None else None,
        }
        with metrics_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapters for CogVideoX.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to training metadata JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to store LoRA checkpoints.")
    parser.add_argument("--config", type=Path, help="YAML config overriding defaults.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_kwargs = read_config(args.config)
    config = LoraTrainingConfig(**config_kwargs)
    train_lora(
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        config=config,
        config_overrides=config_kwargs,
    )


if __name__ == "__main__":
    main()

