from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


LOGGER = logging.getLogger(__name__)


@dataclass
class ClipScorerConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
    batch_size: int = 4


class ClipSimilarityScorer:
    """Lightweight CLIP similarity scorer for video validation."""

    def __init__(self, config: ClipScorerConfig | None = None) -> None:
        cfg = config or ClipScorerConfig()
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(cfg.model_name)
        self.model = CLIPModel.from_pretrained(cfg.model_name).to(device)
        self.device = device
        self.batch_size = cfg.batch_size

    def score(self, images: Iterable[Image.Image], text: str) -> float:
        frames: List[Image.Image] = list(images)
        if not frames:
            raise ValueError("No frames provided for CLIP scoring.")

        scores: List[float] = []
        for idx in range(0, len(frames), self.batch_size):
            batch = frames[idx : idx + self.batch_size]
            inputs = self.processor(
                text=[text] * len(batch),
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image.squeeze().detach().cpu()
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
            scores.extend(logits.tolist())
        return float(sum(scores) / len(scores))


def select_keyframes(frames: torch.Tensor, count: int = 4) -> List[Image.Image]:
    """Select evenly spaced frames (torch tensor BxFxCxHxW) and convert to PIL."""
    if frames.ndim != 5:
        raise ValueError("Frames tensor must have shape (B, F, C, H, W).")
    _, num_frames, _, _, _ = frames.shape
    indices = torch.linspace(0, num_frames - 1, steps=min(count, num_frames)).long()
    pil_frames: List[Image.Image] = []
    for idx in indices:
        frame = frames[0, idx].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255.0).clip(0, 255).astype("uint8")
        pil_frames.append(Image.fromarray(frame))
    return pil_frames

