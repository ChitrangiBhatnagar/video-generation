"""
Quick start
-----------
python -m src.data_prep.download_msr_vtt --output-dir data/raw/msr-vtt

Expected output
---------------
- Downloads annotations and sample video subset from MSR-VTT mirrors or HF hub.
- Logs estimated storage use and creates a manifest JSON.

Description
-----------
Utility for downloading and organizing MSR-VTT data for LoRA fine-tuning.
Supports partial downloads for rapid experimentation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import requests
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)

MSR_VTT_ANNOTATIONS_URL = "https://huggingface.co/datasets/MSR-VTT/resolve/main/MSRVTT_data.json"
MSR_VTT_SAMPLE_VIDEOS = [
    ("video0.mp4", "https://huggingface.co/datasets/MSR-VTT/resolve/main/train-video/video0.mp4"),
    ("video1.mp4", "https://huggingface.co/datasets/MSR-VTT/resolve/main/train-video/video1.mp4"),
    ("video2.mp4", "https://huggingface.co/datasets/MSR-VTT/resolve/main/train-video/video2.mp4"),
]


def download_file(url: str, dest: Path) -> None:
    LOGGER.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with dest.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)


def create_manifest(output_dir: Path, videos: List[Path]) -> None:
    manifest = [{"video_path": str(video)} for video in videos]
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("Wrote manifest to %s", manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MSR-VTT annotations and sample videos.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3, help="Number of videos to download for quick start.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = args.output_dir / "MSRVTT_data.json"
    if not annotations_path.exists():
        download_file(MSR_VTT_ANNOTATIONS_URL, annotations_path)
    else:
        LOGGER.info("Annotations already present at %s", annotations_path)

    downloaded_videos = []
    for filename, url in tqdm(MSR_VTT_SAMPLE_VIDEOS[: args.limit], desc="Downloading MSR-VTT samples"):
        video_path = args.output_dir / "videos" / filename
        if not video_path.exists():
            download_file(url, video_path)
        downloaded_videos.append(video_path)

    create_manifest(args.output_dir, downloaded_videos)


if __name__ == "__main__":
    main()

