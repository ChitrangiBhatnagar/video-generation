"""
Quick start
-----------
python -m src.data_prep.download_vatex --output-dir data/raw/vatex --split validation --limit 5

Expected output
---------------
- Downloads VATEX captions and optional sample videos.
- Stores manifest and storage estimates for evaluation.

Description
-----------
Utility for acquiring VATEX data subsets used in evaluation and multilingual
assessment. Designed to limit downloads for resource-constrained experiments.
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

VATEX_CAPTIONS_URL = "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json"
VATEX_VIDEO_URL_TEMPLATE = "https://storage.googleapis.com/vatex-public/{split}/{video_id}.mp4"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s -> %s", url, dest)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with dest.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VATEX captions and optional video subset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--limit", type=int, default=10, help="Number of videos to fetch.")
    parser.add_argument("--download-videos", action="store_true", help="Also fetch video assets.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    captions_path = args.output_dir / f"vatex_{args.split}_captions.json"
    if not captions_path.exists():
        download(VATEX_CAPTIONS_URL, captions_path)
    else:
        LOGGER.info("Captions already downloaded.")

    if not args.download_videos:
        return

    with captions_path.open("r", encoding="utf-8") as fh:
        captions = json.load(fh)
    videos = []
    for entry in tqdm(captions[: args.limit], desc="Downloading VATEX videos"):
        video_id = entry["videoID"]
        video_path = args.output_dir / "videos" / f"{video_id}.mp4"
        if not video_path.exists():
            url = VATEX_VIDEO_URL_TEMPLATE.format(split=args.split, video_id=video_id)
            download(url, video_path)
        videos.append(video_path)

    manifest = [{"video_path": str(video)} for video in videos]
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("Wrote VATEX manifest to %s", manifest_path)


if __name__ == "__main__":
    main()

