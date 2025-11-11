"""
Quick start
-----------
python -m src.data_prep.prepare_metadata \\
    --msr-vtt data/raw/msr-vtt \\
    --output data/metadata/training_metadata.json \\
    --limit 1000

Expected output
---------------
- Creates `training_metadata.json` with records {image_path, caption, scenario}.
- Extracts first frames for each video sample when frame cache is missing.

Description
-----------
Converts MSR-VTT and optional VATEX/Panda datasets into a curated metadata file
suited for LoRA fine-tuning. Limits sample count to keep training lightweight.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import imageio.v3 as iio
from PIL import Image


LOGGER = logging.getLogger(__name__)


def extract_first_frame(video_path: Path, frame_dir: Path) -> Path:
    frame_dir.mkdir(parents=True, exist_ok=True)
    target = frame_dir / f"{video_path.stem}.png"
    if target.exists():
        return target
    LOGGER.info("Extracting first frame from %s", video_path)
    frame = iio.imread(video_path, index=0)
    Image.fromarray(frame).save(target)
    return target


def build_records_from_msr_vtt(root: Path, limit: int) -> List[Dict[str, str]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MSR-VTT manifest missing at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        videos = json.load(fh)

    records: List[Dict[str, str]] = []
    captions_map_path = root / "MSRVTT_data.json"
    with captions_map_path.open("r", encoding="utf-8") as fh:
        captions_data = json.load(fh)["annotations"]
    caption_lookup = {entry["video_id"]: entry["caption"] for entry in captions_data}

    for entry in videos[:limit]:
        video_path = Path(entry["video_path"])
        frame_path = extract_first_frame(video_path, root / "frames")
        video_id = video_path.stem
        caption = caption_lookup.get(video_id, "A descriptive prompt for the video.")
        records.append(
            {
                "video_path": str(video_path),
                "image_path": str(frame_path),
                "caption": caption,
                "scenario": "press_conference",
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare training metadata JSON.")
    parser.add_argument("--msr-vtt", type=Path, required=True, help="MSR-VTT download root.")
    parser.add_argument("--vatex", type=Path, help="Optional VATEX root for evaluation metadata.")
    parser.add_argument("--panda70m", type=Path, help="Optional Panda-70M subset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output metadata JSON path.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of samples.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    records = build_records_from_msr_vtt(args.msr_vtt, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    LOGGER.info("Wrote %d records to %s", len(records), args.output)


if __name__ == "__main__":
    main()

