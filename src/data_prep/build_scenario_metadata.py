"""
Quick start
-----------
python -m src.data_prep.build_scenario_metadata \
    --msr-vtt data/raw/msr-vtt \
    --scenarios configs/data/scenarios.yaml \
    --output data/metadata/scenario_metadata.json \
    --limit 1500 \
    --validation-ratio 0.1

Expected output
---------------
- Writes `scenario_metadata.json` containing `train` and `validation` splits.
- Extracts consistent 720x480 first-frame thumbnails for every record.
- Populates prompt variants and latent cache paths for downstream processing.

Description
-----------
Curates newsroom scenarios from MSR-VTT (and optional datasets) into a
structured metadata artifact suitable for scenario-faithful LoRA training.
It enforces keyword-based alignment between captions and scenarios, ensures
conditioning images match target resolution, and yields validation splits for
evaluation loops.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v3 as iio
from PIL import Image


LOGGER = logging.getLogger(__name__)
DEFAULT_RESOLUTION: Tuple[int, int] = (720, 480)


@dataclass
class ScenarioSpec:
    """Defines curation constraints for a newsroom scenario."""

    keywords: List[str]
    prompt_templates: List[str]
    min_examples: int = 0


def _load_yaml(path: Path) -> Dict[str, Dict[str, Iterable[str] | int]]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pyyaml to load scenario configs.") from exc

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_scenarios(config_path: Path) -> Dict[str, ScenarioSpec]:
    if config_path.suffix == ".json":
        with config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raw = _load_yaml(config_path)

    scenarios: Dict[str, ScenarioSpec] = {}
    for name, spec in raw.items():
        keywords = spec.get("keywords")
        prompt_templates = spec.get("prompt_templates")
        if not keywords:
            raise ValueError(f"Scenario '{name}' missing keywords.")
        if not prompt_templates:
            raise ValueError(f"Scenario '{name}' missing prompt_templates.")
        scenarios[name] = ScenarioSpec(
            keywords=[kw.lower() for kw in keywords],
            prompt_templates=list(prompt_templates),
            min_examples=int(spec.get("min_examples", 0)),
        )
    return scenarios


def extract_first_frame(video_path: Path, frame_dir: Path, resolution: Tuple[int, int]) -> Path:
    frame_dir.mkdir(parents=True, exist_ok=True)
    target = frame_dir / f"{video_path.stem}.png"
    if target.exists():
        return target
    LOGGER.info("Extracting first frame from %s", video_path)
    frame = iio.imread(video_path, index=0)
    image = Image.fromarray(frame).convert("RGB").resize(resolution, Image.BICUBIC)
    image.save(target)
    return target


def assign_scenario(caption: str, scenarios: Dict[str, ScenarioSpec]) -> Optional[str]:
    caption_lower = caption.lower()
    best_match: Optional[str] = None
    best_score = 0
    for name, spec in scenarios.items():
        matches = sum(1 for kw in spec.keywords if kw in caption_lower)
        if matches > best_score:
            best_match = name
            best_score = matches
    return best_match


def build_prompt_variants(
    base_caption: str,
    scenario_name: str,
    scenario_spec: ScenarioSpec,
    max_variants: int = 3,
) -> List[str]:
    templates = list(scenario_spec.prompt_templates)
    random.shuffle(templates)
    variants = []
    for template in templates[:max_variants]:
        variants.append(template.format(caption=base_caption, scenario=scenario_name.replace("_", " ")))
    return [base_caption] + variants


def build_records_from_msr_vtt(
    root: Path,
    limit: int,
    scenarios: Dict[str, ScenarioSpec],
    resolution: Tuple[int, int],
) -> List[Dict[str, str]]:
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
        if not video_path.exists():
            LOGGER.debug("Skipping %s: missing video file.", video_path)
            continue

        frame_path = extract_first_frame(video_path, root / "frames", resolution)
        video_id = video_path.stem
        caption = caption_lookup.get(video_id, "A descriptive prompt for the video.")
        scenario = assign_scenario(caption, scenarios)
        if not scenario:
            LOGGER.debug("Skipping %s: unable to assign scenario for caption '%s'.", video_id, caption)
            continue
        prompt_variants = build_prompt_variants(caption, scenario, scenarios[scenario])
        records.append(
            {
                "video_path": str(video_path.resolve()),
                "image_path": str(frame_path.resolve()),
                "caption": caption,
                "prompt_variants": prompt_variants,
                "scenario": scenario,
                "latent_path": str((root / "latents" / f"{video_id}.npz").resolve()),
            }
        )
    return records


def split_records(
    records: List[Dict[str, str]],
    validation_ratio: float,
) -> Dict[str, List[Dict[str, str]]]:
    by_scenario: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        by_scenario.setdefault(record["scenario"], []).append(record)

    train: List[Dict[str, str]] = []
    validation: List[Dict[str, str]] = []
    for scenario, items in by_scenario.items():
        random.shuffle(items)
        val_count = max(1, int(len(items) * validation_ratio)) if items else 0
        validation.extend(items[:val_count])
        train.extend(items[val_count:])
        LOGGER.info(
            "Scenario %s -> train=%d validation=%d (total=%d)",
            scenario,
            len(items[val_count:]),
            len(items[:val_count]),
            len(items),
        )
    return {"train": train, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scenario-faithful metadata splits.")
    parser.add_argument("--msr-vtt", type=Path, required=True, help="MSR-VTT dataset root.")
    parser.add_argument("--vatex", type=Path, help="Optional VATEX dataset root.")
    parser.add_argument("--panda70m", type=Path, help="Optional Panda-70M subset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output metadata JSON path.")
    parser.add_argument("--limit", type=int, default=1200, help="Maximum number of samples to inspect.")
    parser.add_argument("--scenarios", type=Path, required=True, help="Scenario config file (YAML or JSON).")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples per scenario reserved for validation.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}",
        help="Target WxH resolution for conditioning frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    random.seed(args.seed)

    try:
        width_str, height_str = args.resolution.lower().split("x")
        resolution = (int(width_str), int(height_str))
    except ValueError as exc:
        raise ValueError(f"Invalid resolution '{args.resolution}'. Expected format 'WIDTHxHEIGHT'.") from exc

    scenarios = load_scenarios(args.scenarios)
    records = build_records_from_msr_vtt(
        args.msr_vtt,
        args.limit,
        scenarios=scenarios,
        resolution=resolution,
    )

    split = split_records(records, args.validation_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    LOGGER.info(
        "Wrote %d train / %d validation records to %s",
        len(split["train"]),
        len(split["validation"]),
        args.output,
    )


if __name__ == "__main__":
    main()

