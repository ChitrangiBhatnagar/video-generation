from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from src.metrics import MetricResult, analyze_video
except ModuleNotFoundError:  # pragma: no cover
    MetricResult = None  # type: ignore[assignment]
    analyze_video = None  # type: ignore[assignment]

SUMMARY_PATH = Path("PROJECT_SUMMARY.json")


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def collect_video_metrics() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    videos_dir = Path("outputs/examples")
    for video_file in sorted(videos_dir.glob("*.mp4")):
        try:
            if analyze_video is not None:
                metric = analyze_video(video_file)
                results.append(
                    {
                        "video_path": str(metric.video_path),
                        "temporal_consistency": metric.temporal_consistency,
                        "avg_brightness": metric.avg_brightness,
                        "duration_seconds": metric.duration_seconds,
                        "fps": metric.fps,
                    }
                )
            else:
                frames = iio.imread(video_file)
                fps = 8
                if frames.ndim == 4:
                    diffs = np.diff(frames.astype(np.float32), axis=0)
                    variance = float(np.mean(diffs ** 2))
                    score = float(np.exp(-variance / 5e5))
                    brightness = float(frames.mean() / 255.0)
                    duration = frames.shape[0] / fps
                else:
                    score = 1.0
                    brightness = 0.5
                    duration = 0.0
                results.append(
                    {
                        "video_path": str(video_file),
                        "temporal_consistency": score,
                        "avg_brightness": brightness,
                        "duration_seconds": duration,
                        "fps": fps,
                        "note": "Simple frame-difference metric (fallback).",
                    }
                )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Failed to analyze {video_file}: {exc}")

    if not results:
        return {"videos": [], "avg_temporal_consistency": 0.0}

    avg_temp = mean(item["temporal_consistency"] for item in results)
    payload = {
        "videos": results,
        "avg_temporal_consistency": avg_temp,
    }
    return payload


def collect_config_hashes() -> List[Dict[str, str]]:
    config_paths = [
        Path("configs/model/cogvideox_baseline.yaml"),
        Path("configs/model/optimized_inference.yaml"),
        Path("configs/training/lora_default.yaml"),
        Path("configs/training/validation_prompts.yaml"),
    ]
    hashes: List[Dict[str, str]] = []
    for path in config_paths:
        if path.exists():
            hashes.append({"path": str(path), "sha256": hash_file(path)})
    return hashes


def collect_dataset_stats() -> Dict[str, Any]:
    metadata_path = Path("data/metadata/training_metadata.json")
    sample_count = 0
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
        sample_count = len(records)
    return {
        "training_metadata_path": str(metadata_path),
        "sample_count": sample_count,
    }


def collect_artifacts() -> Dict[str, Any]:
    checkpoints_dir = Path("checkpoints")
    adapters = sorted([str(p) for p in checkpoints_dir.glob("**/adapter_*")])
    return {
        "example_videos": sorted([str(p) for p in Path("outputs/examples").glob("*.mp4")]),
        "checkpoints": adapters,
    }


def main() -> None:
    summary: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": "THUDM/CogVideoX-5B-I2V",
            "project": "video-generation",
        },
        "metrics": collect_video_metrics(),
        "datasets": collect_dataset_stats(),
        "configs": collect_config_hashes(),
        "artifacts": collect_artifacts(),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[info] Wrote project summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

