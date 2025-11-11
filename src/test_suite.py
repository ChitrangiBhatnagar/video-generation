"""
Quick start
-----------
pytest src/test_suite.py -k "smoke"

Expected output
---------------
- Runs smoke tests validating configuration files, sample assets, and metadata.
- Provides placeholders for heavier integration tests once GPU access is
  available.

Description
-----------
Test suite aggregating smoke, integration, and regression checks for the video
generation pipeline. Designed to be invoked by CI and during phase acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_training_metadata() -> list[dict]:
    metadata_path = ROOT / "data" / "metadata" / "training_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_config_files() -> list[Path]:
    return list((ROOT / "configs").rglob("*.yaml"))


@pytest.mark.smoke
def test_training_metadata_exists() -> None:
    records = load_training_metadata()
    assert len(records) >= 10, "Expected at least 10 samples for smoke coverage"
    sample = records[0]
    assert "image_path" in sample and "caption" in sample and "scenario" in sample


@pytest.mark.smoke
def test_config_yaml_is_valid() -> None:
    configs = iter_config_files()
    assert configs, "No config files discovered"
    for config_file in configs:
        with config_file.open("r", encoding="utf-8") as fh:
            yaml.safe_load(fh)


@pytest.mark.regression
def test_sample_images_exist() -> None:
    samples_dir = ROOT / "data" / "samples"
    images = list(samples_dir.glob("*.png"))
    assert len(images) >= 2, "Expected at least two sample images"
    for image in images:
        assert image.stat().st_size > 0, f"Image {image} is empty"


@pytest.mark.integration
def test_project_summary_schema() -> None:
    summary_path = ROOT / "PROJECT_SUMMARY.json"
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert "metrics" in summary and "artifacts" in summary
    assert summary["metrics"]["avg_temporal_consistency"] >= 0.0

