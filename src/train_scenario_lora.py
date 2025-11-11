"""
Quick start
-----------
python -m src.train_scenario_lora \
    --metadata data/metadata/scenario_metadata.json \
    --output-root checkpoints/scenario_loras \
    --config configs/training/lora_default.yaml \
    --scenarios press_room market_floor

Expected output
---------------
- Trains one LoRA adapter per selected scenario with shared hyperparameters.
- Saves adapters under `<output-root>/<scenario>/` with epoch/final checkpoints.
- Logs validation metrics per scenario and aggregates summary CSV.

Description
-----------
Convenience orchestrator that fine-tunes scenario-specific LoRA adapters using
the base training loop (`train_lora`). Each scenario is trained with filtered
metadata records and its own output directory while reusing the same config.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

from .train_lora import LoraTrainingConfig, train_lora


LOGGER = logging.getLogger(__name__)


def discover_scenarios(metadata_path: Path) -> List[str]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) if metadata_path.suffix in {".yaml", ".yml"} else None
    if payload is None:
        import json

        with metadata_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    if isinstance(payload, dict):
        scenarios = {record["scenario"] for split in payload.values() for record in split if "scenario" in record}
    else:
        scenarios = {record["scenario"] for record in payload if "scenario" in record}
    LOGGER.info("Discovered scenarios: %s", ", ".join(sorted(scenarios)))
    return sorted(scenarios)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scenario-specific LoRA adapters.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to scenario metadata JSON/YAML.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for scenario adapters.")
    parser.add_argument("--config", type=Path, required=True, help="Base training config.")
    parser.add_argument("--scenarios", nargs="*", help="Subset of scenarios to train. Default: all found in metadata.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--keep-top-k", type=int, help="Override adapter pruning limit.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as fh:
        config_kwargs = yaml.safe_load(fh) or {}
    config = LoraTrainingConfig(**config_kwargs)
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.keep_top_k is not None:
        config.keep_top_k_adapters = args.keep_top_k

    metadata_scenarios = discover_scenarios(args.metadata)
    scenarios = args.scenarios or metadata_scenarios
    missing = [scenario for scenario in scenarios if scenario not in metadata_scenarios]
    if missing:
        raise ValueError(f"Scenarios not found in metadata: {missing}")

    summary_rows = []
    for scenario in scenarios:
        LOGGER.info("=== Training scenario '%s' ===", scenario)
        scenario_config = LoraTrainingConfig(**config.__dict__)
        scenario_config.scenario_filter = [scenario]
        scenario_output = args.output_root / scenario
        scenario_output.mkdir(parents=True, exist_ok=True)

        train_lora(
            metadata_path=args.metadata,
            output_dir=scenario_output,
            config=scenario_config,
            config_overrides=config_kwargs,
        )
        summary_rows.append({"scenario": scenario, "output_dir": str(scenario_output)})

    summary_path = args.output_root / "scenario_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "output_dir"])
        writer.writeheader()
        writer.writerows(summary_rows)
    LOGGER.info("Scenario training complete. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()

