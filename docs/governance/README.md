# Scenario LoRA Governance

## Dataset Traceability
- **Source datasets**: MSR-VTT (primary), optional VATEX/Panda-70M (evaluation/expansion).
- **Scenario config**: `configs/data/scenarios.yaml` documents keywords, prompt templates, and minimum counts.
- **Metadata artifact**: `data/metadata/scenario_metadata.json` records curated train/validation splits alongside prompt variants, scenario labels, and latent cache paths.
- **First-frame cache**: Deterministic 720×480 thumbnails stored under `data/raw/msr-vtt/frames/`.
- **Latent cache**: Preprocessed `.npz` tensors written to `data/prepared/latents/<scenario>/`.

## Training Reproducibility
- `accelerate launch src/train_lora.py --metadata data/metadata/scenario_metadata.json --output-dir checkpoints/lora_unified --config configs/training/lora_default.yaml`.
- Scenario-specific orchestration: `accelerate launch src/train_scenario_lora.py --metadata data/metadata/scenario_metadata.json --output-root checkpoints/scenario_loras --config configs/training/lora_default.yaml`.
- All configuration deltas logged in `checkpoints/<run>/validation_metrics.jsonl` with temporal + CLIP scores per epoch.
- Automatically prunes to top `keep_top_k_adapters` (default `3`) based on combined validation metrics.

## Evaluation Checklist
- Fixed validation prompts stored in `configs/training/validation_prompts.yaml`.
- Post-epoch validation captures:
  - Temporal consistency (optical-flow) scores.
  - CLIP similarity against prompts.
  - Sample MP4s in `checkpoints/<run>/sample_epoch_*`.
- `make validate` compares baseline vs LoRA for regression checks.

## Deployment Controls
- Scenario presets and LoRA scale limits managed via `configs/model/optimized_inference.yaml`.
- FastAPI endpoint exposes `scenario`, `lora_scale`, and `lora_path` form fields with server-side validation.
- `make optimized` uses preset `market_floor` adapter at 0.85 scale for smoke testing.
- Artifacts published through `/artifacts/{filename}` with temporal score metadata.

## Refresh Procedure
1. `make data preprocess` to regenerate metadata + latents when new footage arrives.
2. Train unified + scenario adapters (`make train`, `make train-scenarios`).
3. Review `validation_metrics.jsonl` + sample clips; document outcomes in changelog.
4. Update `configs/model/optimized_inference.yaml` with new adapter paths/scales.
5. Regenerate project summary (`make summary`) to snapshot config hashes and metrics.


