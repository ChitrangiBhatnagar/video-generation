## Configuration Guide

### Model Configs (`configs/model`)
- `cogvideox_baseline.yaml`: baseline inference defaults (49 frames, 8 fps, 720×480, guidance 6.5, 40 steps).
- `optimized_inference.yaml`: optimized runner toggles FP16, VAE slicing/tiling, DPMSolver multistep scheduler, 28 steps.
- Override at runtime using CLI flags (`--num-frames`, `--guidance-scale`, `--num-inference-steps`).

### Training Configs (`configs/training`)
- `lora_default.yaml`:
  - Rank 64, alpha 128, dropout 0.05 (≈8.5 % trainable parameters).
  - Batch size 1 with gradient accumulation 4 → effective batch 4.
  - 3 epochs, mixed precision bf16, warmup 100 steps, checkpoint every 200 steps.
- `validation_prompts.yaml`: canonical scenarios referencing sample images for qualitative comparison.
- Update `num_video_frames` in `LoraTrainingConfig` if using longer clips.

### Data Configs (`configs/data`)
- `msr_vtt.yaml`: storage estimates, frame rates, scripts.
- `vatex.yaml`: evaluation split, multilingual note.
- `panda70m_subset.yaml`: guidance for optional high-scale subset.

### Customization Tips
- Add new scenario YAML under `configs/training` for domain-specific prompt/image pairs.
- Capture alternate scheduler settings in new model config and point CLI to it when benchmarking.
- Keep hash references up to date via `make summary` to reflect config changes in `PROJECT_SUMMARY.json`.

