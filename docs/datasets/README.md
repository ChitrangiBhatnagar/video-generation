## Dataset Preparation

### MSR-VTT (Primary Training Set)
- **Download**: `python -m src.data_prep.download_msr_vtt --output-dir data/raw/msr-vtt --limit 1500`
- **Contents**: Annotations (`MSRVTT_data.json`), videos (`train-video/*.mp4`), generated manifest.
- **Scenario Metadata**: `python -m src.data_prep.build_scenario_metadata --msr-vtt data/raw/msr-vtt --scenarios configs/data/scenarios.yaml --output data/metadata/scenario_metadata.json --limit 1500 --validation-ratio 0.1`
- **Latent Preprocessing**: `python -m src.data_prep.preprocess_video_latents --metadata data/metadata/scenario_metadata.json --split train --output-dir data/prepared/latents --fps 8 --size 720x480 --augment --encode-latents`
- **Storage**: ~29 GB for full dataset; curated scenario subset stays <6 GB + latent cache.
- **Usage**: Metadata now supplies prompt variants, scenario labels, and cached latent paths for LoRA training.

### VATEX (Evaluation)
- **Download**: `python -m src.data_prep.download_vatex --output-dir data/raw/vatex --split validation --limit 50 --download-videos`
- **Purpose**: Multilingual captions for evaluation, cross-domain prompts.
- **Notes**: Storage ~18 GB for validation subset when including videos.

### Panda-70M (Optional Scaling)
- **Guidance**: `python -m src.data_prep.download_panda70m --output-dir data/raw/panda70m --instructions-only`
- **Access**: Requires request via official site; script writes README with manual steps.
- **Subset Suggestion**: 500 clips spanning urban, nature, transport, crowd scenarios (~120 GB).

### Metadata Schema
```json
{
  "video_path": "data/raw/msr-vtt/videos/video0.mp4",
  "image_path": "data/raw/msr-vtt/frames/video0.png",
  "caption": "A presenter discussing the latest headlines.",
  "prompt_variants": [
    "A presenter discussing the latest headlines.",
    "Press room camera shot capturing a press room briefing with A presenter discussing the latest headlines."
  ],
  "scenario": "press_room",
  "latent_path": "data/prepared/latents/press_room/video0.npz"
}
```

### Storage & Time Estimates
- Frame extraction (1k samples @ 720×480, 8 fps): ~45 minutes on quad-core CPU.
- Scenario metadata for 1.5k samples: ~6 minutes on 8-core CPU.
- Latent preprocessing (8 fps, 720×480, encode latents): ~40 minutes on single GPU.
- Ensure `data/prepared` volume has ≥60 GB free to accommodate frames and cached tensors.

### Tips
- Cache Hugging Face downloads by exporting `HF_HOME=/data/hf-cache`.
- Maintain consistent frame rate (8 fps) to align with LoRA training config.
- Inspect metadata with `jq '.train[0]' data/metadata/scenario_metadata.json` before launching training.

