## Dataset Preparation

### MSR-VTT (Primary Training Set)
- **Download**: `python -m src.data_prep.download_msr_vtt --output-dir data/raw/msr-vtt --limit 500`
- **Contents**: Annotations (`MSRVTT_data.json`), videos (`train-video/*.mp4`), generated manifest.
- **Prep Metadata**: `python -m src.data_prep.prepare_metadata --msr-vtt data/raw/msr-vtt --output data/metadata/training_metadata.json --limit 1500`
- **Storage**: ~29 GB for full dataset; quickstart limit trims to <5 GB.
- **Usage**: Provide `video_path`, `image_path` (first frame), `caption`, `scenario` fields for LoRA training.

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
  "scenario": "press_conference"
}
```

### Storage & Time Estimates
- Frame extraction (1k samples @ 720×480, 8 fps): ~45 minutes on quad-core CPU.
- Metadata build for 1.5k samples: ~5 minutes.
- Ensure `data/prepared` volume has ≥50 GB free to accommodate frames and cached tensors.

### Tips
- Cache Hugging Face downloads by exporting `HF_HOME=/data/hf-cache`.
- Maintain consistent frame rate (8 fps) to align with LoRA training config.
- Inspect metadata with `head data/metadata/training_metadata.json` before launching training.

