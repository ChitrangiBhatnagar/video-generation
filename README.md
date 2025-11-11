## Video Generation Pipeline

### Overview
- **Goal**: Generate 6-second 720×480 videos (8 fps) from a single conditioning image plus a scenario prompt, targeting newsroom and press workflows.
- **Model Stack**: Hugging Face *CogVideoX-5B-I2V* pipelines with PyTorch + Diffusers, LoRA fine-tuning, and GPU-aware optimizations (offload, FP16, schedulers).
- **Phases**: Environment ➝ Data ➝ Baseline inference ➝ LoRA training ➝ Optimization & deployment, each with concrete acceptance checks.

### Repository Layout
- `src/` pipelines, training, evaluation, utilities.
- `configs/` YAML configs for models, datasets, training, validation.
- `data/` sample assets, metadata stubs, download outputs.
- `outputs/examples/` example MP4 clips (~6 s, 8 fps).
- `checkpoints/` placeholder for LoRA adapters (created after training).
- `docs/` API reference, dataset notes, benchmark snapshots.
- `Makefile` shortcuts for setup/training/inference.
- `Dockerfile` CUDA-ready deployment image.

### Environment Setup
- **Python**: 3.11 within Conda or virtualenv.
- **CUDA**: 11.8+ recommended. Ensure `nvidia-smi` reports available GPU.
- Create env: `conda create -n video-gen python=3.11 -y`.
- Activate: `conda activate video-gen`.
- Install: `pip install -r requirements.txt`.
- Verify GPU + pipeline load: `python -m src.test_setup`.

### Quick Commands
- Baseline clip:  
  `python -m src.baseline_inference --input-image data/samples/sample_image_1.png --prompt "Press briefing at city hall" --output-path outputs/examples/baseline_cityhall.mp4`
- Optimized clip w/ LoRA:  
  `python -m src.optimized_inference --input-image data/samples/sample_image_2.png --prompt "Market commentary on breaking news" --lora checkpoints/lora_press/adapter_final`
- Batch run:  
  `python -m src.batch_processor --jobs-file data/metadata/batch_jobs.json --mode optimized --output-dir outputs/batch_runs`
- Train LoRA:  
  `accelerate launch src/train_lora.py --metadata data/metadata/training_metadata.json --output-dir checkpoints/lora_press --config configs/training/lora_default.yaml`
- Validate adapters:  
  `python -m src.validation.validate_lora --lora checkpoints/lora_press/adapter_final --scenarios configs/training/validation_prompts.yaml`
- Serve API:  
  `uvicorn src.api_server:app --host 0.0.0.0 --port 8000`

### Phase Acceptance Criteria
- **Phase 1 (Environment)**: `src/test_setup.py` loads CogVideoX in FP16, prints GPU name, confirms CPU offload.
- **Phase 2 (Data)**: `src/data_prep/*.py` download MSR-VTT/VATEX/Panda notes, metadata builder creates ≥1 k JSON entries; sample stubs provided in `data/metadata`.
- **Phase 3 (Baseline)**: `src/baseline_inference.py` generates 49-frame clips, logs latency, VRAM, optical-flow temporal score ≥0.6.
- **Phase 4 (LoRA)**: `src/train_lora.py` encodes video frames ➝ UNet w/ LoRA, logs loss, saves adapters per epoch, writes sample outputs to `outputs/examples`.
- **Phase 5 (Optimization)**: `src/optimized_inference.py` + `src/batch_processor.py` + `src/api_server.py` deliver optimized inference, batch throughput, FastAPI endpoints w/ `/generate-video` + `/health`.

### Temporal Consistency Metric
- Implemented in `src/temporal_consistency.py` using OpenCV Dual TV-L1 optical flow.
- Scores derived from flow magnitude variance with exponential normalization; acceptance threshold ≥0.6 for stable clips.
- Metric integrated into baseline, optimized inference, validation, metrics logging, and API responses.

### LoRA Training Highlights
- Dataset loader ingests `video_path`, `image_path`, prompt metadata; encodes videos via CogVideoX VAE.
- Conditioning images encoded to latent space and provided as added UNet context.
- LoRA adapters attached to attention modules with <10 % trainable params; gradient checkpointing + bf16 accelerate support.
- Metrics logged via Accelerate, optional CSV in `outputs/metrics.csv`; sample MP4s stored per epoch.

### Dataset Preparation
- `python -m src.data_prep.download_msr_vtt --output-dir data/raw/msr-vtt --limit 50`
- `python -m src.data_prep.prepare_metadata --msr-vtt data/raw/msr-vtt --output data/metadata/training_metadata.json --limit 1500`
- Optional evaluation slices: VATEX (`download_vatex.py`), Panda-70M guidance.
- See detailed instructions under `docs/datasets/README.md`.

### Benchmarks & Metrics
- Run combined benchmarks:  
  `python -m src.evaluation.benchmark --input-image data/samples/sample_image_1.png --prompt "Press room Q&A session" --mode both --csv outputs/benchmarks.csv`
- Metrics recorded: average latency, temporal score, peak VRAM, throughput.
- Snapshot metrics + artifact references stored in `PROJECT_SUMMARY.json`.

### API Endpoints
- `GET /health` returns pipeline status, model metadata.
- `POST /generate-video` (multipart) => MP4 path + temporal score; mode `baseline|optimized`, optional LoRA path, inference overrides.
- Artifact download: `GET /artifacts/{filename}` streaming MP4 from `outputs/examples`.
- See `docs/api/README.md` for payloads, expected responses, error codes.

### Testing & Quality Gates
- Smoke tests: `pytest src/test_suite.py -k smoke`.
- Full suite includes config validation, sample asset presence, summary schema checks.
- Lint (optional): `ruff src` or `python -m compileall src`.
- Ensure requirements for optical flow (`opencv-contrib-python`) installed before running metrics.

### Docker & Deployment
- Build image: `docker build -t video-generation .`
- Run on GPU host: `docker run --gpus all -p 8000:8000 -v $(pwd)/outputs:/app/outputs video-generation`
- Container entrypoint starts FastAPI server with optimized inference pipeline ready.

### Troubleshooting
- **CUDA OOM**: Reduce `num_frames`, `height/width`, or enable offload flags. Use `--num-inference-steps 28`.
- **Slow inference**: Switch to optimized runner, use DPMSolver scheduler, ensure FP16.
- **Optical flow import error**: Install `opencv-contrib-python==4.9.0.80`.
- **Dataset access**: Panda-70M requires manual approval—see `download_panda70m.py` guidance.
- **Weights caching**: Set `HF_HOME` or `TRANSFORMERS_CACHE` to reuse downloads.

### Contributing & Next Steps
- Extend benchmarks with alternative text-to-video pipelines for ablation.
- Add standardized press prompt templates for consistent evaluations.
- Integrate WandB or MLflow logging via `Accelerate` hooks.
- Consider distributed training across multiple GPUs via `accelerate config`.

