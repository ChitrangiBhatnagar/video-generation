# VideoFusion: Video Generation from Text - Getting Started Guide

## Overview

This project implements **VideoFusion** - a Decomposed Diffusion Model (DPM) for high-quality video generation from text prompts and conditioning images. The system uses:

- **CogVideoX-5B-I2V** model from THUDM (Tsinghua University)
- **PyTorch + Diffusers** for deep learning
- **LoRA fine-tuning** for scenario-specific adaptation
- **FastAPI** for production deployment

## Quick Setup

### 1. Environment Activation
The virtual environment is already set up. Activate it:

```powershell
cd "C:\Users\chitrangi bhatnagar\video-generation"
& ".\venv\Scripts\Activate.ps1"
```

### 2. Verify Installation
Check that all dependencies are installed:

```powershell
python -m src.test_setup
```

This will:
- Test CUDA availability (optional, CPU fallback works)
- Load the CogVideoX-5B-I2V model
- Verify tensor operations work correctly

## Usage Patterns

### Pattern 1: Baseline Video Generation (Text-to-Video)

Generate a 6-second video from a text prompt and a conditioning image:

```powershell
python -m src.baseline_inference `
  --input-image data/samples/sample_image_1.png `
  --prompt "Press briefing in modern newsroom" `
  --output-path outputs/examples/my_video.mp4
```

**Parameters:**
- `--input-image`: Reference image to condition the video generation
- `--prompt`: Text description of the scene/action
- `--output-path`: Where to save the MP4 file
- `--seed`: Random seed for reproducibility (optional)
- `--num-inference-steps`: Quality vs speed tradeoff (default: 40)
- `--guidance-scale`: How strictly to follow the prompt (default: 6.5)

**Output:**
- MP4 video file (720×480, 8 fps, ~6 seconds)
- Logs with timing, VRAM usage, and temporal consistency metrics

---

### Pattern 2: Optimized Inference (with LoRA)

For faster generation with scenario-specific fine-tuning:

```powershell
python -m src.optimized_inference `
  --input-image data/samples/sample_image_2.png `
  --prompt "Market commentary on breaking news" `
  --scenario market_floor `
  --lora-scale 0.85 `
  --output-path outputs/examples/optimized_markets.mp4
```

**New Parameters:**
- `--scenario`: Preset scenario (e.g., `market_floor`, `newsroom`, `studio`)
- `--lora-scale`: Strength of fine-tuned adapter (0.0-1.0)
- `--enable-scheduler-offload`: Memory optimization

---

### Pattern 3: Batch Processing

Generate multiple videos in one run:

```powershell
python -m src.batch_processor `
  --jobs-file data/metadata/batch_jobs.json `
  --mode optimized `
  --output-dir outputs/batch_runs
```

Create `data/metadata/batch_jobs.json`:
```json
[
  {
    "image": "data/samples/sample_image_1.png",
    "prompt": "Anchor reading morning news",
    "scenario": "newsroom",
    "output": "outputs/batch_runs/news_01.mp4"
  },
  {
    "image": "data/samples/sample_image_2.png",
    "prompt": "Market analyst discussing stocks",
    "scenario": "market_floor",
    "output": "outputs/batch_runs/market_01.mp4"
  }
]
```

---

### Pattern 4: LoRA Fine-Tuning

Train a scenario-specific adapter on your dataset:

```powershell
accelerate launch src/train_lora.py `
  --metadata data/metadata/scenario_metadata.json `
  --output-dir checkpoints/lora_unified `
  --config configs/training/lora_default.yaml `
  --num-epochs 10 `
  --learning-rate 0.0001
```

**Key steps:**
1. Prepare training data with `data/metadata/scenario_metadata.json`
2. Run training (outputs checkpoints every epoch)
3. Validate with `src/validation/validate_lora.py`
4. Use in inference with `--lora-path` flag

---

### Pattern 5: FastAPI Server (Production)

Start the REST API server:

```powershell
cd "C:\Users\chitrangi bhatnagar\video-generation"
& ".\venv\Scripts\Activate.ps1"
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

**POST /generate-video**
```powershell
$image = Get-Item "data/samples/sample_image_1.png"
$params = @{
    prompt = "Press briefing at city hall"
    duration_seconds = 6
    guidance_scale = 6.5
}

curl.exe -X POST "http://localhost:8000/generate-video" `
  -F "image=@$($image.FullName)" `
  -F "prompt=$($params.prompt)" `
  -F "duration_seconds=$($params.duration_seconds)"
```

**GET /health**
```powershell
curl.exe http://localhost:8000/health
```

Returns:
```json
{
  "status": "ok",
  "gpu_available": true,
  "model_loaded": true
}
```

---

## Available Sample Images

Three sample conditioning images are provided in `data/samples/`:

1. **sample_image_1.png** - Professional newsroom setting
2. **sample_image_2.png** - Business/trading floor setting
3. **sample_image_3.png** - Studio/interview setting

Use these with text prompts for various scenarios:

```powershell
# News scenario
python -m src.baseline_inference `
  --input-image data/samples/sample_image_1.png `
  --prompt "Breaking news announcement with scrolling ticker" `
  --output-path outputs/examples/breaking_news.mp4

# Markets scenario
python -m src.baseline_inference `
  --input-image data/samples/sample_image_2.png `
  --prompt "Stock market analysis with graphs updating" `
  --output-path outputs/examples/market_update.mp4

# Interview scenario
python -m src.baseline_inference `
  --input-image data/samples/sample_image_3.png `
  --prompt "Expert interview with animated charts" `
  --output-path outputs/examples/expert_interview.mp4
```

---

## Project Structure

```
src/
├── baseline_inference.py         # Text-to-video generation
├── optimized_inference.py        # LoRA-enhanced inference
├── batch_processor.py            # Batch video generation
├── api_server.py                 # FastAPI REST server
├── train_lora.py                 # LoRA fine-tuning
├── train_scenario_lora.py        # Scenario-specific training
├── temporal_consistency.py       # Video quality metrics
├── data_prep/                    # Dataset preparation
│   ├── download_msr_vtt.py
│   ├── download_vatex.py
│   ├── build_scenario_metadata.py
│   └── preprocess_video_latents.py
├── evaluation/                   # Benchmarking
│   ├── benchmark.py
│   └── clip_similarity.py
└── utils/                        # Utilities
    ├── logging_config.py
    ├── pipeline_utils.py
    └── seed_utils.py

configs/
├── model/                        # Model configurations
├── training/                     # LoRA training configs
└── data/                         # Dataset configs

data/
├── samples/                      # Sample images (ready to use)
├── metadata/                     # Metadata JSON files
├── raw/                          # Downloaded raw datasets
└── prepared/                     # Preprocessed latents

outputs/
└── examples/                     # Generated videos

checkpoints/
└── (Created after LoRA training)

```

---

## Troubleshooting

### Issue: Model Download Timeout
**Solution:** The model (5.7 GB) may take time to download. Use a stable internet connection and be patient. Files are cached in `~/.cache/huggingface/hub/`.

### Issue: CUDA Out of Memory
**Solutions:**
1. Reduce `num_inference_steps` from 40 to 20-30
2. Use CPU-only mode (slower but works): The code auto-detects and falls back to CPU
3. Enable scheduler offload in optimized_inference

### Issue: No GPU Detected
**Resolution:** CPU mode is automatically enabled. Performance will be slower but the code will work.

---

## Advanced Configuration

### Modify Default Parameters

Edit `configs/training/lora_default.yaml`:
```yaml
lora:
  rank: 64
  alpha: 128
  dropout: 0.05

training:
  batch_size: 8
  learning_rate: 0.0001
  num_epochs: 10
  warmup_steps: 500
```

### Use Different Models

Update `config.model_id` in source files to use alternative models:
```python
# From CogVideoX-5B-I2V to another variant
model_id: "THUDM/CogVideoX-2B"  # Smaller model
```

---

## Next Steps

1. **Generate Your First Video:**
   ```powershell
   python -m src.baseline_inference `
     --input-image data/samples/sample_image_1.png `
     --prompt "Your custom text prompt here" `
     --output-path outputs/examples/my_first_video.mp4
   ```

2. **Explore the API:**
   ```powershell
   # Start server
   uvicorn src.api_server:app --host 0.0.0.0 --port 8000
   
   # Test in another terminal
   curl.exe http://localhost:8000/health
   ```

3. **Fine-tune for Your Domain:**
   - Prepare metadata in `data/metadata/`
   - Run training with `accelerate launch src/train_lora.py`
   - Use trained adapter in inference

4. **Deploy:**
   ```powershell
   # Build Docker image
   docker build -t videofusion:latest .
   
   # Run container
   docker run --gpus all -p 8000:8000 videofusion:latest
   ```

---

## Resources

- **CogVideoX Paper:** https://arxiv.org/abs/2408.06072
- **Diffusers Library:** https://huggingface.co/docs/diffusers
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **Model Card:** https://huggingface.co/THUDM/CogVideoX-5B-I2V

---

## Support

For issues or questions:
1. Check `README.md` for project overview
2. Review `docs/` folder for detailed documentation
3. Check existing GitHub issues (if applicable)
4. Verify model is cached: `ls ~/.cache/huggingface/hub/`

Happy video generation! 🎬
