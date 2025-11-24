# VideoFusion: Text-to-Video Generation with Decomposed Diffusion Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **VideoFusion** implements decomposed diffusion models for high-quality video generation from text prompts and conditioning images. Built on CogVideoX-5B-I2V with LoRA fine-tuning for scenario-specific adaptation.

---

## 🎯 Overview

This project provides a complete pipeline for training and deploying video generation models:

- **Text-to-Video Generation**: Create 6-second videos (720×480, 8fps) from text descriptions
- **Image-Conditioned**: Use reference images to guide video generation
- **LoRA Fine-tuning**: Efficient scenario-specific adaptation
- **Production Ready**: FastAPI server with REST endpoints
- **GPU Optimized**: CUDA acceleration with memory-efficient techniques

### Key Features

✅ **Baseline Inference** - Quick video generation with pretrained models  
✅ **Optimized Inference** - Memory-efficient with scheduler offloading  
✅ **LoRA Training** - Fine-tune for custom scenarios (newsrooms, studios, etc.)  
✅ **Batch Processing** - Generate multiple videos in one run  
✅ **REST API** - Production-ready FastAPI server  
✅ **Temporal Consistency** - Optical flow metrics for video quality  
✅ **CLIP Scoring** - Semantic similarity evaluation  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 16GB+ RAM (32GB recommended)
- GPU with 8GB+ VRAM (optional, CPU mode available)
- Windows/Linux/macOS

### Installation

```powershell
# Clone the repository
git clone https://github.com/ChitrangiBhatnagar/video-generation.git
cd video-generation

# Create and activate virtual environment
python -m venv venv
& ".\venv\Scripts\Activate.ps1"  # Windows PowerShell
# source venv/bin/activate  # Linux/Mac

# Install dependencies
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets peft opencv-contrib-python Pillow imageio imageio-ffmpeg scipy pandas tqdm pyyaml huggingface-hub fastapi uvicorn[standard] python-multipart httpx loguru rich wandb --upgrade

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Generate Your First Video

```powershell
python -m src.baseline_inference `
  --input-image data/samples/sample_image_1.png `
  --prompt "Press briefing in modern newsroom" `
  --output-path outputs/examples/my_first_video.mp4
```

**Note**: First run will download the CogVideoX-5B-I2V model (~5.7GB). This may take 10-30 minutes depending on your connection.

---

## 📖 Usage Guide

### 1. Baseline Video Generation

Generate videos with the pretrained CogVideoX model:

```powershell
python -m src.baseline_inference `
  --input-image data/samples/sample_image_1.png `
  --prompt "Breaking news anchor delivering headlines" `
  --output-path outputs/examples/news.mp4 `
  --num-inference-steps 40 `
  --guidance-scale 6.5 `
  --seed 42
```

**Parameters:**
- `--input-image`: Path to conditioning image
- `--prompt`: Text description of the video content
- `--output-path`: Where to save the generated MP4
- `--num-inference-steps`: Quality vs speed (20-50, default: 40)
- `--guidance-scale`: How closely to follow prompt (1-20, default: 6.5)
- `--seed`: Random seed for reproducibility

**Output:**
- MP4 video (720×480, 8fps, ~6 seconds, 49 frames)
- Console logs with timing and VRAM usage
- Temporal consistency metrics

### 2. Optimized Inference

For faster generation with lower memory usage:

```powershell
python -m src.optimized_inference `
  --input-image data/samples/sample_image_2.png `
  --prompt "Market analyst discussing stock trends" `
  --scenario market_floor `
  --lora-scale 0.85 `
  --output-path outputs/examples/market.mp4 `
  --enable-scheduler-offload
```

**New Parameters:**
- `--scenario`: Preset scenario (newsroom, market_floor, studio)
- `--lora-path`: Path to trained LoRA adapter
- `--lora-scale`: LoRA strength (0.0-1.0)
- `--enable-scheduler-offload`: Enable memory optimization

### 3. Batch Processing

Generate multiple videos from a JSON file:

**Create `data/metadata/batch_jobs.json`:**
```json
[
  {
    "image": "data/samples/sample_image_1.png",
    "prompt": "Morning news anchor reading headlines",
    "scenario": "newsroom",
    "output": "outputs/batch/news_01.mp4"
  },
  {
    "image": "data/samples/sample_image_2.png",
    "prompt": "Financial expert analyzing market data",
    "scenario": "market_floor",
    "output": "outputs/batch/market_01.mp4"
  },
  {
    "image": "data/samples/sample_image_3.png",
    "prompt": "Interview with technology expert",
    "scenario": "studio",
    "output": "outputs/batch/interview_01.mp4"
  }
]
```

**Run batch processing:**
```powershell
python -m src.batch_processor `
  --jobs-file data/metadata/batch_jobs.json `
  --mode optimized `
  --output-dir outputs/batch_runs
```

### 4. LoRA Fine-Tuning

Train scenario-specific adapters for better quality:

#### Step 1: Prepare Metadata

Create `data/metadata/scenario_metadata.json`:
```json
{
  "train": [
    {
      "video_path": "data/raw/videos/newsroom_01.mp4",
      "image_path": "data/raw/images/newsroom_01.png",
      "prompt": "News anchor delivering breaking news",
      "scenario": "newsroom"
    }
  ],
  "validation": [
    {
      "video_path": "data/raw/videos/newsroom_val_01.mp4",
      "image_path": "data/raw/images/newsroom_val_01.png",
      "prompt": "Evening news broadcast",
      "scenario": "newsroom"
    }
  ]
}
```

#### Step 2: Configure Training

Edit `configs/training/lora_default.yaml`:
```yaml
lora:
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]

training:
  batch_size: 1
  learning_rate: 0.0001
  num_epochs: 10
  warmup_steps: 500
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  
optimizer:
  type: "adamw"
  weight_decay: 0.01
  
scheduler:
  type: "cosine"
  num_warmup_steps: 500
```

#### Step 3: Launch Training

```powershell
accelerate config  # Run once to configure accelerate

accelerate launch src/train_lora.py `
  --metadata data/metadata/scenario_metadata.json `
  --output-dir checkpoints/lora_newsroom `
  --config configs/training/lora_default.yaml `
  --num-epochs 10 `
  --learning-rate 0.0001
```

**Training outputs:**
- Checkpoints saved every epoch: `checkpoints/lora_newsroom/checkpoint-{epoch}`
- Final adapter: `checkpoints/lora_newsroom/adapter_final`
- Sample videos: `outputs/examples/training_samples/`
- Logs: `checkpoints/lora_newsroom/training.log`

#### Step 4: Validate LoRA

```powershell
python -m src.validation.validate_lora `
  --lora checkpoints/lora_newsroom/adapter_final `
  --scenarios configs/training/validation_prompts.yaml `
  --output-dir outputs/validation
```

### 5. Production API Server

Deploy as a REST API:

```powershell
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**

#### Health Check
```powershell
curl.exe http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "gpu_available": true,
  "model_loaded": true,
  "version": "0.1.0"
}
```

#### Generate Video
```powershell
curl.exe -X POST "http://localhost:8000/generate-video" `
  -F "image=@data/samples/sample_image_1.png" `
  -F "prompt=Press conference at government building" `
  -F "duration_seconds=6" `
  -F "guidance_scale=6.5"
```

Response:
```json
{
  "video_path": "outputs/api_generated/video_abc123.mp4",
  "duration": 6.0,
  "fps": 8,
  "resolution": "720x480",
  "temporal_consistency": 0.87,
  "generation_time": 45.3
}
```

---

## 🏗️ Project Structure

```
video-generation/
├── src/
│   ├── baseline_inference.py         # Basic video generation
│   ├── optimized_inference.py        # Memory-optimized generation
│   ├── batch_processor.py            # Batch processing
│   ├── api_server.py                 # FastAPI REST server
│   ├── train_lora.py                 # LoRA training script
│   ├── train_scenario_lora.py        # Scenario-specific training
│   ├── temporal_consistency.py       # Video quality metrics
│   ├── metrics.py                    # Evaluation metrics
│   ├── test_setup.py                 # Installation verification
│   ├── test_suite.py                 # Unit tests
│   ├── data_prep/                    # Dataset preparation
│   │   ├── download_msr_vtt.py
│   │   ├── download_vatex.py
│   │   ├── download_panda70m.py
│   │   ├── build_scenario_metadata.py
│   │   └── preprocess_video_latents.py
│   ├── evaluation/                   # Benchmarking & evaluation
│   │   ├── benchmark.py
│   │   └── clip_similarity.py
│   ├── utils/                        # Utilities
│   │   ├── logging_config.py
│   │   ├── pipeline_utils.py
│   │   └── seed_utils.py
│   └── validation/
│       └── validate_lora.py
├── configs/
│   ├── model/
│   │   ├── cogvideox_baseline.yaml
│   │   └── optimized_inference.yaml
│   ├── training/
│   │   ├── lora_default.yaml
│   │   └── validation_prompts.yaml
│   └── data/
│       ├── msr_vtt.yaml
│       ├── vatex.yaml
│       ├── panda70m_subset.yaml
│       └── scenarios.yaml
├── data/
│   ├── samples/                      # Example conditioning images
│   │   ├── sample_image_1.png
│   │   ├── sample_image_2.png
│   │   └── sample_image_3.png
│   ├── metadata/                     # Training/validation metadata
│   ├── raw/                          # Downloaded datasets
│   └── prepared/                     # Preprocessed latents
├── outputs/
│   └── examples/                     # Generated videos
├── checkpoints/                      # LoRA adapters (created during training)
├── docs/                             # Documentation
├── requirements.txt
├── Makefile
├── Dockerfile
├── README.md
└── GETTING_STARTED.md
```

---

## 🔧 Configuration

### Model Configuration

Edit `configs/model/cogvideox_baseline.yaml`:
```yaml
model:
  model_id: "THUDM/CogVideoX-5B-I2V"
  torch_dtype: "float16"  # or "float32" for CPU
  
generation:
  num_frames: 49
  fps: 8
  height: 480
  width: 720
  guidance_scale: 6.5
  num_inference_steps: 40
  
optimization:
  enable_model_cpu_offload: true
  enable_sequential_cpu_offload: true
  enable_vae_slicing: true
  enable_vae_tiling: true
```

### Training Configuration

Key parameters in `configs/training/lora_default.yaml`:

- **rank**: LoRA rank (16-128, higher = more capacity but slower)
- **alpha**: LoRA alpha (usually 2x rank)
- **learning_rate**: 1e-4 to 1e-5 typical range
- **batch_size**: Depends on VRAM (1-4 typical)
- **gradient_accumulation_steps**: Effective batch size multiplier

---

## 📊 Evaluation & Metrics

### Temporal Consistency

Measures frame-to-frame coherence using optical flow:

```powershell
python -m src.temporal_consistency `
  --video-path outputs/examples/my_video.mp4 `
  --output-csv outputs/metrics/temporal_scores.csv
```

Score interpretation:
- **> 0.8**: Excellent temporal consistency
- **0.6-0.8**: Good consistency
- **< 0.6**: Poor consistency (may have flickering)

### CLIP Similarity

Measures semantic alignment between prompt and video:

```powershell
python -m src.evaluation.clip_similarity `
  --video-path outputs/examples/my_video.mp4 `
  --prompt "Your text prompt here" `
  --output-csv outputs/metrics/clip_scores.csv
```

### Benchmark Comparison

Compare baseline vs optimized inference:

```powershell
python -m src.evaluation.benchmark `
  --input-image data/samples/sample_image_1.png `
  --prompt "News anchor reading headlines" `
  --mode both `
  --csv outputs/benchmarks.csv
```

---

## 🐛 Troubleshooting

### Import Errors with Accelerate

**Error:**
```
ImportError: attempted relative import with no known parent package
```

**Solution:** The training script has been fixed to handle both module and direct execution. Ensure you're using:
```powershell
accelerate launch src/train_lora.py ...
```

### CUDA Out of Memory

**Solutions:**
1. Reduce `num_inference_steps` (try 20-30 instead of 40)
2. Enable memory optimizations in optimized_inference
3. Reduce video resolution in config
4. Use CPU mode (automatic fallback)

### Model Download Timeout

**Solutions:**
1. Check internet connection stability
2. Resume download (partial files are cached)
3. Use VPN if Hugging Face is blocked
4. Manually download from: https://huggingface.co/THUDM/CogVideoX-5B-I2V

### Slow Generation on CPU

**Expected behavior:** CPU mode is 10-50x slower than GPU. Consider:
1. Using a cloud GPU instance (Google Colab, AWS, etc.)
2. Renting local GPU time
3. Reducing `num_inference_steps` for faster results

---

## 🎓 Advanced Usage

### Custom Scenarios

Create new scenario presets in `configs/data/scenarios.yaml`:

```yaml
custom_scenario:
  name: "Tech Conference"
  description: "Technology conference presentation setting"
  keywords: ["tech", "presentation", "conference", "stage"]
  lora_path: "checkpoints/lora_tech_conf/adapter_final"
  default_guidance: 7.0
```

### Multi-GPU Training

Configure accelerate for multi-GPU:

```powershell
accelerate config
# Select: multi-GPU, number of GPUs, mixed precision (fp16)

accelerate launch --multi_gpu --num_processes 2 src/train_lora.py ...
```

### Docker Deployment

```powershell
# Build image
docker build -t videofusion:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 -v ./outputs:/app/outputs videofusion:latest

# Run CPU-only
docker run -p 8000:8000 videofusion:latest
```

### Custom Datasets

Prepare your own dataset:

1. Organize videos and frames:
```
data/raw/my_dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── images/
    ├── frame_001.png
    ├── frame_002.png
    └── ...
```

2. Create metadata:
```powershell
python -m src.data_prep.prepare_metadata `
  --video-dir data/raw/my_dataset/videos `
  --image-dir data/raw/my_dataset/images `
  --output data/metadata/my_dataset.json `
  --prompts-file data/metadata/prompts.txt
```

3. Train LoRA:
```powershell
accelerate launch src/train_lora.py `
  --metadata data/metadata/my_dataset.json `
  --output-dir checkpoints/lora_custom
```

---

## 📚 Resources

### Papers
- **CogVideoX**: [Arxiv](https://arxiv.org/abs/2408.06072)
- **VideoFusion**: Decomposed Diffusion Models for Video Generation
- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Documentation
- [Diffusers Library](https://huggingface.co/docs/diffusers)
- [PEFT (LoRA)](https://huggingface.co/docs/peft)
- [Accelerate](https://huggingface.co/docs/accelerate)

### Model Card
- [CogVideoX-5B-I2V on Hugging Face](https://huggingface.co/THUDM/CogVideoX-5B-I2V)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **THUDM** for CogVideoX model
- **Hugging Face** for Diffusers library
- **Microsoft** for LoRA/PEFT implementation
- All contributors and the open-source community

---

## 📧 Contact

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/ChitrangiBhatnagar/video-generation/issues)
- Email: chitrangi.bhatnagar@example.com

---

**Built with ❤️ for the video generation community**
