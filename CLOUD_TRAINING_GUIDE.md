# Cloud Training Guide - VideoFusion

## 🚨 Local Training Issue

Your local machine encountered a **MemoryError** (error code 3221225477) when trying to load the CogVideoX-5B-I2V model. This is because:

- **Model Size**: ~20GB when loaded
- **Required RAM**: 32GB+ (64GB recommended)
- **Your System**: Insufficient RAM for this large model

**Solution**: Use cloud GPU platforms for training. Here's how:

---

## ☁️ Option 1: Google Colab (Recommended - Easy & Fast)

### Step-by-Step Setup

1. **Go to Google Colab**
   - Visit: https://colab.research.google.com
   - Sign in with your Google account

2. **Create New Notebook**
   - Click "File" → "New notebook"

3. **Enable GPU**
   - Click "Runtime" → "Change runtime type"
   - Select "T4 GPU" or "A100 GPU" (if available)
   - Click "Save"

4. **Setup Your Project** (Run these cells)

```python
# Cell 1: Clone repository
!git clone https://github.com/ChitrangiBhatnagar/video-generation.git
%cd video-generation
!ls -la
```

```python
# Cell 2: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install diffusers transformers accelerate datasets peft opencv-contrib-python Pillow imageio imageio-ffmpeg scipy pandas tqdm pyyaml huggingface-hub fastapi uvicorn python-multipart httpx loguru rich wandb
```

```python
# Cell 3: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

```python
# Cell 4: Configure accelerate
!python setup_accelerate.py
```

```python
# Cell 5: Prepare training data
# Upload your training data or use samples
from google.colab import files

# Option A: Upload metadata file
uploaded = files.upload()

# Option B: Create sample metadata
import json
metadata = {
    "train": [
        {
            "video_path": "data/samples/sample_video_1.mp4",
            "image_path": "data/samples/sample_image_1.png",
            "prompt": "News anchor delivering breaking news",
            "scenario": "newsroom"
        }
    ],
    "validation": []
}

with open('data/metadata/scenario_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Metadata created")
```

```python
# Cell 6: Start training
!accelerate launch src/train_lora.py \
  --metadata data/metadata/scenario_metadata.json \
  --output-dir checkpoints/lora_unified \
  --config configs/training/lora_default.yaml \
  --num-epochs 5
```

```python
# Cell 7: Download trained model
from google.colab import files
import shutil

# Zip the checkpoint
!zip -r lora_trained.zip checkpoints/lora_unified/

# Download
files.download('lora_trained.zip')
```

### Colab Tips:
- **Free Tier**: Limited to ~12 hours per session
- **Colab Pro** ($10/month): Better GPUs, longer sessions
- **Save to Google Drive**: Mount drive to save checkpoints automatically

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
!mkdir -p /content/drive/MyDrive/video-generation-checkpoints
!cp -r checkpoints/* /content/drive/MyDrive/video-generation-checkpoints/
```

---

## ☁️ Option 2: Kaggle Notebooks (FREE GPU)

### Step-by-Step Setup

1. **Go to Kaggle**
   - Visit: https://www.kaggle.com
   - Sign in or create account (FREE)

2. **Create New Notebook**
   - Click "Code" → "New Notebook"
   - Title: "VideoFusion Training"

3. **Enable GPU**
   - Click "Settings" (right panel)
   - Under "Accelerator", select "GPU T4 x2" or "GPU P100"
   - Click "Save"

4. **Add Your Repository**
   - Click "Add data" → "GitHub"
   - Enter: `ChitrangiBhatnagar/video-generation`
   - Click "Add"

5. **Install Dependencies**

```python
# Cell 1: Navigate to project
import os
os.chdir('/kaggle/working')
!git clone https://github.com/ChitrangiBhatnagar/video-generation.git
os.chdir('/kaggle/working/video-generation')
```

```python
# Cell 2: Install packages
!pip install diffusers transformers accelerate datasets peft opencv-contrib-python imageio imageio-ffmpeg loguru rich wandb --quiet
```

```python
# Cell 3: Setup accelerate
!python setup_accelerate.py
```

```python
# Cell 4: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

```python
# Cell 5: Train
!accelerate launch src/train_lora.py \
  --metadata data/metadata/scenario_metadata.json \
  --output-dir /kaggle/working/checkpoints \
  --config configs/training/lora_default.yaml
```

```python
# Cell 6: Download trained model
from IPython.display import FileLink
import shutil

shutil.make_archive('/kaggle/working/trained_model', 'zip', '/kaggle/working/checkpoints')
FileLink('/kaggle/working/trained_model.zip')
```

### Kaggle Advantages:
- ✅ **FREE**: 30 hours/week GPU time
- ✅ **No credit card** required
- ✅ **P100 or T4 GPUs**
- ✅ **20GB datasets**

---

## ☁️ Option 3: RunPod (Budget-Friendly)

### Setup Guide

1. **Sign Up**
   - Visit: https://www.runpod.io
   - Create account

2. **Deploy Pod**
   - Click "Deploy" → "GPU Pods"
   - Select GPU: RTX 3090 ($0.44/hr) or RTX 4090 ($0.69/hr)
   - Template: "PyTorch"
   - Click "Deploy On-Demand"

3. **Connect via Jupyter**
   - Click "Connect" → "Start Jupyter Lab"
   - Opens Jupyter interface

4. **Setup in Terminal**

```bash
# Open terminal in Jupyter
git clone https://github.com/ChitrangiBhatnagar/video-generation.git
cd video-generation

pip install -r requirements.txt
python setup_accelerate.py

# Train
accelerate launch src/train_lora.py \
  --metadata data/metadata/scenario_metadata.json \
  --output-dir checkpoints/lora_unified \
  --config configs/training/lora_default.yaml
```

5. **Download Results**
   - Use Jupyter file browser to download `checkpoints/` folder

### RunPod Tips:
- 💰 Use **Spot Instances** (80% cheaper, can be interrupted)
- 📊 Monitor GPU usage in dashboard
- 💾 Save to network storage for persistence

---

## 📊 Quick Comparison

| Platform | Cost | GPU | Setup Time | Best For |
|----------|------|-----|------------|----------|
| **Kaggle** | FREE | P100/T4 | 5 min | Learning, Testing |
| **Colab Free** | FREE | T4 | 3 min | Quick experiments |
| **Colab Pro** | $10/mo | V100/A100 | 3 min | Regular use |
| **RunPod** | $0.44/hr | RTX 3090 | 10 min | Budget training |
| **Paperspace** | $8/mo + compute | Various | 15 min | Ease of use |

---

## 🎯 Recommended Workflow

### For This Project:

1. **Start with Kaggle** (FREE, 30hr/week)
   - Test training pipeline
   - Experiment with hyperparameters
   - Generate initial results

2. **Upgrade to Colab Pro** if needed ($10/month)
   - Longer training sessions
   - Better GPUs (A100)
   - More frequent training

3. **Use RunPod for production** ($0.44+/hr)
   - Large-scale training
   - Cost-effective for long jobs
   - More control over environment

---

## ⚡ Quick Start Commands

### Once you're on a cloud platform:

```bash
# Clone repo
git clone https://github.com/ChitrangiBhatnagar/video-generation.git
cd video-generation

# Install
pip install -r requirements.txt

# Configure
python setup_accelerate.py

# Train
accelerate launch src/train_lora.py \
  --metadata data/metadata/scenario_metadata.json \
  --output-dir checkpoints/lora_unified \
  --config configs/training/lora_default.yaml \
  --num-epochs 10 \
  --learning-rate 0.0001
```

---

## 📥 Getting Results Back

### Download Trained Model:

**Colab:**
```python
from google.colab import files
!zip -r trained_model.zip checkpoints/
files.download('trained_model.zip')
```

**Kaggle:**
```python
from IPython.display import FileLink
import shutil
shutil.make_archive('trained_model', 'zip', 'checkpoints')
FileLink('trained_model.zip')
```

**RunPod:**
- Use Jupyter file browser to download
- Or use `scp` to transfer files

---

## 🆘 Troubleshooting

### Out of Memory on Cloud:
```python
# Reduce batch size in config
# configs/training/lora_default.yaml
training:
  batch_size: 1  # Reduce from 2 or 4
  gradient_accumulation_steps: 8  # Increase to compensate
```

### Session Timeout:
- **Save checkpoints frequently** (already configured)
- Use Google Drive/cloud storage mounting
- Resume from last checkpoint

### Slow Downloads:
```python
# Use HF cache
from huggingface_hub import snapshot_download
snapshot_download("THUDM/CogVideoX-5B-I2V", cache_dir="/kaggle/working/hf_cache")
```

---

## 💡 Pro Tips

1. ✅ **Test locally** with the lightweight demo first
2. ✅ **Start with small experiments** on free tiers
3. ✅ **Monitor costs** on paid platforms
4. ✅ **Save checkpoints to cloud storage**
5. ✅ **Use mixed precision training** (already enabled)

---

## 🎉 You're Ready!

Choose your platform and start training in the cloud. Your local machine can continue running the lightweight demo for testing!

**Questions?** Check:
- `README_DETAILED.md` - Full documentation
- `GETTING_STARTED.md` - Usage examples
- GitHub Issues - Community support
