# PIB-VideoGen System

An AI system that converts multi-language text scripts and government images into branded videos for the Press Information Bureau.

## Project Overview

**Objective**: Build an AI system that converts multi-language text scripts and up to eight government images into 16–30s branded videos at 1280×720 px.

**Use Cases**:
- Policy briefings
- Emergency alerts
- Instructional guides
- Ceremonial announcements

## Research Foundations

- **VideoFusion (CVPR 2023)**: Decomposed diffusion (shared base noise λ=0.7, residual λ=0.3)
- **Align Your Latents (CVPR 2023)**: High-resolution latent diffusion with temporal alignment
- **Text2Video-Zero (ICCV 2023)**: Zero-shot generation for rapid emergency mode
- **Controllable Instructional (TMM 2024)**: Step-by-step text control
- **Fact-Checking Module**: Cross-reference PIB database and trusted APIs

## Key Components & Workflow

1. **Input Validation & Preprocessing**
   - Text: ≤ 100 words; auto-summarize longer scripts; detect language per sentence
   - Images: ≤ 8 inputs; auto-crop/resize to ≥ 512 px; super-resolve if needed

2. **Fact Checking**
   - Extract factual claims (dates, figures, locations)
   - Query PIB internal database + official APIs
   - Flag "⚠ Check Required" and halt if any claim is unverified or conflicting

3. **Noise Decomposition**
   - Generate shared base noise from the central image using pretrained Stable Diffusion (λ=0.7)
   - Generate per-frame residual noise from text via cross-frame attention (λ=0.3)

4. **Latent Synthesis**
   - Fuse text embeddings, images, base+residual noise in Align-Your-Latents module to produce 16-frame latent video

5. **Emergency Zero-Shot**
   - If urgent: true, bypass training, use Text2Video-Zero to produce < 60s turnaround for alerts

6. **Super-Resolution & Styling**
   - Upsample to 1280×720; overlay PIB seal; apply official color grading; embed subtitles and generate SRT

7. **Quality & Compliance Checks**
   - Metrics: FVD<200, IS>70, temporal coherence>0.9
   - Artifact detection: flicker, misalignment; adjust λ and regenerate up to 3 times
   - Accessibility: WCAG 2.1 AA compliance; ≥ 4s subtitle display
   - Final human review for any "⚠ Check Required" flags

## Project Structure

```
├── docs/                      # Documentation files
│   ├── architecture/          # System architecture diagrams
│   ├── api_specs/             # API specifications
│   ├── training/              # Training and fine-tuning schedules
│   └── qa/                    # QA checklists and review guidelines
│
├── src/                       # Source code
│   ├── preprocessing/         # Input validation and preprocessing
│   ├── fact_checking/         # Fact checking module
│   ├── noise_decomposition/   # Noise decomposition module
│   ├── latent_synthesis/      # Latent synthesis module
│   ├── emergency_zeroshot/    # Emergency zero-shot module
│   ├── super_resolution/      # Super-resolution and styling
│   └── quality_compliance/    # Quality and compliance checks
│
├── models/                    # Model files and weights
│   ├── base/                  # Base models
│   ├── fine_tuned/            # Fine-tuned models
│   └── version_control/       # Version control for models
│
├── data/                      # Data files
│   ├── sample_inputs/         # Sample input texts and images
│   ├── fact_check_sources/    # Fact checking data sources
│   └── output_samples/        # Sample output videos
│
├── tests/                     # Test files
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── performance/           # Performance tests
│
├── config/                    # Configuration files
│   ├── preprocessing/         # Preprocessing configuration
│   ├── models/                # Model configuration
│   └── quality/               # Quality metrics configuration
│
└── scripts/                   # Utility scripts
    ├── setup/                 # Setup scripts
    ├── training/              # Training scripts
    └── evaluation/            # Evaluation scripts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/pib/video-generation.git
cd video-generation

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from pib_videogen import VideoGenerator

# Initialize the generator
generator = VideoGenerator(config_path="config/default.yaml")

# Generate a video
video_path = generator.generate(
    text="Sample text for video generation",
    images=["path/to/image1.jpg", "path/to/image2.jpg"],
    urgent=False
)

print(f"Video generated at: {video_path}")
```

## Governance & Maintenance

- Document all data sources and API endpoints for fact checks
- Version control for models and λ parameters
- Quarterly audit of video outputs and fact-check logs
- Training pipeline: continuous fine-tuning of base and residual generators with new PIB datasets

## Deliverables

- System architecture diagram
- Detailed API spec for fact-checking
- Model training & fine-tuning schedules
- QA checklists and human review guidelines
- Prototype generator with sample videos for all use cases

## License

Government of India - Press Information Bureau