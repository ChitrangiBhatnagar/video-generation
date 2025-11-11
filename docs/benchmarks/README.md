## Benchmark Notes

### Baseline vs Optimized Workflow
- Execute `python -m src.evaluation.benchmark --input-image data/samples/sample_image_1.png --prompt "Press room Q&A session" --mode both --repetitions 3`.
- Outputs latency, temporal consistency, and peak VRAM for baseline and optimized pipelines.
- Results can be appended to `outputs/benchmarks.csv` for longitudinal tracking.

### Sample Metrics (placeholder hardware: RTX 4090 24 GB)
| Mode       | Latency (s) | Temporal Score | Peak VRAM (GB) |
|------------|-------------|----------------|----------------|
| Baseline   | 92.4 ± 4.1  | 0.63 ± 0.02    | 17.8           |
| Optimized  | 58.7 ± 2.9  | 0.67 ± 0.01    | 14.2           |

> Update the table after running on your target GPU. Include LoRA-enabled results when adapters are trained.

### Throughput Goals
- Target 60–90 s per 6 s clip on mid-range GPUs.
- Batch processor (`src/batch_processor.py`) reports per-job latency and temporal score for aggregated runs.
- Use `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` alongside runs to collect VRAM peaks.

### Tips
- Warm up pipelines before timing to amortize initial weights load.
- For consistent comparisons, fix seeds (`--seed`), inference steps, and resolution.
- Record hardware, driver, CUDA, and diffusers versions in benchmark logs for reproducibility.

