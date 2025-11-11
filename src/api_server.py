"""
Quick start
-----------
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload

curl -X POST \"http://localhost:8000/generate-video\" \\
  -F \"image=@data/samples/sample_image_1.png\" \\
  -F \"prompt=A calm city skyline at dusk with scrolling headlines\" \\
  -F \"duration_seconds=6\"

Expected output
---------------
- Returns a JSON payload with a presigned path to the generated MP4 artifact.
- `/health` endpoint responds with service metadata.

Description
-----------
FastAPI application exposing a GPU-backed endpoint for video generation using
CogVideoX baseline or optimized inference with optional LoRA adapters. Designed
for deployment via the provided Dockerfile.
"""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .baseline_inference import BaselineInferenceConfig, generate_video as baseline_generate, load_pipeline as load_baseline
from .optimized_inference import (
    OptimizedInferenceConfig,
    benchmark_generation as optimized_generate,
    load_pipeline as load_optimized,
)
from .temporal_consistency import TemporalConsistencyEvaluator
from .utils.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)
configure_logging()


@dataclass
class ServerState:
    baseline_pipeline = None
    optimized_pipeline = None
    evaluator = TemporalConsistencyEvaluator()


state = ServerState()
app = FastAPI(title="CogVideoX Video Generation API", version="0.1.0")


def get_baseline_pipeline() -> any:
    if state.baseline_pipeline is None:
        state.baseline_pipeline = load_baseline(BaselineInferenceConfig())
    return state.baseline_pipeline


def get_optimized_pipeline() -> any:
    if state.optimized_pipeline is None:
        state.optimized_pipeline = load_optimized(OptimizedInferenceConfig())
    return state.optimized_pipeline


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": "CogVideoX-5B-I2V",
            "pipelines": {
                "baseline_loaded": state.baseline_pipeline is not None,
                "optimized_loaded": state.optimized_pipeline is not None,
            },
        }
    )


@app.post("/generate-video")
async def generate_video(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    duration_seconds: int = Form(6),
    mode: str = Form("optimized"),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
) -> JSONResponse:
    if duration_seconds not in (6, 8, 10):
        raise HTTPException(status_code=400, detail="Supported durations: 6, 8, 10 seconds.")

    with tempfile.NamedTemporaryFile(suffix=Path(image.filename).suffix or ".png", delete=False) as temp:
        contents = await image.read()
        temp.write(contents)
        temp_path = Path(temp.name)

    try:
        if mode == "baseline":
            config = BaselineInferenceConfig()
            if guidance_scale:
                config.guidance_scale = guidance_scale
            if num_inference_steps:
                config.num_inference_steps = num_inference_steps
            pipeline = get_baseline_pipeline()
            output_path = baseline_generate(
                pipeline=pipeline,
                config=config,
                input_image=temp_path,
                prompt=prompt,
            )
        else:
            config = OptimizedInferenceConfig()
            if guidance_scale:
                config.guidance_scale = guidance_scale
            if num_inference_steps:
                config.num_inference_steps = num_inference_steps
            config.num_frames = int(duration_seconds * config.fps)
            pipeline = get_optimized_pipeline()
            output_path = optimized_generate(
                pipeline=pipeline,
                config=config,
                input_image=temp_path,
                prompt=prompt,
                lora_path=None,
            )
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    score = state.evaluator.score_video_file(output_path)
    return JSONResponse(
        {
            "status": "success",
            "output_path": str(output_path),
            "temporal_consistency": score,
            "duration_seconds": duration_seconds,
        }
    )


@app.get("/artifacts/{filename}")
def download_artifact(filename: str) -> FileResponse:
    file_path = Path("outputs/examples") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(file_path)

