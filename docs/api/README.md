## API Reference

### Base URL
- Local development: `http://localhost:8000`
- Docker container: `http://<host>:8000`

### Authentication
- API is currently unsecured for internal experimentation. Front the service with gateway auth for production.

### Endpoints
- **GET `/health`**
  - Returns model + pipeline readiness.
  - Response:
    ```json
    {
      "status": "ok",
      "model": "CogVideoX-5B-I2V",
      "pipelines": {
        "baseline_loaded": true,
        "optimized_loaded": false
      }
    }
    ```
- **POST `/generate-video`**
  - Multipart form fields:
    - `image`: uploaded PNG/JPG, 720×480 recommended.
    - `prompt`: scenario description text.
    - `duration_seconds`: 6 (default), 8, or 10.
    - `mode`: `"optimized"` (default) or `"baseline"`.
    - Optional overrides: `guidance_scale`, `num_inference_steps`.
    - Scenario-aware options (optimized mode):
      - `scenario`: key present in `configs/model/optimized_inference.yaml::lora_presets`.
      - `lora_scale`: float (0.0–1.5) to dial LoRA strength; `0` effectively disables adapter influence.
      - `lora_path`: absolute/relative adapter directory to override presets.
  - Success response:
    ```json
    {
      "status": "success",
      "output_path": "outputs/examples/optimized_markets.mp4",
      "temporal_consistency": 0.68,
      "duration_seconds": 6
    }
    ```
  - Errors:
    - `400` invalid payload or unsupported duration.
    - `500` inference failure (check logs).

- **GET `/artifacts/{filename}`**
  - Streams generated MP4s residing in `outputs/examples`.
  - Use to download example videos for QA or benchmarking.

### Usage Notes
- Default runtime enables CPU offload and FP16 for VRAM safety.
- Temporal consistency is computed via optical-flow metric (`TemporalConsistencyEvaluator`).
- Scenario presets and LoRA scaling leverage `configs/model/optimized_inference.yaml`; update the file for new adapters.
- Extendable: replace LoRA path handling or add API key checks by customizing `ServerState` and FastAPI dependencies.

### Example Curl
```bash
curl -X POST "http://localhost:8000/generate-video" \
  -F "image=@data/samples/sample_image_2.png" \
  -F "prompt=Anchor delivering a market wrap from the trading floor" \
  -F "duration_seconds=6" \
  -F "mode=optimized" \
  -F "scenario=market_floor" \
  -F "lora_scale=0.85"
```

