PYTHON ?= python
ACCELERATE ?= accelerate

.PHONY: help setup data baseline optimized train train-scenarios preprocess validate benchmark api docker-build test lint summary

help:
	@echo "Targets:"
	@echo "  setup        Install dependencies and verify CUDA."
	@echo "  data         Download MSR-VTT sample subset and build metadata."
	@echo "  baseline     Run baseline inference with sample assets."
	@echo "  optimized    Run optimized inference (requires LoRA path optional)."
	@echo "  train        Launch LoRA training with accelerate."
	@echo "  validate     Generate validation clips with trained LoRA."
	@echo "  benchmark    Execute baseline vs optimized benchmark."
	@echo "  api          Start FastAPI server with uvicorn."
	@echo "  docker-build Build CUDA-enabled deployment image."
	@echo "  test         Run smoke pytest suite."
	@echo "  lint         Run static checks (ruff optional)."
	@echo "  summary      Regenerate PROJECT_SUMMARY.json."

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m src.test_setup

data:
	$(PYTHON) -m src.data_prep.download_msr_vtt --output-dir data/raw/msr-vtt --limit 25
	$(PYTHON) -m src.data_prep.build_scenario_metadata --msr-vtt data/raw/msr-vtt --scenarios configs/data/scenarios.yaml --output data/metadata/scenario_metadata.json --limit 1500 --validation-ratio 0.1

baseline:
	$(PYTHON) -m src.baseline_inference --input-image data/samples/sample_image_1.png --prompt "Press briefing in modern newsroom" --output-path outputs/examples/baseline_press_conference.mp4

optimized:
	$(PYTHON) -m src.optimized_inference --input-image data/samples/sample_image_2.png --prompt "Anchor delivering market update" --scenario market_floor --lora-scale 0.85 --output-path outputs/examples/optimized_markets.mp4

train:
	$(ACCELERATE) launch src/train_lora.py --metadata data/metadata/scenario_metadata.json --output-dir checkpoints/lora_unified --config configs/training/lora_default.yaml

train-scenarios:
	$(ACCELERATE) launch src/train_scenario_lora.py --metadata data/metadata/scenario_metadata.json --output-root checkpoints/scenario_loras --config configs/training/lora_default.yaml

preprocess:
	$(PYTHON) -m src.data_prep.preprocess_video_latents --metadata data/metadata/scenario_metadata.json --split train --output-dir data/prepared/latents --fps 8 --size 720x480 --augment --encode-latents

validate:
	$(PYTHON) -m src.validation.validate_lora --lora checkpoints/lora_unified/adapter_final --scenarios configs/training/validation_prompts.yaml

benchmark:
	$(PYTHON) -m src.evaluation.benchmark --input-image data/samples/sample_image_1.png --prompt "News panel discussing current events" --mode both --csv outputs/benchmarks.csv

api:
	uvicorn src.api_server:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t video-generation .

test:
	pytest src/test_suite.py -k smoke

lint:
	ruff src || true

summary:
	$(PYTHON) scripts/update_project_summary.py

