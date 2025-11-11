PYTHON ?= python
ACCELERATE ?= accelerate

.PHONY: help setup data baseline optimized train validate benchmark api docker-build test lint summary

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
	$(PYTHON) -m src.data_prep.prepare_metadata --msr-vtt data/raw/msr-vtt --output data/metadata/training_metadata.json --limit 1200

baseline:
	$(PYTHON) -m src.baseline_inference --input-image data/samples/sample_image_1.png --prompt "Press briefing in modern newsroom" --output-path outputs/examples/baseline_press_conference.mp4

optimized:
	$(PYTHON) -m src.optimized_inference --input-image data/samples/sample_image_2.png --prompt "Anchor delivering market update" --lora checkpoints/lora_press/adapter_final --output-path outputs/examples/optimized_markets.mp4

train:
	$(ACCELERATE) launch src/train_lora.py --metadata data/metadata/training_metadata.json --output-dir checkpoints/lora_press --config configs/training/lora_default.yaml

validate:
	$(PYTHON) -m src.validation.validate_lora --lora checkpoints/lora_press/adapter_final --scenarios configs/training/validation_prompts.yaml

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

