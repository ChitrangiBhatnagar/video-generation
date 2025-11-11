"""
Quick start
-----------
python -m src.test_setup

Expected output
---------------
- Prints CUDA availability, GPU name, and confirms CogVideoX pipeline loaded
  successfully with CPU offload enabled.

Description
-----------
Environment validation script to ensure PyTorch, CUDA, and CogVideoX are
properly configured before running expensive training or inference phases.
"""

from __future__ import annotations

import logging

import torch
from diffusers import CogVideoXImageToVideoPipeline

from .utils.logging_config import configure_logging
from .utils.pipeline_utils import enable_offload_for_pipeline


LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))

    LOGGER.info("Loading CogVideoX-5B-I2V with float16 weights...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5B-I2V",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
    enable_offload_for_pipeline(pipe)
    LOGGER.info("Pipeline loaded successfully with CPU offload.")


if __name__ == "__main__":
    main()

