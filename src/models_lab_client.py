"""
Client utilities for interacting with the ModelsLab text-to-video API.

This module exposes a single public function, `generate_video`, that wraps the
`https://modelslab.com/api/v7/video-fusion/text-to-video` endpoint and provides
type-safe parameters, environment-based authentication, and simple error
handling. A small CLI is also included to make manual invocation trivial.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import requests

API_URL = "https://modelslab.com/api/v7/video-fusion/text-to-video"
MODEL_ID = "veo-3.1"
VALID_ASPECT_RATIOS = {"16:9", "9:16"}
VALID_DURATIONS = {4, 6, 8}


class ModelsLabAPIError(RuntimeError):
    """Raised when the ModelsLab API responds with a non-successful result."""


def generate_video(
    prompt: str,
    aspect_ratio: str,
    duration: int,
    generate_audio: bool,
    negative_prompt: Optional[str] = None,
    enhance_prompt: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Request video generation from the ModelsLab API.

    Parameters
    ----------
    prompt:
        The textual description of the video content to generate.
    aspect_ratio:
        Aspect ratio string accepted by the API; currently "16:9" or "9:16".
    duration:
        Requested duration in seconds. Supported values are 4, 6, or 8.
    generate_audio:
        Whether the generated video should include audio.
    negative_prompt:
        Optional prompt describing artifacts or concepts to steer away from.
    enhance_prompt:
        Optional flag to ask the API to boost/enhance the supplied prompt.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON payload from the ModelsLab API response.

    Raises
    ------
    RuntimeError
        If the API key is missing or the request fails.
    ValueError
        If any of the provided arguments are outside supported ranges.
    """
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        raise ValueError(
            f"aspect_ratio must be one of {sorted(VALID_ASPECT_RATIOS)}, "
            f"got {aspect_ratio!r}"
        )

    if duration not in VALID_DURATIONS:
        raise ValueError(
            f"duration must be one of {sorted(VALID_DURATIONS)}, got {duration}"
        )

    api_key = os.getenv("MODELSLAB_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable MODELSLAB_API_KEY is not set.")

    payload: Dict[str, Any] = {
        "key": api_key,
        "model_id": MODEL_ID,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "generate_audio": generate_audio,
    }

    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt

    if enhance_prompt is not None:
        payload["enhance_prompt"] = enhance_prompt

    response = requests.post(API_URL, json=payload, timeout=60)

    if response.status_code != requests.codes.ok:
        raise ModelsLabAPIError(
            f"Request failed with status {response.status_code}: {response.text}"
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ModelsLabAPIError(
            "ModelsLab API returned a non-JSON response."
        ) from exc


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a video using the ModelsLab Veo 3.1 model."
    )
    parser.add_argument("--prompt", required=True, help="Text describing the video.")
    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        choices=sorted(VALID_ASPECT_RATIOS),
        help="Aspect ratio for the generated video.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=8,
        choices=sorted(VALID_DURATIONS),
        help="Duration of the generated video, in seconds.",
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Include audio in the generated video.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt to avoid specific content.",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Request prompt enhancement from the API if supported.",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    result = generate_video(
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        duration=args.duration,
        generate_audio=args.generate_audio,
        negative_prompt=args.negative_prompt,
        enhance_prompt=args.enhance_prompt if args.enhance_prompt else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

