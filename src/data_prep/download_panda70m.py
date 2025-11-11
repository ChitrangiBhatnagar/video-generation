"""
Quick start
-----------
python -m src.data_prep.download_panda70m --output-dir data/raw/panda70m --instructions-only

Expected output
---------------
- Downloads or prints curated subset instructions for Panda-70M dataset access.
- Creates README with usage and licensing notes.

Description
-----------
Provides guidance for acquiring a manageable Panda-70M subset with attention to
licensing requirements. Supports dry-run instructions when direct download is
not feasible in automated environments.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


INSTRUCTIONS = """\
Panda-70M Subset Instructions
=============================

1. Request access via the official website: https://panda70m.github.io/
2. After approval, download the curated subset referenced in configs/data/panda70m_subset.yaml.
3. Suggested subset: select 500 clips covering urban, nature, transport, and crowd scenarios.
4. Store videos under data/raw/panda70m/videos and metadata CSV under data/raw/panda70m/annotations.
5. Estimated storage: ~120 GB for the recommended subset. Reserve adequate disk and GPU bandwidth.

When running this script without --instructions-only, place approved download
commands (e.g., gsutil / wget) inside scripts/data/panda70m_fetch.sh, marked as
commented placeholders until credentials are available.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guidance for Panda-70M subset acquisition.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instructions-only", action="store_true", help="Only write instructions, skip download step.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instructions_path = args.output_dir / "README.md"
    instructions_path.write_text(INSTRUCTIONS, encoding="utf-8")
    LOGGER.info("Wrote Panda-70M instructions to %s", instructions_path)

    if args.instructions_only:
        LOGGER.info("Skipping download; manual steps required due to licensing.")
        return

    fetch_script = args.output_dir / "panda70m_fetch.sh"
    fetch_script.write_text("#!/bin/bash\n# Add gsutil or wget commands here after access is granted.\n", encoding="utf-8")
    LOGGER.info("Created placeholder fetch script at %s", fetch_script)


if __name__ == "__main__":
    main()

