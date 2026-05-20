#!/usr/bin/env python3
"""
run_parse.py — Parse proven CNC programs from a manifest and export speed/feed records.

Usage:
    python run_parse.py                              # uses latest manifest in exports/
    python run_parse.py --manifest exports/manifest_20250120_143000.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.parser import parse_from_manifest
from src.utils import get_logger

logger = get_logger("run_parse")


def _latest_manifest(exports_dir: Path) -> Path | None:
    """Return the most-recently-created manifest CSV, or None if none exist."""
    manifests = sorted(exports_dir.glob("manifest_*.csv"), reverse=True)
    return manifests[0] if manifests else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse proven CNC programs and export speed/feed extraction records."
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest CSV (default: latest manifest_*.csv in --exports dir)",
    )
    parser.add_argument(
        "--exports",
        default="exports",
        help="Export directory (default: exports/)",
    )
    args = parser.parse_args()

    exports_dir = Path(__file__).parent / args.exports

    if args.manifest:
        manifest_csv = Path(args.manifest)
    else:
        manifest_csv = _latest_manifest(exports_dir)
        if manifest_csv is None:
            logger.error(
                "No manifest CSV found in exports/. Run run_scan.py first."
            )
            return 1
        logger.info(f"Using latest manifest: {manifest_csv.name}")

    if not manifest_csv.exists():
        logger.error(f"Manifest file not found: {manifest_csv}")
        return 1

    logger.info("=" * 60)
    logger.info("  Proven Program Engine — Parser")
    logger.info("=" * 60)

    outputs = parse_from_manifest(manifest_csv, exports_dir)

    logger.info("=" * 60)
    logger.info(f"  Cuts CSV       : {outputs['cuts']}")
    logger.info(f"  Parser summary : {outputs['summary']}")
    logger.info(f"  Tool summary   : {outputs['tool_summary']}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
