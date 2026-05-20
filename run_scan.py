#!/usr/bin/env python3
"""
run_scan.py — Scan P:\\ for proven CNC programs and export a manifest.

Usage:
    python run_scan.py
    python run_scan.py --root "P:\\" --exports exports --days 730
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.scanner import run_scan
from src.utils import get_logger

logger = get_logger("run_scan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Proven CNC folders and export a program manifest."
    )
    parser.add_argument(
        "--root",
        default=r"P:\\",
        help="Root CNC drive to scan (default: P:\\)",
    )
    parser.add_argument(
        "--exports",
        default="exports",
        help="Directory for manifest output files (default: exports/)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Maximum file age in days (default: 730 = 2 years)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    exports_dir = Path(__file__).parent / args.exports

    logger.info("=" * 60)
    logger.info("  Proven Program Engine — Scanner")
    logger.info("=" * 60)

    csv_path, json_path = run_scan(root, exports_dir, cutoff_days=args.days)

    logger.info("=" * 60)
    logger.info(f"  Manifest CSV  : {csv_path}")
    logger.info(f"  Manifest JSON : {json_path}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
