"""
run_job_scan.py — Phase 6A: Job Metadata Scanner CLI entry point.

Scans job folders and shared part prints. Exports timestamped CSVs.
Never writes to source folders.

Usage:
    py run_job_scan.py
    py run_job_scan.py --lookback-days 365
    py run_job_scan.py --job-root "G:\\Manufacturing\\JOB FOLDERS\\2024 Orders"
    py run_job_scan.py --prints-root "G:\\Manufacturing\\Programming\\CAM Files\\Shared Part Prints"
    py run_job_scan.py --exports-dir "C:\\my\\exports"
"""

import argparse
import sys
from pathlib import Path

from src.job_metadata_scanner import (
    JOB_FOLDERS_ROOT,
    SHARED_PRINTS_ROOT,
    _DEFAULT_LOOKBACK_DAYS,
    run_job_scan,
)
from src.utils import get_logger

logger = get_logger("run_job_scan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6A — Job Metadata Scanner")
    parser.add_argument(
        "--job-root",
        type=Path,
        default=JOB_FOLDERS_ROOT,
        help="Root folder for 2024 job orders (default: G:\\Manufacturing\\JOB FOLDERS\\2024 Orders)",
    )
    parser.add_argument(
        "--prints-root",
        type=Path,
        default=SHARED_PRINTS_ROOT,
        help="Root folder for shared part prints (default: G:\\Manufacturing\\Programming\\CAM Files\\Shared Part Prints)",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=Path(__file__).parent / "exports",
        help="Directory for CSV exports (default: exports/)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=_DEFAULT_LOOKBACK_DAYS,
        help=f"Only include files modified within this many days (default: {_DEFAULT_LOOKBACK_DAYS})",
    )
    args = parser.parse_args()

    meta_path, prints_path, ops_path = run_job_scan(
        job_root=args.job_root,
        prints_root=args.prints_root,
        exports_dir=args.exports_dir,
        lookback_days=args.lookback_days,
    )

    print(f"\nExports written:")
    print(f"  Job metadata    : {meta_path}")
    print(f"  Shared prints   : {prints_path}")
    print(f"  Router ops      : {ops_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
