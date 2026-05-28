"""
run_job_link.py — Phase 6B: Program–Job Linker CLI entry point.

Links proven CNC programs to job metadata and backfills source-verified
material into the S/F database. All matching is deterministic.

Auto-detects the latest exports unless paths are supplied explicitly.
Never writes to P:\\ or G:\\. Never overwrites existing exports.

Usage:
    py run_job_link.py
    py run_job_link.py --exports-dir "C:\\my\\exports"
    py run_job_link.py --manifest exports/manifest_20260528_120000.csv
    py run_job_link.py --cuts exports/cuts_20260528_120000.csv
    py run_job_link.py --job-metadata exports/job_metadata_20260528_120000.csv
    py run_job_link.py --shared-print exports/shared_print_index_20260528_120000.csv
    py run_job_link.py --router-ops exports/router_operations_20260528_120000.csv
"""

import argparse
import sys
from pathlib import Path

from src.job_linker import run_job_link
from src.utils import get_logger

logger = get_logger("run_job_link")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 6B — Program–Job Linker and Material Backfill"
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=Path(__file__).parent / "exports",
        help="Directory containing input CSVs and receiving output CSVs (default: exports/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Explicit path to manifest_*.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--cuts",
        type=Path,
        default=None,
        help="Explicit path to cuts_*.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--job-metadata",
        type=Path,
        default=None,
        help="Explicit path to job_metadata_*.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--shared-print",
        type=Path,
        default=None,
        help="Explicit path to shared_print_index_*.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--router-ops",
        type=Path,
        default=None,
        help="Explicit path to router_operations_*.csv (auto-detected if omitted)",
    )
    args = parser.parse_args()

    links_path, backfill_path, context_path = run_job_link(
        exports_dir=args.exports_dir,
        manifest_path=args.manifest,
        cuts_path=args.cuts,
        job_metadata_path=args.job_metadata,
        shared_print_path=args.shared_print,
        router_ops_path=args.router_ops,
    )

    print("\nExports written:")
    print(f"  Program–job links : {links_path}")
    print(f"  Material backfill : {backfill_path}")
    print(f"  Router context    : {context_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
