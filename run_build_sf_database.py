"""
run_build_sf_database.py — Build the final proven S/F database.

Auto-detects the latest exports from all prior phases and builds:
  exports/proven_sf_database_*.csv   — one row per S/F cut record
  exports/proven_sf_summary_*.csv    — aggregated by tool/material/mode
  exports/proven_sf_programmer_view_*.csv — clean programmer-facing ranges

Usage:
    py run_build_sf_database.py
    py run_build_sf_database.py --exports-dir "C:\\my\\exports"
    py run_build_sf_database.py --cuts exports/cuts_20260528_120000.csv
"""

import argparse
import sys
from pathlib import Path

from src.proven_sf_database import run_build_sf_database
from src.utils import get_logger

logger = get_logger("run_build_sf_database")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build proven S/F database from all phase outputs"
    )
    parser.add_argument("--exports-dir", type=Path,
                        default=Path(__file__).parent / "exports",
                        help="Directory containing input CSVs and receiving outputs (default: exports/)")
    parser.add_argument("--cuts",       type=Path, default=None,
                        help="Explicit cuts_*.csv path (auto-detected if omitted)")
    parser.add_argument("--links",      type=Path, default=None,
                        help="Explicit program_job_links_*.csv path")
    parser.add_argument("--tooldb",     type=Path, default=None,
                        help="Explicit tooldb_reference_*.csv path")
    parser.add_argument("--tooling",    type=Path, default=None,
                        help="Explicit tooling_review_*.csv path")
    parser.add_argument("--mat-cands",  type=Path, default=None,
                        help="Explicit material_candidates_*.csv path")
    parser.add_argument("--router-ctx", type=Path, default=None,
                        help="Explicit router_program_context_*.csv path")
    args = parser.parse_args()

    db_path, summ_path, prog_path = run_build_sf_database(
        exports_dir=args.exports_dir,
        cuts_path=args.cuts,
        links_path=args.links,
        tooldb_path=args.tooldb,
        tooling_path=args.tooling,
        mat_cands_path=args.mat_cands,
        router_ctx_path=args.router_ctx,
    )

    print("\nExports written:")
    print(f"  S/F database : {db_path}")
    print(f"  S/F summary  : {summ_path}")
    print(f"  Programmer view : {prog_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
