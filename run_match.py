"""
run_match.py — Phase 3 CLI: load reference table, match tool summary, export candidates.

Usage:
    py run_match.py [--wb PATH] [--summary PATH] [--exports DIR]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.utils import get_logger
from src.reference_loader import load_reference, export_reference
from src.material_matcher import export_material_candidates

logger = get_logger("run_match")

_WB_DEFAULT = Path(__file__).parent / "Programming Machine and Tooling Lists.xlsx"
_EXPORTS_DIR = Path(__file__).parent / "exports"


def _find_latest(exports_dir: Path, pattern: str) -> Path | None:
    matches = sorted(exports_dir.glob(pattern))
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Phase 3: material candidate matching")
    parser.add_argument(
        "--wb",
        default=str(_WB_DEFAULT),
        help="Path to Excel workbook (default: Programming Machine and Tooling Lists.xlsx)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Path to tool_summary_*.csv (default: latest in exports/)",
    )
    parser.add_argument(
        "--exports",
        default=str(_EXPORTS_DIR),
        help="Output directory for exports (default: exports/)",
    )
    args = parser.parse_args()

    wb_path = Path(args.wb)
    exports_dir = Path(args.exports)
    exports_dir.mkdir(exist_ok=True)

    if not wb_path.exists():
        logger.error(f"Workbook not found: {wb_path}")
        sys.exit(1)

    if args.summary:
        summary_path = Path(args.summary)
    else:
        summary_path = _find_latest(exports_dir, "tool_summary_*.csv")

    if summary_path is None or not summary_path.exists():
        logger.error("No tool_summary_*.csv found in exports/. Run run_parse.py first.")
        sys.exit(1)

    logger.info(f"Workbook:     {wb_path}")
    logger.info(f"Tool summary: {summary_path}")
    logger.info(f"Exports dir:  {exports_dir}")

    reference = load_reference(wb_path)
    ref_path = export_reference(reference, exports_dir)
    logger.info(f"Reference   -> {ref_path}")

    tool_summary_df = pd.read_csv(summary_path)
    logger.info(f"Loaded {len(tool_summary_df)} tool summary groups")

    candidates_path = export_material_candidates(tool_summary_df, reference, exports_dir)
    logger.info(f"Candidates  -> {candidates_path}")


if __name__ == "__main__":
    main()
