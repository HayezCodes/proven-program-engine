"""
run_tooling_review.py — Generate a user-editable tooling review CSV.

Loads the shop tooling reference from the Excel workbook and compares it
against parsed CNC program tool usage from the latest cuts CSV.

Outputs:
  exports/machine_tooling_reference_<timestamp>.csv  — normalized reference
  exports/tooling_review_<timestamp>.csv             — review sheet (user-editable)

Usage:
  py run_tooling_review.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.tooling_reference_loader import load_tooling_reference, export_tooling_reference
from src.tooling_matcher import export_tooling_review

_WB_DEFAULT = Path(__file__).parent / "Programming Machine and Tooling Lists.xlsx"
_EXPORTS_DIR = Path(__file__).parent / "exports"


def _find_latest(exports_dir: Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching a glob pattern."""
    matches = sorted(exports_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def main() -> None:
    if not _WB_DEFAULT.exists():
        print(f"ERROR: Workbook not found: {_WB_DEFAULT}")
        sys.exit(1)

    _EXPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load tooling reference
    reference = load_tooling_reference(_WB_DEFAULT)
    print(f"Loaded {len(reference)} tooling reference records.")
    export_tooling_reference(reference, _EXPORTS_DIR, timestamp=timestamp)

    # Find latest cuts CSV
    cuts_path = _find_latest(_EXPORTS_DIR, "cuts_*.csv")
    if cuts_path is None:
        print("ERROR: No cuts_*.csv found in exports/. Run run_parse.py first.")
        sys.exit(1)

    print(f"Using cuts file: {cuts_path.name}")
    cuts_df = pd.read_csv(cuts_path, dtype=str)

    # Convert tool_number to numeric where possible
    if "tool_number" in cuts_df.columns:
        cuts_df["tool_number"] = pd.to_numeric(cuts_df["tool_number"], errors="coerce")

    # Export review
    review_path = export_tooling_review(cuts_df, reference, _EXPORTS_DIR, timestamp=timestamp)
    print(f"Tooling review  -> {review_path.name}")
    print("Done. Open the review CSV to fill in review_action, corrected_description, and notes.")


if __name__ == "__main__":
    main()
