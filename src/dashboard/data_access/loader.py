"""
loader.py — Auto-detect and cache the latest export files.

All exports are read-only. Dashboard never writes to exports/ or P:\\.
Pure Python module — no Streamlit dependency. Fully testable.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_EXPORTS_DIR = Path(__file__).parents[3] / "exports"

# Dtype hints to suppress mixed-type warnings on large CSVs
_CUTS_DTYPES: dict = {
    "block_skip": "boolean",
    "is_duplicate": "boolean",
}

_CANDIDATE_DTYPES: dict = {
    "confidence_score": float,
}


def _find_latest(exports_dir: Path, pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching the glob pattern, or None."""
    matches = sorted(exports_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_csv(path: Optional[Path], dtype=None, **kwargs) -> Optional[pd.DataFrame]:
    """Load a CSV file. Returns None if path is None or file is unreadable."""
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=dtype, low_memory=False, **kwargs)
    except Exception:
        return None


def load_latest_cuts(exports_dir: Path = _EXPORTS_DIR) -> Optional[pd.DataFrame]:
    """Load the most recent cuts_*.csv export."""
    path = _find_latest(exports_dir, "cuts_*.csv")
    return _load_csv(path, dtype=_CUTS_DTYPES)


def load_latest_tool_summary(exports_dir: Path = _EXPORTS_DIR) -> Optional[pd.DataFrame]:
    """Load the most recent tool_summary_*.csv export."""
    path = _find_latest(exports_dir, "tool_summary_*.csv")
    return _load_csv(path)


def load_latest_material_candidates(exports_dir: Path = _EXPORTS_DIR) -> Optional[pd.DataFrame]:
    """Load the most recent material_candidates_*.csv export."""
    path = _find_latest(exports_dir, "material_candidates_*.csv")
    return _load_csv(path, dtype=_CANDIDATE_DTYPES)


def load_latest_tooling_review(exports_dir: Path = _EXPORTS_DIR) -> Optional[pd.DataFrame]:
    """Load the most recent tooling_review_*.csv export."""
    path = _find_latest(exports_dir, "tooling_review_*.csv")
    return _load_csv(path)


def get_export_status(exports_dir: Path = _EXPORTS_DIR) -> dict:
    """Return availability metadata for each export type."""
    patterns = {
        "cuts": "cuts_*.csv",
        "tool_summary": "tool_summary_*.csv",
        "material_candidates": "material_candidates_*.csv",
        "tooling_review": "tooling_review_*.csv",
    }
    status: dict[str, dict] = {}
    for key, pattern in patterns.items():
        path = _find_latest(exports_dir, pattern)
        if path:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            status[key] = {
                "found": True,
                "filename": path.name,
                "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                "size_kb": round(path.stat().st_size / 1024, 1),
                "path": path,
            }
        else:
            status[key] = {
                "found": False,
                "filename": None,
                "modified": None,
                "size_kb": 0,
                "path": None,
            }
    return status


def build_proven_tools_df(
    material_candidates: pd.DataFrame,
    tooling_review: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge material_candidates with tooling_review to produce the Proven Tools view.

    Joins on (machine_folder, tool_number). Tooling columns are suffixed _ref to
    avoid conflicts with material candidate columns.
    """
    if material_candidates is None or material_candidates.empty:
        return pd.DataFrame()

    df = material_candidates.copy()

    if tooling_review is not None and not tooling_review.empty:
        ref_cols = [
            "machine_folder", "tool_number",
            "program_description", "reference_description",
            "decimal_size", "match_status", "reference_needs_review",
        ]
        ref_subset = tooling_review[[c for c in ref_cols if c in tooling_review.columns]].copy()

        # Ensure join keys are the same type
        for col in ("tool_number",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in ref_subset.columns:
                ref_subset[col] = pd.to_numeric(ref_subset[col], errors="coerce")

        df = df.merge(
            ref_subset,
            on=["machine_folder", "tool_number"],
            how="left",
            suffixes=("", "_tooling"),
        )

    return df
