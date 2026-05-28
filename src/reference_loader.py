"""
reference_loader.py — Load and normalize the shop S/F reference table from Excel.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger

logger = get_logger(__name__)

_WB_SHEET = "SPEEDS AND FEEDS "  # trailing space is intentional — do not remove

# Row positions in the sheet (0-indexed)
_DATA_START_ROW = 3   # rows 0-2 are title/section/header; data starts at row 3
_DATA_NROWS = 11

# Column positions (0-indexed); col 0 is empty, col 1 is ref number
_COL_MATERIAL = 2
_COL_TURNING_ROUGH_SFM = 3
_COL_TURNING_FINISH_SFM = 4
_COL_TURNING_ROUGH_DOC = 5
_COL_TURNING_ROUGH_IPR = 6
_COL_TURNING_FINISH_IPR = 7
_COL_MILLING_SFM = 9
_COL_MILLING_ROUGH_IPM = 10
_COL_MILLING_FINISH_IPM = 11


def _normalize_material(raw: str) -> str:
    """Strip whitespace, collapse internal spaces, uppercase."""
    return re.sub(r"\s+", " ", str(raw).strip()).upper()


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_finish_feed(raw_val) -> dict:
    """Parse a finish-feed cell into up to three intent bands.

    One value  → finish_feed_mid only (diameter-to-size feed).
    Two values → finish_feed_low (lowest) + finish_feed_high (highest), mid=None.
    Three+     → finish_feed_low, finish_feed_mid (median), finish_feed_high.

    Always preserves finish_feed_raw as the original cell text.
    """
    _empty = {
        "finish_feed_raw": "",
        "finish_feed_low": None,
        "finish_feed_mid": None,
        "finish_feed_high": None,
    }

    if raw_val is None:
        return _empty

    try:
        if pd.isna(raw_val):
            return _empty
    except (TypeError, ValueError):
        pass

    raw_str = str(raw_val).strip()

    # Extract all decimal numbers present in the cell text
    nums = sorted([float(m) for m in re.findall(r"\d+\.\d+", raw_str)])

    if not nums:
        return {**_empty, "finish_feed_raw": raw_str}

    if len(nums) == 1:
        return {
            "finish_feed_raw": raw_str,
            "finish_feed_low": None,
            "finish_feed_mid": nums[0],
            "finish_feed_high": None,
        }
    if len(nums) == 2:
        return {
            "finish_feed_raw": raw_str,
            "finish_feed_low": nums[0],
            "finish_feed_mid": None,
            "finish_feed_high": nums[1],
        }
    # Three or more: take lowest, middle, highest
    return {
        "finish_feed_raw": raw_str,
        "finish_feed_low": nums[0],
        "finish_feed_mid": nums[len(nums) // 2],
        "finish_feed_high": nums[-1],
    }


def load_reference(wb_path: Path) -> list[dict]:
    """Load and parse the SPEEDS AND FEEDS reference table from the Excel workbook.

    Returns a list of dicts, one per material row, with normalized fields.
    """
    # dtype=object on the finish-feed column preserves raw text for multi-value cells
    raw = pd.read_excel(
        wb_path, sheet_name=_WB_SHEET, header=None,
        dtype={_COL_TURNING_FINISH_IPR: object},
    )
    data_rows = raw.iloc[_DATA_START_ROW: _DATA_START_ROW + _DATA_NROWS]

    materials = []
    for _, row in data_rows.iterrows():
        raw_name = row.iloc[_COL_MATERIAL]
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            continue

        finish_feed = _parse_finish_feed(row.iloc[_COL_TURNING_FINISH_IPR])

        materials.append({
            "material_raw": str(raw_name).strip(),
            "material": _normalize_material(str(raw_name)),
            "turning_rough_sfm": _safe_float(row.iloc[_COL_TURNING_ROUGH_SFM]),
            "turning_finish_sfm": _safe_float(row.iloc[_COL_TURNING_FINISH_SFM]),
            "turning_rough_doc": _safe_float(row.iloc[_COL_TURNING_ROUGH_DOC]),
            "turning_rough_ipr": _safe_float(row.iloc[_COL_TURNING_ROUGH_IPR]),
            "turning_finish_ipr": _safe_float(row.iloc[_COL_TURNING_FINISH_IPR]),  # backward compat
            **finish_feed,
            "milling_sfm": _safe_float(row.iloc[_COL_MILLING_SFM]),
            "milling_rough_ipm": _safe_float(row.iloc[_COL_MILLING_ROUGH_IPM]),
            "milling_finish_ipm": _safe_float(row.iloc[_COL_MILLING_FINISH_IPM]),
        })

    logger.info(f"Loaded {len(materials)} materials from reference table ({wb_path.name})")
    return materials


def export_reference(
    reference: list[dict],
    exports_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Write the normalized reference table to CSV and return the path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(reference)
    out_path = exports_dir / f"shop_sf_reference_{timestamp}.csv"
    assert_safe_write(out_path)
    df.to_csv(out_path, index=False)

    logger.info(f"Reference table -> {out_path}  ({len(df)} materials)")
    return out_path
