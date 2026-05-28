"""
tooling_reference_loader.py — Load machine tooling lists from the shop Excel workbook.

The 'Milling Tools' sheet contains side-by-side tool lists for multiple machines.
This module parses each machine block and returns one record per tool slot.
Output is a reviewable reference layer — not authoritative ground truth.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger

logger = get_logger(__name__)

_TOOLING_SHEET = "Milling Tools"
_SECTION_HEADER_ROW = 1  # 0-indexed row containing machine section labels
_DATA_START_ROW = 3       # 0-indexed row where tool data begins

# Absolute pandas column indices for each machine's (tool#, description, decimal) columns.
# Keyed by the column index where that machine's section header appears.
_MACHINE_COL_BLOCKS: dict[int, tuple[int, int, int]] = {
    1:  (1,  2,  3),   # 655 Haas: tool# at col 1, description at col 2, decimal at col 3
    8:  (8,  9,  10),  # 432 Mazak: range text at col 8 (not a tool#), description at col 9
    12: (12, 13, 14),  # 654 Okuma: tool# at col 12, description at col 13, decimal at col 14
}

# Columns where the tool# cell contains range text instead of a sequential integer.
# Tool numbers for these blocks are inferred from row position.
_RANGE_TEXT_COLS = {8}


def _extract_machine_id(header_text: str) -> str | None:
    """Extract leading 3-4 digit machine number from a section header cell."""
    m = re.match(r"^\s*(\d{3,4})\b", str(header_text).strip())
    return m.group(1) if m else None


def _is_blank(val) -> bool:
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _safe_tool_number(val) -> int | None:
    """Parse a tool-number cell to int. Returns None if not a plain integer."""
    if _is_blank(val):
        return None
    try:
        f = float(val)
        if f == int(f) and f > 0:
            return int(f)
    except (ValueError, TypeError):
        pass
    return None


def _safe_decimal(val) -> float | None:
    if _is_blank(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _normalize_description(val) -> str | None:
    """Strip whitespace from a description cell, return None if empty."""
    if _is_blank(val):
        return None
    s = str(val).strip()
    return s if s else None


def load_tooling_reference(wb_path: Path) -> list[dict]:
    """Load the Milling Tools sheet and return one record per tool slot per machine.

    Each record includes needs_review=True when the tool number was inferred
    (not explicitly listed) or when the description is absent.
    """
    df = pd.read_excel(wb_path, sheet_name=_TOOLING_SHEET, header=None, dtype=object)

    # Discover machine blocks from the section header row
    machines: list[dict] = []
    header_row = df.iloc[_SECTION_HEADER_ROW]
    for hcol, (col_tool, col_desc, col_dec) in _MACHINE_COL_BLOCKS.items():
        cell = header_row.iloc[hcol]
        if _is_blank(cell):
            continue
        machine_id = _extract_machine_id(str(cell))
        if not machine_id:
            continue
        machines.append({
            "machine_id": machine_id,
            "machine_label": str(cell).strip(),
            "col_tool": col_tool,
            "col_desc": col_desc,
            "col_dec": col_dec,
            "inferred_num": col_tool in _RANGE_TEXT_COLS,
        })

    records = []
    data_rows = df.iloc[_DATA_START_ROW:]

    for mach in machines:
        col_tool = mach["col_tool"]
        col_desc = mach["col_desc"]
        col_dec = mach["col_dec"]
        inferred = mach["inferred_num"]

        for row_offset, (_, row) in enumerate(data_rows.iterrows()):
            raw_tool_val = row.iloc[col_tool]
            desc = _normalize_description(row.iloc[col_desc])
            decimal = _safe_decimal(row.iloc[col_dec])

            if inferred:
                # col_tool holds range text ("10.6/11.6"), not a sequential tool#.
                # Skip rows where both range col and description are empty.
                if _is_blank(raw_tool_val) and desc is None:
                    continue
                tool_number_raw = str(raw_tool_val).strip() if not _is_blank(raw_tool_val) else ""
                tool_number = row_offset + 1  # 1-based from data start row
                needs_review = True
            else:
                if _is_blank(raw_tool_val):
                    continue  # no tool slot at this row
                tool_number = _safe_tool_number(raw_tool_val)
                tool_number_raw = str(raw_tool_val).strip()
                needs_review = (tool_number is None or desc is None)

            records.append({
                "machine_id": mach["machine_id"],
                "machine_label": mach["machine_label"],
                "tool_number": tool_number,
                "tool_number_raw": tool_number_raw,
                "description": desc,
                "decimal_size": decimal,
                "needs_review": needs_review,
            })

    logger.info(
        f"Loaded {len(records)} tooling records from "
        f"{len(machines)} machines ({wb_path.name})"
    )
    return records


def export_tooling_reference(
    records: list[dict],
    exports_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Write the tooling reference to CSV and return the path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(records)
    out_path = exports_dir / f"machine_tooling_reference_{timestamp}.csv"
    assert_safe_write(out_path)
    df.to_csv(out_path, index=False)

    logger.info(f"Tooling reference -> {out_path}  ({len(df)} records)")
    return out_path
