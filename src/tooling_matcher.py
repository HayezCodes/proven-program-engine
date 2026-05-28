"""
tooling_matcher.py — Compare parsed CNC tool usage against the shop tooling reference.

Produces a user-editable review CSV. The reference is treated as a suggestion layer,
not authoritative ground truth. All match determinations are deterministic and explicit.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger

logger = get_logger(__name__)

# Columns the user fills in during review (always blank in the generated output)
_REVIEW_COLS = ["review_action", "corrected_description", "notes"]

_REVIEW_OUTPUT_COLS = [
    "machine_folder",
    "tool_number",
    "active_t_code",
    "program_description",
    "reference_description",
    "decimal_size",
    "match_status",
    "reference_needs_review",
    "review_action",
    "corrected_description",
    "notes",
]


def _extract_machine_id(folder_name) -> str | None:
    """Extract leading 3-4 digit machine ID from a machine_folder string."""
    if not folder_name or (isinstance(folder_name, float)):
        return None
    m = re.match(r"^\s*(\d{3,4})\b", str(folder_name).strip())
    return m.group(1) if m else None


def _most_common(values) -> str | None:
    """Return the most frequently occurring non-null value from an iterable."""
    counts: dict[str, int] = {}
    for v in values:
        if v is not None and str(v).strip():
            key = str(v).strip()
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def _determine_match_status(
    program_description: str | None,
    ref_entry: dict | None,
    machine_known: bool,
) -> str:
    if not machine_known:
        return "no_reference_data"
    if ref_entry is None:
        return "missing_from_reference"
    if ref_entry.get("needs_review"):
        return "needs_review"
    ref_desc = ref_entry.get("description")
    if not ref_desc:
        return "no_description_in_reference"
    if not program_description:
        return "no_program_description"
    if program_description.upper().strip() == ref_desc.upper().strip():
        return "description_match"
    return "description_differs"


def match_tool_usage(
    cuts_df: pd.DataFrame,
    reference: list[dict],
) -> pd.DataFrame:
    """Compare each unique (machine_folder, tool_number) in cuts_df against the reference.

    Returns a DataFrame with one row per unique tool-machine combination plus
    blank user-editable columns for review.
    """
    if cuts_df.empty:
        return pd.DataFrame(columns=_REVIEW_OUTPUT_COLS)

    # Build reference index keyed by (machine_id, tool_number)
    ref_index: dict[tuple[str, int | None], dict] = {}
    for rec in reference:
        key = (rec["machine_id"], rec["tool_number"])
        ref_index[key] = rec
    known_machines = {rec["machine_id"] for rec in reference}

    # Ensure required columns exist
    for col in ("tool_number", "active_t_code", "machine_folder"):
        if col not in cuts_df.columns:
            cuts_df = cuts_df.copy()
            cuts_df[col] = None

    has_desc = "tool_description" in cuts_df.columns

    # Group by (machine_folder, tool_number)
    group_cols = ["machine_folder", "tool_number"]
    rows = []
    for group_key, group_df in cuts_df.groupby(group_cols, dropna=False):
        machine_folder, tool_number = group_key

        machine_id = _extract_machine_id(machine_folder)
        machine_known = machine_id in known_machines

        # Representative t_code: most common active_t_code in group
        active_t_code = _most_common(group_df["active_t_code"])

        # Representative program description (if available in cuts CSV)
        program_description: str | None = None
        if has_desc:
            program_description = _most_common(group_df["tool_description"])

        # Look up reference entry
        tool_num_int: int | None = None
        try:
            tool_num_int = int(tool_number) if tool_number is not None else None
        except (ValueError, TypeError):
            pass

        ref_entry = ref_index.get((machine_id, tool_num_int)) if machine_id else None

        match_status = _determine_match_status(program_description, ref_entry, machine_known)

        rows.append({
            "machine_folder": machine_folder,
            "tool_number": tool_number,
            "active_t_code": active_t_code,
            "program_description": program_description,
            "reference_description": ref_entry.get("description") if ref_entry else None,
            "decimal_size": ref_entry.get("decimal_size") if ref_entry else None,
            "match_status": match_status,
            "reference_needs_review": ref_entry.get("needs_review", False) if ref_entry else False,
            "review_action": "",
            "corrected_description": "",
            "notes": "",
        })

    result = pd.DataFrame(rows, columns=_REVIEW_OUTPUT_COLS)
    result = result.sort_values(
        ["machine_folder", "tool_number"], na_position="last"
    ).reset_index(drop=True)

    logger.info(
        f"Tooling review: {len(result)} unique tool-machine combinations "
        f"({sum(result['match_status'] == 'description_match')} matched, "
        f"{sum(result['match_status'] == 'missing_from_reference')} missing from reference)"
    )
    return result


def export_tooling_review(
    cuts_df: pd.DataFrame,
    reference: list[dict],
    exports_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Run tooling match and write tooling_review_*.csv."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = match_tool_usage(cuts_df, reference)
    out_path = exports_dir / f"tooling_review_{timestamp}.csv"
    assert_safe_write(out_path)
    df.to_csv(out_path, index=False)

    logger.info(f"Tooling review  -> {out_path}  ({len(df)} rows)")
    return out_path
