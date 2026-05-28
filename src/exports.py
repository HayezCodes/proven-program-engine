"""
exports.py — Aggregate metrics and summary exports for Proven Program Engine.

All functions are pure aggregations over already-extracted records.
No CNC files are read here. No values are inferred.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parser summary (JSON)
# ---------------------------------------------------------------------------

def build_parser_summary(records: list[dict]) -> dict:
    """Compute aggregate metrics over a list of parsed records.

    Returns a plain dict suitable for JSON serialization.
    """
    total = len(records)
    if total == 0:
        return {"total_records": 0}

    s_records = [r for r in records if r.get("s_value") is not None]
    f_records = [r for r in records if r.get("f_value") is not None]
    orphan = [r for r in records if not r.get("active_t_code")]
    missing_mode = [r for r in s_records if r.get("s_mode") in ("UNKNOWN", "")]

    unique_tools = len({r["tool_number"] for r in records if r.get("tool_number")})
    unique_programs = len({r["program_id"] for r in records if r.get("program_id") is not None})
    unique_t_codes = len({r["active_t_code"] for r in records if r.get("active_t_code")})

    # Unique (tool, s_value, f_value) combos
    sf_combos = {
        (r.get("tool_number"), r.get("s_value"), r.get("f_value"))
        for r in records
        if r.get("s_value") is not None and r.get("f_value") is not None
    }

    spindle_limit_count = sum(
        1 for r in records if r.get("s_type") == "LIMIT"
    )
    block_skip_count = sum(1 for r in records if r.get("block_skip"))

    # Distribution helpers
    def _dist(records_subset: list[dict], field: str) -> dict:
        counter: dict[str, int] = {}
        for r in records_subset:
            val = str(r.get(field, ""))
            counter[val] = counter.get(val, 0) + 1
        return dict(sorted(counter.items(), key=lambda kv: -kv[1]))

    # Top tools by occurrence
    tool_counts: dict[str, int] = {}
    for r in records:
        t = r.get("active_t_code", "")
        if t:
            tool_counts[t] = tool_counts.get(t, 0) + 1
    top_tools = sorted(tool_counts.items(), key=lambda kv: -kv[1])[:20]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_records": total,
        "files_parsed": unique_programs,
        "unique_t_codes": unique_t_codes,
        "unique_tool_numbers": unique_tools,
        "total_s_values": len(s_records),
        "total_f_values": len(f_records),
        "records_missing_t_code": len(orphan),
        "records_missing_spindle_mode": len(missing_mode),
        "spindle_limit_records": spindle_limit_count,
        "block_skip_records": block_skip_count,
        "unique_sf_combos": len(sf_combos),
        "confidence_distribution": _dist(records, "extraction_confidence"),
        "s_mode_distribution": _dist(s_records, "s_mode"),
        "f_mode_distribution": _dist(f_records, "f_mode"),
        "s_type_distribution": _dist(s_records, "s_type"),
        "top_tools_by_record_count": [
            {"t_code": t, "record_count": c} for t, c in top_tools
        ],
    }


def export_parser_summary(
    records: list[dict],
    exports_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Write parser summary JSON and return its path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = build_parser_summary(records)
    out_path = exports_dir / f"parser_summary_{timestamp}.json"

    assert_safe_write(out_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        f"Parser summary -> {out_path}  "
        f"({summary['total_records']} records, "
        f"{summary['unique_t_codes']} T codes, "
        f"{summary['unique_sf_combos']} unique S+F combos)"
    )
    return out_path


# ---------------------------------------------------------------------------
# Tool summary (CSV)
# ---------------------------------------------------------------------------

def build_tool_summary(records: list[dict]) -> pd.DataFrame:
    """Aggregate speed/feed stats grouped by tool_number, machine_folder, s_mode.

    Returns a DataFrame with one row per group. Groups with no S or F data
    are excluded (requires at least one non-null s_value or f_value in group).
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Keep only records with at least one value and an active tool
    df = df[df["active_t_code"].notna() & (df["active_t_code"] != "")]
    df = df[df["s_value"].notna() | df["f_value"].notna()]

    if df.empty:
        return pd.DataFrame()

    group_cols = ["tool_number", "active_t_code", "machine_folder", "s_mode"]

    # Ensure the columns exist even if empty
    for col in group_cols + ["s_value", "f_value", "program_id"]:
        if col not in df.columns:
            df[col] = None

    # Replace empty string s_mode with "UNKNOWN" for cleaner grouping
    df["s_mode"] = df["s_mode"].replace("", "UNKNOWN").fillna("UNKNOWN")

    grouped = df.groupby(group_cols, dropna=False)

    rows = []
    for group_key, group_df in grouped:
        tool_num, t_code, machine, mode = group_key

        s_data = group_df["s_value"].dropna()
        f_data = group_df["f_value"].dropna()

        rows.append({
            "tool_number": tool_num,
            "active_t_code": t_code,
            "machine_folder": machine,
            "s_mode": mode,
            "s_count": len(s_data),
            "s_mean": round(s_data.mean(), 2) if len(s_data) else None,
            "s_min": s_data.min() if len(s_data) else None,
            "s_max": s_data.max() if len(s_data) else None,
            "f_count": len(f_data),
            "f_mean": round(f_data.mean(), 6) if len(f_data) else None,
            "f_min": f_data.min() if len(f_data) else None,
            "f_max": f_data.max() if len(f_data) else None,
            "record_count": len(group_df),
            "unique_program_count": group_df["program_id"].nunique(),
        })

    result = pd.DataFrame(rows)
    # Sort by occurrences descending for easy review
    if not result.empty:
        result = result.sort_values("record_count", ascending=False).reset_index(drop=True)
    return result


def export_tool_summary(
    records: list[dict],
    exports_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Write tool summary CSV and return its path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = build_tool_summary(records)
    out_path = exports_dir / f"tool_summary_{timestamp}.csv"
    assert_safe_write(out_path)
    df.to_csv(out_path, index=False)

    logger.info(f"Tool summary   -> {out_path}  ({len(df)} groups)")
    return out_path
