"""
proven_sf_lookup.py — Build the app-facing proven S/F lookup table.

Aggregates the full proven_sf_database into a clean, grouped lookup:

    material × machine × tool_type × tool_description
    × operation_intent × s_mode × f_mode

  → S_low / S_mid / S_high  (min / median / max of proven spindle values)
  → F_low / F_mid / F_high  (min / median / max of proven feed values)
  → occurrence_count         (number of S/F records in group)
  → confidence               (best sf_record_confidence in group)

Confidence levels:
    HIGH    — direct job/router material link
    MEDIUM  — router consensus (multiple jobs agree on material)
    LOW     — no verified material (inferred from S/F reference only)

Designed for the dashboard lookup UI:
    - HIGH + MEDIUM shown by default
    - LOW hidden unless user enables
    - No job/part/drawing/router fields in this export

READ-ONLY.  Never writes to P:\\, G:\\, or M:\\.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .tool_classifier import classify_tool_type
from .utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

SF_LOOKUP_COLS: list[str] = [
    "material",
    "machine_folder",
    "tool_type",
    "tool_description",
    "operation_intent",
    "s_mode",
    "S_low",
    "S_mid",
    "S_high",
    "f_mode",
    "F_low",
    "F_mid",
    "F_high",
    "occurrence_count",
    "confidence",
]

# Fields that must NOT appear in the lookup export (job/routing/source details)
_BANNED_COLS = {
    "source_file", "filename", "program_id",
    "matched_job_number", "matched_part_number", "matched_drawing_number",
    "matched_revision", "matched_shared_print_file", "matched_router_file",
    "linked_router_file", "router_work_center", "router_operation_description",
    "link_method", "link_confidence", "link_reason",
    "raw_line", "prev_line", "next_line",
    "review_reason", "extraction_confidence",
}

# Confidence ordering for "best in group" logic
_CONF_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _material(row: dict) -> str:
    """Resolve display material: verified_material → material_candidate_1 → UNKNOWN."""
    vm = str(row.get("verified_material", "") or "").strip()
    if vm and vm != "UNKNOWN":
        return vm
    mc = str(row.get("material_candidate_1", "") or "").strip()
    return mc if mc else "UNKNOWN"


def _best_confidence(series: pd.Series) -> str:
    """Return the highest sf_record_confidence present in the series."""
    vals = set(series.dropna().astype(str).str.upper())
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in vals:
            return level
    return "LOW"


def _norm_mode(val) -> str:
    """Normalise empty / UNKNOWN / NaN mode strings for display."""
    try:
        if pd.isna(val):
            return "UNKNOWN"
    except (TypeError, ValueError):
        pass
    v = str(val or "").strip().upper()
    return "UNKNOWN" if v in ("", "UNKNOWN", "NAN", "NONE") else v


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_sf_lookup(sf_db_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate proven_sf_database into the app-facing lookup table.

    Returns a DataFrame with SF_LOOKUP_COLS columns.
    Excludes any rows with no S value AND no F value.
    """
    if sf_db_df is None or sf_db_df.empty:
        return pd.DataFrame(columns=SF_LOOKUP_COLS)

    df = sf_db_df.copy()

    # Only rows that have at least one S or F value
    has_s = pd.to_numeric(df.get("S", pd.Series(dtype=float)), errors="coerce").notna()
    has_f = pd.to_numeric(df.get("F", pd.Series(dtype=float)), errors="coerce").notna()
    df = df[has_s | has_f].copy()

    if df.empty:
        return pd.DataFrame(columns=SF_LOOKUP_COLS)

    # ── Derive group-key columns ──────────────────────────────────────────

    df["material"] = df.apply(_material, axis=1)

    df["tool_type"] = df.apply(
        lambda r: classify_tool_type(
            str(r.get("resolved_tool_name", "") or ""),
            str(r.get("feed_intent_candidate", "") or ""),
            str(r.get("s_mode", "") or ""),
        ),
        axis=1,
    )

    df["tool_description"] = df["resolved_tool_name"].fillna("").astype(str).str.strip()

    df["operation_intent"] = (
        df.get("feed_intent_candidate", pd.Series(dtype=str))
        .fillna("").astype(str).str.strip()
    )

    # Normalise mode columns before grouping
    df["s_mode_norm"] = df.get("s_mode", pd.Series(dtype=str)).apply(_norm_mode)
    df["f_mode_norm"] = df.get("f_mode", pd.Series(dtype=str)).apply(_norm_mode)

    # Numeric S and F
    df["S_num"] = pd.to_numeric(df.get("S", pd.Series(dtype=float)), errors="coerce")
    df["F_num"] = pd.to_numeric(df.get("F", pd.Series(dtype=float)), errors="coerce")

    group_keys = [
        "material",
        "machine_folder",
        "tool_type",
        "tool_description",
        "operation_intent",
        "s_mode_norm",
        "f_mode_norm",
    ]
    for k in group_keys:
        if k not in df.columns:
            df[k] = ""
        df[k] = df[k].fillna("").astype(str)

    # ── Aggregate ─────────────────────────────────────────────────────────

    rows: list[dict] = []

    for keys, grp in df.groupby(group_keys, dropna=False, sort=False):
        s_data = grp["S_num"].dropna()
        f_data = grp["F_num"].dropna()

        conf_col = "sf_record_confidence" if "sf_record_confidence" in grp.columns else None
        confidence = _best_confidence(grp[conf_col]) if conf_col else "LOW"

        row = dict(zip(group_keys, keys))
        row.update({
            "S_low":  round(s_data.min(),    0) if len(s_data) else None,
            "S_mid":  round(s_data.median(), 0) if len(s_data) else None,
            "S_high": round(s_data.max(),    0) if len(s_data) else None,
            "F_low":  round(f_data.min(),    5) if len(f_data) else None,
            "F_mid":  round(f_data.median(), 5) if len(f_data) else None,
            "F_high": round(f_data.max(),    5) if len(f_data) else None,
            "occurrence_count": len(grp),
            "confidence": confidence,
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=SF_LOOKUP_COLS)

    result = pd.DataFrame(rows)

    # Rename normalised mode columns to display names
    result.rename(columns={"s_mode_norm": "s_mode", "f_mode_norm": "f_mode"}, inplace=True)

    # Sort: material alpha, then by occurrence_count desc
    result.sort_values(
        ["material", "occurrence_count"],
        ascending=[True, False],
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)

    # Return only the defined columns (in order), adding any missing as None
    for col in SF_LOOKUP_COLS:
        if col not in result.columns:
            result[col] = None

    return result[SF_LOOKUP_COLS]


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

def export_sf_lookup(df: pd.DataFrame, exports_dir: Path, timestamp: str) -> Path:
    """Write the lookup table to exports/proven_sf_lookup_TIMESTAMP.csv."""
    out = exports_dir / f"proven_sf_lookup_{timestamp}.csv"
    assert_safe_write(out)
    (pd.DataFrame(columns=SF_LOOKUP_COLS) if df is None or df.empty else df).to_csv(
        out, index=False
    )
    logger.info(f"SF lookup      -> {out}  ({0 if df is None or df.empty else len(df)} row(s))")
    return out
