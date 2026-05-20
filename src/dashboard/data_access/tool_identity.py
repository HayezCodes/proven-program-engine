"""
tool_identity.py — Resolve tool name/description from multiple sources.

Priority (highest to lowest):
  1. OVERRIDE        — manual corrected_description from overrides CSV
  2. TOOLDB_ASSEMBLY — TlAssembly join record (assembly_join extraction_method)
  3. TOOLDB_LATHE_FALLBACK — lathe fallback record (always needs_review=True)
  4. PROGRAM         — parsed tool comment from CNC program
  5. EXCEL           — Excel tooling reference description
  6. UNKNOWN         — no usable name found

READ ONLY against all source data.
"""

import re
from typing import Optional

import pandas as pd

_MACH_NUM_RE = re.compile(r'\b(\d{3,4})\b')

SOURCE_OVERRIDE = "OVERRIDE"
SOURCE_TOOLDB_ASSEMBLY = "TOOLDB_ASSEMBLY"
SOURCE_TOOLDB_FALLBACK = "TOOLDB_LATHE_FALLBACK"
SOURCE_PROGRAM = "PROGRAM"
SOURCE_EXCEL = "EXCEL"
SOURCE_UNKNOWN = "UNKNOWN"

ALL_SOURCES = [
    SOURCE_OVERRIDE,
    SOURCE_TOOLDB_ASSEMBLY,
    SOURCE_TOOLDB_FALLBACK,
    SOURCE_PROGRAM,
    SOURCE_EXCEL,
    SOURCE_UNKNOWN,
]

_RESOLVED_COLS = [
    "resolved_tool_name",
    "resolved_tool_description",
    "resolved_tool_source",
    "resolved_tool_confidence",
    "resolved_tool_needs_review",
]

_UNKNOWN_RESULT = {
    "resolved_tool_name": "UNKNOWN TOOL",
    "resolved_tool_description": "",
    "resolved_tool_source": SOURCE_UNKNOWN,
    "resolved_tool_confidence": "LOW",
    "resolved_tool_needs_review": True,
}


def _normalize_machine_id(val) -> str:
    """Extract first 3–4 digit number from a machine folder name or ID string."""
    if val is None:
        return ""
    if isinstance(val, float):
        if pd.isna(val):
            return ""
        val = str(int(val))
    s = str(val).strip()
    m = _MACH_NUM_RE.search(s)
    return m.group(1) if m else s


def _to_tool_str(val) -> str:
    """Normalize a tool number to a plain integer string (e.g. 1.0 → '1')."""
    if val is None:
        return ""
    if isinstance(val, float):
        if pd.isna(val):
            return ""
        return str(int(val))
    try:
        return str(int(float(str(val))))
    except (ValueError, TypeError):
        return str(val).strip()


def _is_usable(val) -> bool:
    """Return True if val is a non-empty, non-None, non-nan string."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() not in ("nan", "none")


def resolve_tool_identity(
    *,
    parsed_description: str = "",
    reference_description: str = "",
    tooldb_records: Optional[list[dict]] = None,
    override: Optional[dict] = None,
) -> dict:
    """
    Resolve tool identity from available sources in priority order.

    Parameters
    ----------
    parsed_description : str
        Tool comment parsed from the CNC program (program_description column).
    reference_description : str
        Description from the Excel tooling reference (reference_description column).
    tooldb_records : list[dict] or None
        All TOOLDB records matching (machine_id, tool_number). May contain
        both assembly_join and lathe_fallback records.
    override : dict or None
        Manual override with at least a 'corrected_description' key.

    Returns
    -------
    dict with keys: resolved_tool_name, resolved_tool_description,
    resolved_tool_source, resolved_tool_confidence, resolved_tool_needs_review.
    """
    # 1. Manual override
    if override:
        name = (override.get("corrected_description") or "").strip()
        if name and name.lower() not in ("nan", "none"):
            return {
                "resolved_tool_name": name,
                "resolved_tool_description": (override.get("notes") or "").strip(),
                "resolved_tool_source": SOURCE_OVERRIDE,
                "resolved_tool_confidence": "HIGH",
                "resolved_tool_needs_review": False,
            }

    # 2. TOOLDB assembly_join (highest-confidence TOOLDB source)
    if tooldb_records:
        for rec in tooldb_records:
            if rec.get("extraction_method") == "assembly_join":
                name = (rec.get("tool_name") or "").strip()
                if name and name.lower() not in ("nan", "none"):
                    return {
                        "resolved_tool_name": name,
                        "resolved_tool_description": (rec.get("holder_name") or "").strip(),
                        "resolved_tool_source": SOURCE_TOOLDB_ASSEMBLY,
                        "resolved_tool_confidence": "HIGH",
                        "resolved_tool_needs_review": False,
                    }

    # 3. TOOLDB lathe_fallback (needs review — not confirmed by assembly record)
    if tooldb_records:
        for rec in tooldb_records:
            if rec.get("extraction_method") == "lathe_fallback":
                name = (rec.get("tool_name") or "").strip()
                if name and name.lower() not in ("nan", "none"):
                    return {
                        "resolved_tool_name": name,
                        "resolved_tool_description": "",
                        "resolved_tool_source": SOURCE_TOOLDB_FALLBACK,
                        "resolved_tool_confidence": "MEDIUM",
                        "resolved_tool_needs_review": True,
                    }

    # 4. Program parsed description
    if _is_usable(parsed_description):
        return {
            "resolved_tool_name": parsed_description.strip(),
            "resolved_tool_description": "",
            "resolved_tool_source": SOURCE_PROGRAM,
            "resolved_tool_confidence": "MEDIUM",
            "resolved_tool_needs_review": False,
        }

    # 5. Excel reference description
    if _is_usable(reference_description):
        return {
            "resolved_tool_name": reference_description.strip(),
            "resolved_tool_description": "",
            "resolved_tool_source": SOURCE_EXCEL,
            "resolved_tool_confidence": "MEDIUM",
            "resolved_tool_needs_review": False,
        }

    # 6. Unknown
    return dict(_UNKNOWN_RESULT)


# ---------------------------------------------------------------------------
# DataFrame-level resolution
# ---------------------------------------------------------------------------

def _build_tooldb_lookup(tooldb_ref: Optional[pd.DataFrame]) -> dict:
    """Build {(machine_id_norm, tool_number_str): [records]} from tooldb_reference."""
    if tooldb_ref is None or tooldb_ref.empty:
        return {}
    lookup: dict[tuple, list] = {}
    for _, row in tooldb_ref.iterrows():
        mid = _normalize_machine_id(row.get("machine_id"))
        tn = _to_tool_str(row.get("tool_number"))
        if mid and tn:
            lookup.setdefault((mid, tn), []).append(dict(row))
    return lookup


def _build_override_lookup(overrides: Optional[pd.DataFrame]) -> dict:
    """Build {(machine_folder_str, tool_number_str): record} from overrides."""
    if overrides is None or overrides.empty:
        return {}
    lookup: dict[tuple, dict] = {}
    for _, row in overrides.iterrows():
        mf = str(row.get("machine_folder", "")).strip()
        tn = _to_tool_str(row.get("tool_number"))
        if mf and tn:
            lookup[(mf, tn)] = dict(row)
    return lookup


def resolve_tool_identity_df(
    df: pd.DataFrame,
    tooldb_ref: Optional[pd.DataFrame] = None,
    overrides: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add resolved_tool_* columns to a DataFrame that has machine_folder and tool_number.

    Reads from row columns (if present):
      program_description  — parsed CNC tool comment (PROGRAM source)
      reference_description — Excel tooling reference (EXCEL source)
      corrected_description — already-merged override (OVERRIDE source)

    Also accepts external lookups via `tooldb_ref` and `overrides` parameters
    for use before apply_tooling_overrides has been called.

    Returns a copy of df with five new columns added.
    """
    if df is None or df.empty:
        return df

    tooldb_lookup = _build_tooldb_lookup(tooldb_ref)
    override_lookup = _build_override_lookup(overrides)

    def _resolve_row(row: pd.Series) -> dict:
        mf = str(row.get("machine_folder", "") or "").strip()
        tn = _to_tool_str(row.get("tool_number"))
        mid_norm = _normalize_machine_id(mf)

        # Override: prefer already-merged corrected_description in the row,
        # then fall back to the external override lookup.
        override = None
        cd = str(row.get("corrected_description", "") or "").strip()
        if _is_usable(cd):
            override = {
                "corrected_description": cd,
                "notes": str(row.get("notes", "") or "").strip(),
            }
        elif mf and tn:
            override = override_lookup.get((mf, tn))

        tooldb_records = tooldb_lookup.get((mid_norm, tn)) if (mid_norm and tn) else None

        # program_description may come under two column names after df merges
        parsed_desc = ""
        for col in ("program_description", "program_description_tooling"):
            v = str(row.get(col, "") or "").strip()
            if _is_usable(v):
                parsed_desc = v
                break

        ref_desc = str(row.get("reference_description", "") or "").strip()

        return resolve_tool_identity(
            parsed_description=parsed_desc,
            reference_description=ref_desc,
            tooldb_records=tooldb_records,
            override=override,
        )

    resolved = df.apply(_resolve_row, axis=1, result_type="expand")
    result = df.copy()
    for col in resolved.columns:
        result[col] = resolved[col]

    if "resolved_tool_needs_review" in result.columns:
        result["resolved_tool_needs_review"] = result["resolved_tool_needs_review"].astype(object)

    return result
