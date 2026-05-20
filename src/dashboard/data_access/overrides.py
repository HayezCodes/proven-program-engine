"""
overrides.py — Load and save local correction overrides.

Overrides live in data/overrides/ and are never written to exports/ or P:\\.
They act as a correction layer displayed on top of inferred data.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_OVERRIDES_DIR = Path(__file__).parents[3] / "data" / "overrides"

_TOOLING_OVERRIDE_PATH = _OVERRIDES_DIR / "tooling_overrides.csv"
_MATERIAL_OVERRIDE_PATH = _OVERRIDES_DIR / "material_overrides.csv"

_TOOLING_COLS = [
    "machine_folder", "tool_number",
    "review_action", "corrected_description", "notes",
    "override_timestamp",
]

_MATERIAL_COLS = [
    "machine_folder", "active_t_code", "tool_number",
    "review_decision", "reviewer_note",
    "override_timestamp",
]


def load_tooling_overrides(path: Path = _TOOLING_OVERRIDE_PATH) -> pd.DataFrame:
    """Load existing tooling override records. Returns empty DataFrame if none exist."""
    if path.exists():
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            pass
    return pd.DataFrame(columns=_TOOLING_COLS)


def save_tooling_overrides(
    df: pd.DataFrame,
    path: Path = _TOOLING_OVERRIDE_PATH,
) -> None:
    """Write the complete tooling overrides DataFrame to CSV (full replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def upsert_tooling_override(
    machine_folder: str,
    tool_number,
    review_action: str,
    corrected_description: str,
    notes: str,
    path: Path = _TOOLING_OVERRIDE_PATH,
) -> None:
    """Insert or update a single tooling override keyed on (machine_folder, tool_number)."""
    df = load_tooling_overrides(path)

    key = {"machine_folder": str(machine_folder), "tool_number": str(tool_number)}
    new_row = {
        **key,
        "review_action": review_action,
        "corrected_description": corrected_description,
        "notes": notes,
        "override_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    mask = (df["machine_folder"].astype(str) == key["machine_folder"]) & (
        df["tool_number"].astype(str) == key["tool_number"]
    )

    if mask.any():
        for col, val in new_row.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    save_tooling_overrides(df, path)


def apply_tooling_overrides(
    review_df: pd.DataFrame,
    overrides: Optional[pd.DataFrame] = None,
    overrides_path: Path = _TOOLING_OVERRIDE_PATH,
) -> pd.DataFrame:
    """Merge overrides into the review DataFrame, marking overridden rows."""
    if review_df is None or review_df.empty:
        return review_df

    if overrides is None:
        overrides = load_tooling_overrides(overrides_path)

    if overrides.empty:
        review_df = review_df.copy()
        review_df["is_overridden"] = pd.array([False] * len(review_df), dtype=object)
        return review_df

    df = review_df.copy()
    df["tool_number_str"] = df["tool_number"].astype(str)
    df["machine_folder_str"] = df["machine_folder"].astype(str)

    ov = overrides.copy()
    ov = ov.rename(columns={
        "review_action": "ov_review_action",
        "corrected_description": "ov_corrected_description",
        "notes": "ov_notes",
    })
    ov["tool_number_str"] = ov["tool_number"].astype(str)
    ov["machine_folder_str"] = ov["machine_folder"].astype(str)
    ov = ov.drop(columns=["machine_folder", "tool_number", "override_timestamp"], errors="ignore")

    df = df.merge(
        ov[["machine_folder_str", "tool_number_str",
            "ov_review_action", "ov_corrected_description", "ov_notes"]],
        on=["machine_folder_str", "tool_number_str"],
        how="left",
    )

    # Apply override values where they exist
    has_override = df["ov_review_action"].notna() & (df["ov_review_action"] != "")
    df["is_overridden"] = has_override.astype(object)

    if "review_action" in df.columns:
        df.loc[has_override, "review_action"] = df.loc[has_override, "ov_review_action"]
    if "corrected_description" in df.columns:
        df.loc[has_override, "corrected_description"] = df.loc[
            has_override, "ov_corrected_description"
        ]
    if "notes" in df.columns:
        df.loc[has_override, "notes"] = df.loc[has_override, "ov_notes"]

    df = df.drop(
        columns=["machine_folder_str", "tool_number_str",
                 "ov_review_action", "ov_corrected_description", "ov_notes"],
        errors="ignore",
    )
    return df


def load_material_overrides(path: Path = _MATERIAL_OVERRIDE_PATH) -> pd.DataFrame:
    """Load existing material override records. Returns empty DataFrame if none exist."""
    if path.exists():
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            pass
    return pd.DataFrame(columns=_MATERIAL_COLS)


def upsert_material_override(
    machine_folder: str,
    active_t_code: str,
    tool_number,
    review_decision: str,
    reviewer_note: str,
    path: Path = _MATERIAL_OVERRIDE_PATH,
) -> None:
    """Insert or update a single material override keyed on (machine_folder, active_t_code)."""
    df = load_material_overrides(path)

    key = {
        "machine_folder": str(machine_folder),
        "active_t_code": str(active_t_code),
        "tool_number": str(tool_number),
    }
    new_row = {
        **key,
        "review_decision": review_decision,
        "reviewer_note": reviewer_note,
        "override_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    mask = (
        (df["machine_folder"].astype(str) == key["machine_folder"])
        & (df["active_t_code"].astype(str) == key["active_t_code"])
    )

    if mask.any():
        for col, val in new_row.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def apply_material_overrides(
    candidates_df: pd.DataFrame,
    overrides: Optional[pd.DataFrame] = None,
    overrides_path: Path = _MATERIAL_OVERRIDE_PATH,
) -> pd.DataFrame:
    """Merge material overrides into the candidates DataFrame."""
    if candidates_df is None or candidates_df.empty:
        return candidates_df

    if overrides is None:
        overrides = load_material_overrides(overrides_path)

    df = candidates_df.copy()
    df["review_decision"] = "pending"
    df["reviewer_note"] = ""

    if overrides.empty:
        return df

    ov = overrides.copy()
    for col in ("machine_folder", "active_t_code"):
        if col in ov.columns:
            ov[col] = ov[col].astype(str)
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = df.merge(
        ov[["machine_folder", "active_t_code", "review_decision", "reviewer_note"]],
        on=["machine_folder", "active_t_code"],
        how="left",
        suffixes=("_orig", ""),
    )

    # Drop originals if suffixed
    for col in ("review_decision_orig", "reviewer_note_orig"):
        if col in df.columns:
            df = df.drop(columns=[col])

    df["review_decision"] = df["review_decision"].fillna("pending")
    df["reviewer_note"] = df["reviewer_note"].fillna("")
    return df
