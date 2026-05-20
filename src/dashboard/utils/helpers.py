"""
helpers.py — Pure utility functions for dashboard data prep and filtering.

No Streamlit imports. Fully testable.
"""

from typing import Any, Optional
import pandas as pd


def safe_list(df: pd.DataFrame, col: str, sort: bool = True) -> list:
    """Return unique non-null values from a DataFrame column."""
    if df is None or df.empty or col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    vals = [v for v in vals if str(v).strip() not in ("", "nan", "None")]
    return sorted(vals, key=lambda x: str(x)) if sort else vals


def filter_df(df: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    """Apply equality filters to a DataFrame.

    Accepts keyword arguments where each key is a column name and value is
    either a single value or a list of values to keep. None / empty list
    means "no filter applied" for that column.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    result = df.copy()
    for col, val in filters.items():
        if col not in result.columns:
            continue
        if val is None:
            continue
        if isinstance(val, list):
            if not val:
                continue
            result = result[result[col].isin(val)]
        else:
            result = result[result[col] == val]
    return result


def text_search(df: pd.DataFrame, query: str, columns: list[str]) -> pd.DataFrame:
    """Return rows where any of the given columns contain the query string (case-insensitive)."""
    if df is None or df.empty or not query or not query.strip():
        return df if df is not None else pd.DataFrame()

    q = query.strip().lower()
    cols_present = [c for c in columns if c in df.columns]
    if not cols_present:
        return df

    mask = pd.Series(False, index=df.index)
    for col in cols_present:
        mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
    return df[mask]


def fmt_sfm(val) -> str:
    """Format an SFM value as a rounded integer string with units."""
    try:
        return f"{int(round(float(val)))} SFM"
    except (ValueError, TypeError):
        return "—"


def fmt_feed_ipr(val) -> str:
    """Format an IPR feed value."""
    try:
        return f"{float(val):.4f} IPR"
    except (ValueError, TypeError):
        return "—"


def fmt_feed_ipm(val) -> str:
    """Format an IPM feed value."""
    try:
        return f"{float(val):.2f} IPM"
    except (ValueError, TypeError):
        return "—"


def fmt_range(lo, hi, unit: str = "") -> str:
    """Format a min/max range string."""
    try:
        lo_s = f"{float(lo):.1f}"
        hi_s = f"{float(hi):.1f}"
        return f"{lo_s} – {hi_s} {unit}".strip()
    except (ValueError, TypeError):
        return "—"


def top_n(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    """Return the top-N rows sorted by a numeric column descending."""
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    return df.nlargest(n, col)


def count_by(df: pd.DataFrame, group_col: str, count_col: str = None) -> pd.DataFrame:
    """Return a value_counts-style DataFrame grouped by group_col."""
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "count"])
    counts = df[group_col].value_counts().reset_index()
    counts.columns = [group_col, "count"]
    return counts


def summarize_sf(df: pd.DataFrame) -> dict:
    """Compute quick S/F summary stats from a material_candidates or tool_summary DataFrame."""
    if df is None or df.empty:
        return {}
    out: dict = {}
    for col in ("s_mean", "s_min", "s_max", "f_mean", "f_min", "f_max"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            if not numeric.empty:
                out[col] = round(numeric.median(), 2)
    return out
