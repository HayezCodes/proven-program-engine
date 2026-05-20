"""
tables.py — Reusable dataframe display and download helpers.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st


def show_table(
    df: pd.DataFrame,
    height: int = 400,
    column_config: Optional[dict] = None,
    hide_index: bool = True,
    use_container_width: bool = True,
) -> None:
    """Render a styled read-only dataframe."""
    if df is None or df.empty:
        st.info("No data to display.")
        return
    st.dataframe(
        df,
        height=height,
        column_config=column_config or {},
        hide_index=hide_index,
        use_container_width=use_container_width,
    )


def download_csv(df: pd.DataFrame, filename: str, label: str = "Export CSV") -> None:
    """Render a download button for a DataFrame as CSV."""
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def row_count_caption(df: pd.DataFrame, noun: str = "row") -> None:
    """Show a small caption with the filtered row count."""
    if df is None:
        return
    n = len(df)
    st.caption(f"{n:,} {noun}{'s' if n != 1 else ''}")
