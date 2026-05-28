"""
1_Proven_SF.py — Programmer-focused proven speeds and feeds view.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.components.tables import download_csv, row_count_caption, show_table
from src.dashboard.data_access.loader import (
    load_latest_proven_sf_database,
    load_proven_sf_dashboard_df,
)
from src.dashboard.styling.theme import apply_theme
from src.dashboard.utils.helpers import safe_list

st.set_page_config(page_title="Proven S/F | PPE", layout="wide")
apply_theme()

st.markdown("## Proven S/F")


@st.cache_data(ttl=300)
def _load():
    return load_proven_sf_dashboard_df(), load_latest_proven_sf_database()


df, full_db = _load()

if df is None or df.empty:
    st.warning("No Proven S/F programmer view available. Run `py run_build_sf_database.py`.")
    st.stop()

filtered = df.copy()

st.sidebar.markdown("### Filters")

if "material" in filtered.columns:
    materials = safe_list(filtered["material"])
    selected_materials = st.sidebar.multiselect("Material", materials, default=materials)
    if selected_materials:
        filtered = filtered[filtered["material"].astype(str).isin(selected_materials)]

if "machine_folder" in filtered.columns:
    machines = safe_list(filtered["machine_folder"])
    selected_machines = st.sidebar.multiselect("Machine", machines, default=machines)
    if selected_machines:
        filtered = filtered[filtered["machine_folder"].astype(str).isin(selected_machines)]

if "tool_number" in filtered.columns:
    tools = safe_list(filtered["tool_number"])
    selected_tools = st.sidebar.multiselect("Tool", tools, default=tools)
    if selected_tools:
        filtered = filtered[filtered["tool_number"].astype(str).isin(selected_tools)]

total_occurrences = (
    pd.to_numeric(filtered.get("occurrence_count", pd.Series(dtype=float)), errors="coerce")
    .fillna(0)
    .sum()
)
review_rows = (
    pd.to_numeric(filtered.get("needs_review_count", pd.Series(dtype=float)), errors="coerce")
    .fillna(0)
    .gt(0)
    .sum()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("S/F Groups", f"{len(filtered):,}")
c2.metric("Occurrences", f"{int(total_occurrences):,}")
c3.metric("Materials", f"{filtered['material'].nunique():,}" if "material" in filtered.columns else "0")
c4.metric("Review Flags", f"{int(review_rows):,}")

display_cols = [c for c in [
    "material",
    "machine_folder",
    "machine_family",
    "tool_number",
    "resolved_tool_name",
    "resolved_tool_description",
    "tool_identity_source",
    "S_min",
    "S_avg",
    "S_max",
    "s_mode",
    "F_min",
    "F_avg",
    "F_max",
    "f_mode",
    "feed_intent_candidate",
    "occurrence_count",
    "program_count",
    "confidence_mix",
    "needs_review_count",
] if c in filtered.columns]

row_count_caption(filtered, "S/F group")

col_cfg = {
    "S_min": st.column_config.NumberColumn("S Min", format="%.0f"),
    "S_avg": st.column_config.NumberColumn("S Avg", format="%.0f"),
    "S_max": st.column_config.NumberColumn("S Max", format="%.0f"),
    "F_min": st.column_config.NumberColumn("F Min", format="%.5f"),
    "F_avg": st.column_config.NumberColumn("F Avg", format="%.5f"),
    "F_max": st.column_config.NumberColumn("F Max", format="%.5f"),
    "occurrence_count": st.column_config.NumberColumn("Occurrences"),
    "program_count": st.column_config.NumberColumn("Programs"),
    "needs_review_count": st.column_config.NumberColumn("Review Flags"),
}

show_table(filtered[display_cols], height=560, column_config=col_cfg)
download_csv(filtered[display_cols], "proven_sf_programmer_view_filtered.csv", "Export Filtered CSV")

with st.expander("Advanced source/debug info"):
    if full_db is None or full_db.empty:
        st.info("Full proven S/F database export is not available.")
    else:
        advanced = full_db.copy()
        for col in ("material", "machine_folder", "tool_number"):
            if col in filtered.columns and col in advanced.columns:
                allowed = set(filtered[col].dropna().astype(str))
                advanced = advanced[advanced[col].astype(str).isin(allowed)]

        advanced_cols = [c for c in [
            "source_file",
            "filename",
            "program_id",
            "matched_job_number",
            "matched_part_number",
            "matched_drawing_number",
            "linked_router_file",
            "link_method",
            "link_confidence",
            "review_reason",
            "raw_line",
            "prev_line",
            "next_line",
        ] if c in advanced.columns]
        if advanced_cols:
            show_table(advanced[advanced_cols], height=360)
        else:
            st.info("No advanced source columns are available in the full database.")
