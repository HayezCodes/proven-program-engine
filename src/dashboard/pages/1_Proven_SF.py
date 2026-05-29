"""
1_Proven_SF.py — Proven S/F Guide
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.components.tables import download_csv, show_table
from src.dashboard.data_access.loader import (
    load_latest_proven_sf_database,
    load_proven_sf_dashboard_df,
)
from src.dashboard.styling.theme import apply_theme
from src.dashboard.utils.helpers import safe_list

st.set_page_config(page_title="Proven S/F Guide | PPE", layout="wide")
apply_theme()

st.markdown("## Proven S/F Guide")
st.caption("Select a material to look up proven speeds and feeds.")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load():
    return load_proven_sf_dashboard_df(), load_latest_proven_sf_database()


df_raw, full_db = _load()

if df_raw is None or df_raw.empty:
    st.warning("No data. Run `py run_build_sf_database.py`.")
    st.stop()

# ---------------------------------------------------------------------------
# Derive material_confidence label from confidence_mix
# HIGH   → verified material from direct job/router link
# MEDIUM → router consensus (multiple jobs agree)
# UNKNOWN → no material verification
# ---------------------------------------------------------------------------

def _dominant(mix: str) -> str:
    m = str(mix or "")
    if "HIGH"   in m: return "HIGH"
    if "MEDIUM" in m: return "MEDIUM"
    return "UNKNOWN"


df = df_raw.copy()
df["material_confidence"] = df["confidence_mix"].apply(_dominant)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Filters")

sel_mat = st.sidebar.multiselect(
    "Material", safe_list(df, "material"), default=[], key="sf_mat",
)
sel_conf = st.sidebar.multiselect(
    "Material Confidence", ["HIGH", "MEDIUM", "UNKNOWN"],
    default=["HIGH", "MEDIUM"], key="sf_conf",
)
sel_mach = st.sidebar.multiselect(
    "Machine", safe_list(df, "machine_folder"), default=[], key="sf_mach",
)
sel_tool = st.sidebar.multiselect(
    "Tool", safe_list(df, "tool_number"), default=[], key="sf_tool",
)
sel_intent = st.sidebar.multiselect(
    "Operation / Feed Intent", safe_list(df, "feed_intent_candidate"),
    default=[], key="sf_intent",
)

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

f = df.copy()
if sel_mat:    f = f[f["material"].astype(str).isin(sel_mat)]
if sel_conf:   f = f[f["material_confidence"].isin(sel_conf)]
if sel_mach:   f = f[f["machine_folder"].astype(str).isin(sel_mach)]
if sel_tool:   f = f[f["tool_number"].astype(str).isin(sel_tool)]
if sel_intent: f = f[f["feed_intent_candidate"].astype(str).isin(sel_intent)]

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

COLS = [c for c in [
    "material",
    "machine_folder",
    "tool_number",
    "resolved_tool_description",
    "feed_intent_candidate",
    "s_mode",
    "S_min", "S_avg", "S_max",
    "f_mode",
    "F_min", "F_avg", "F_max",
    "occurrence_count",
    "program_count",
    "material_confidence",
] if c in f.columns]

COL_CFG = {
    "feed_intent_candidate":     st.column_config.TextColumn("Operation"),
    "resolved_tool_description": st.column_config.TextColumn("Tool Description"),
    "material_confidence":       st.column_config.TextColumn("Mat. Conf."),
    "s_mode":                    st.column_config.TextColumn("S Mode"),
    "f_mode":                    st.column_config.TextColumn("F Mode"),
    "S_min": st.column_config.NumberColumn("S Min", format="%.0f"),
    "S_avg": st.column_config.NumberColumn("S Avg", format="%.0f"),
    "S_max": st.column_config.NumberColumn("S Max", format="%.0f"),
    "F_min": st.column_config.NumberColumn("F Min", format="%.5f"),
    "F_avg": st.column_config.NumberColumn("F Avg", format="%.5f"),
    "F_max": st.column_config.NumberColumn("F Max", format="%.5f"),
    "occurrence_count": st.column_config.NumberColumn("Occurrences"),
    "program_count":    st.column_config.NumberColumn("Programs"),
}

st.caption(f"{len(f):,} result{'s' if len(f) != 1 else ''}")
show_table(f[COLS], height=600, column_config=COL_CFG)
download_csv(f[COLS], "proven_sf.csv", "Export CSV")

# ---------------------------------------------------------------------------
# Advanced / debug (collapsed)
# ---------------------------------------------------------------------------

with st.expander("Advanced / Debug", expanded=False):
    if full_db is None or full_db.empty:
        st.info("Full database not available.")
    else:
        adv = full_db.copy()
        if sel_mach and "machine_folder" in adv.columns:
            adv = adv[adv["machine_folder"].astype(str).isin(sel_mach)]
        if sel_tool and "tool_number" in adv.columns:
            adv = adv[adv["tool_number"].astype(str).isin([str(t) for t in sel_tool])]
        if sel_mat and "verified_material" in adv.columns:
            adv = adv[adv["verified_material"].astype(str).isin(sel_mat)]

        adv_cols = [c for c in [
            "source_file", "filename", "machine_folder", "tool_number",
            "verified_material", "material_source", "material_confidence",
            "matched_job_number", "link_method", "link_confidence",
            "material_consensus_status", "candidate_job_count",
            "review_reason", "raw_line",
        ] if c in adv.columns]

        if adv_cols:
            st.caption(f"{len(adv):,} cut records")
            show_table(adv[adv_cols], height=360)
            download_csv(adv[adv_cols], "proven_sf_debug.csv", "Export Debug CSV")
        else:
            st.info("No advanced columns available.")
