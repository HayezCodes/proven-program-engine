"""
1_Proven_SF.py — Proven S/F Lookup
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.components.tables import download_csv, show_table
from src.dashboard.data_access.loader import load_latest_proven_sf_lookup
from src.dashboard.styling.theme import apply_theme
from src.dashboard.utils.helpers import safe_list

st.set_page_config(page_title="Proven S/F Lookup | PPE", layout="wide")
apply_theme()

st.markdown("## Proven S/F Lookup")
st.caption("Select a material to look up proven speeds and feeds from shop programs.")

# ---------------------------------------------------------------------------
# Load lookup export
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load():
    return load_latest_proven_sf_lookup()


df_raw = _load()

if df_raw is None or df_raw.empty:
    st.warning("No lookup data. Run `py run_build_sf_database.py`.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Filters")

# Material — primary
all_mats = [m for m in safe_list(df_raw, "material") if m != "UNKNOWN"]
sel_mat = st.sidebar.multiselect(
    "Material", all_mats, default=[], key="lk_mat",
)

# Tool type
all_types = safe_list(df_raw, "tool_type")
sel_type = st.sidebar.multiselect(
    "Tool Type", all_types, default=[], key="lk_type",
)

# Machine
all_mach = safe_list(df_raw, "machine_folder")
sel_mach = st.sidebar.multiselect(
    "Machine", all_mach, default=[], key="lk_mach",
)

# Confidence — default HIGH + MEDIUM; LOW hidden
sel_conf = st.sidebar.multiselect(
    "Confidence", ["HIGH", "MEDIUM", "LOW"],
    default=["HIGH", "MEDIUM"], key="lk_conf",
)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

f = df_raw.copy()

# Default: hide UNKNOWN material unless explicitly requested
if sel_mat:
    f = f[f["material"].astype(str).isin(sel_mat)]
else:
    f = f[f["material"].astype(str) != "UNKNOWN"]

if sel_type:
    f = f[f["tool_type"].astype(str).isin(sel_type)]

if sel_mach:
    f = f[f["machine_folder"].astype(str).isin(sel_mach)]

if sel_conf:
    f = f[f["confidence"].astype(str).isin(sel_conf)]
else:
    f = f[f["confidence"].astype(str).isin(["HIGH", "MEDIUM"])]

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

COLS = [c for c in [
    "material",
    "machine_folder",
    "tool_type",
    "tool_description",
    "operation_intent",
    "s_mode",
    "S_low", "S_mid", "S_high",
    "f_mode",
    "F_low", "F_mid", "F_high",
    "occurrence_count",
    "confidence",
] if c in f.columns]

COL_CFG = {
    "tool_description":  st.column_config.TextColumn("Tool"),
    "operation_intent":  st.column_config.TextColumn("Operation"),
    "machine_folder":    st.column_config.TextColumn("Machine"),
    "tool_type":         st.column_config.TextColumn("Tool Type"),
    "s_mode":            st.column_config.TextColumn("S Mode"),
    "f_mode":            st.column_config.TextColumn("F Mode"),
    "confidence":        st.column_config.TextColumn("Confidence"),
    "S_low":  st.column_config.NumberColumn("S Low",  format="%.0f"),
    "S_mid":  st.column_config.NumberColumn("S Mid",  format="%.0f"),
    "S_high": st.column_config.NumberColumn("S High", format="%.0f"),
    "F_low":  st.column_config.NumberColumn("F Low",  format="%.5f"),
    "F_mid":  st.column_config.NumberColumn("F Mid",  format="%.5f"),
    "F_high": st.column_config.NumberColumn("F High", format="%.5f"),
    "occurrence_count": st.column_config.NumberColumn("Occurrences"),
}

st.caption(f"{len(f):,} result{'s' if len(f) != 1 else ''}")
show_table(f[COLS], height=620, column_config=COL_CFG)
download_csv(f[COLS], "proven_sf_lookup.csv", "Export CSV")
