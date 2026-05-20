"""
4_Program_Search.py — Search extracted CNC program records.
Displays prev_line / current fields / next_line context for debugging parser behavior.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.dashboard.styling.theme import apply_theme
from src.dashboard.data_access.loader import load_latest_cuts
from src.dashboard.components.filters import machine_filter, tool_filter, smode_filter
from src.dashboard.components.tables import show_table, download_csv, row_count_caption
from src.dashboard.utils.helpers import text_search

st.set_page_config(page_title="Program Search | PPE", layout="wide")
apply_theme()

st.markdown("## Program Search")
st.caption("Search raw cut records extracted from CNC programs. Use for parser debugging and program lookup.")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load():
    return load_latest_cuts()

cuts = _load()

if cuts is None or cuts.empty:
    st.warning("No cuts export found. Run `py run_parse.py`.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
df = cuts.copy()
df = machine_filter(df, key="ps_machine")
df = tool_filter(df, key="ps_tool")
df = smode_filter(df, key="ps_smode")

# S value range
if "s_value" in df.columns:
    s_numeric = pd.to_numeric(df["s_value"], errors="coerce").dropna()
    if not s_numeric.empty:
        s_min_v, s_max_v = float(s_numeric.min()), float(s_numeric.max())
        if s_min_v < s_max_v:
            s_range = st.sidebar.slider(
                "S Value Range", s_min_v, s_max_v, (s_min_v, s_max_v), key="ps_srange"
            )
            df = df[
                pd.to_numeric(df["s_value"], errors="coerce").between(s_range[0], s_range[1])
                | df["s_value"].isna()
            ]

# F value range
if "f_value" in df.columns:
    f_numeric = pd.to_numeric(df["f_value"], errors="coerce").dropna()
    if not f_numeric.empty:
        f_min_v, f_max_v = float(f_numeric.min()), float(f_numeric.max())
        if f_min_v < f_max_v:
            f_range = st.sidebar.slider(
                "F Value Range", f_min_v, f_max_v, (f_min_v, f_max_v), key="ps_frange",
                format="%.4f",
            )
            df = df[
                pd.to_numeric(df["f_value"], errors="coerce").between(f_range[0], f_range[1])
                | df["f_value"].isna()
            ]

block_skip = st.sidebar.checkbox("Block-skip lines only", key="ps_bskip")
if block_skip and "block_skip" in df.columns:
    df = df[df["block_skip"].astype(str).str.lower().isin(("true", "1"))]

conf_opts = ["All"] + sorted(df["extraction_confidence"].dropna().unique().tolist()) if "extraction_confidence" in df.columns else ["All"]
conf_sel = st.sidebar.selectbox("Extraction Confidence", conf_opts, key="ps_conf")
if conf_sel != "All" and "extraction_confidence" in df.columns:
    df = df[df["extraction_confidence"] == conf_sel]

# ── Text search ───────────────────────────────────────────────────────────────
st.markdown("### Search")
query = st.text_input(
    "Search across program_id, tool_description, prev_line, next_line",
    key="ps_query",
    placeholder="e.g. G96, T0505, 1/2 end mill...",
)

search_cols = ["program_id", "tool_description", "active_t_code", "prev_line", "next_line"]
if query:
    df = text_search(df, query, [c for c in search_cols if c in df.columns])

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
row_count_caption(df, "record")

# Context view columns
context_cols = [c for c in [
    "machine_folder", "program_id", "line_number",
    "active_t_code", "tool_number", "tool_description",
    "s_value", "s_mode", "s_type",
    "f_value", "f_mode",
    "extraction_confidence",
    "block_skip", "is_duplicate",
    "prev_line", "next_line",
] if c in df.columns]

col_cfg = {
    "s_value": st.column_config.NumberColumn("S Value", format="%.0f"),
    "f_value": st.column_config.NumberColumn("F Value", format="%.5f"),
    "line_number": st.column_config.NumberColumn("Line #"),
}

show_table(df[context_cols], height=500, column_config=col_cfg)
download_csv(df, "program_search_results.csv")

# ── Detail expander ───────────────────────────────────────────────────────────
if not df.empty:
    with st.expander("Line Context Detail"):
        st.markdown("Select a row index to see full context including prev/next lines.")
        row_idx = st.number_input("Row index (0-based)", 0, max(len(df) - 1, 0), 0, key="ps_row_idx")
        if 0 <= row_idx < len(df):
            row = df.iloc[row_idx]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Previous Line**")
                st.code(str(row.get("prev_line", "—")))
                st.markdown("**Next Line**")
                st.code(str(row.get("next_line", "—")))
            with c2:
                st.markdown("**Full Record**")
                st.dataframe(row.to_frame(), use_container_width=True)
