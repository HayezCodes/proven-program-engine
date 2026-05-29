"""
1_Proven_SF.py — Proven S/F Guide

Primary workflow: select material → see proven speeds and feeds.
Optional secondary filters: machine, tool, operation.
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

st.set_page_config(page_title="Proven S/F Guide | PPE", layout="wide")
apply_theme()

st.markdown("## Proven S/F Guide")
st.caption("Proven speeds and feeds from CNC programs. Select a material to get started.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load():
    return load_proven_sf_dashboard_df(), load_latest_proven_sf_database()


df_raw, full_db = _load()

if df_raw is None or df_raw.empty:
    st.warning("No Proven S/F data available. Run `py run_build_sf_database.py`.")
    st.stop()

# ---------------------------------------------------------------------------
# Derive material_confidence from confidence_mix
#
# confidence_mix format: "HIGH:42" | "MEDIUM:1145" | "LOW:1922"
# Mapping to user-facing labels:
#   HIGH   — direct router/job material (sf_record_confidence == HIGH)
#   MEDIUM — router consensus or matched (sf_record_confidence == MEDIUM)
#   UNKNOWN — no material verification  (sf_record_confidence == LOW only)
# ---------------------------------------------------------------------------

_CONF_ORDER = ["HIGH", "MEDIUM", "UNKNOWN"]


def _dominant_confidence(mix: str) -> str:
    m = str(mix or "")
    if "HIGH"   in m: return "HIGH"
    if "MEDIUM" in m: return "MEDIUM"
    return "UNKNOWN"


df = df_raw.copy()
df["material_confidence"] = df["confidence_mix"].apply(_dominant_confidence)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("### Filters")

# 1. Material — primary selector (empty = all)
all_materials = safe_list(df, "material")
sel_materials = st.sidebar.multiselect(
    "Material", all_materials, default=[], key="sf_material",
)

# 2. Material Confidence — default HIGH + MEDIUM; UNKNOWN hidden
sel_conf = st.sidebar.multiselect(
    "Material Confidence", _CONF_ORDER, default=["HIGH", "MEDIUM"], key="sf_conf",
)

# 3. Machine (empty = all)
all_machines = safe_list(df, "machine_folder")
sel_machines = st.sidebar.multiselect(
    "Machine", all_machines, default=[], key="sf_machine",
)

# 4. Tool (empty = all)
all_tools = safe_list(df, "tool_number")
sel_tools = st.sidebar.multiselect(
    "Tool", all_tools, default=[], key="sf_tool",
)

# 5. Operation / Feed Intent (empty = all)
all_intents = safe_list(df, "feed_intent_candidate")
sel_intents = st.sidebar.multiselect(
    "Operation / Feed Intent", all_intents, default=[], key="sf_intent",
)

# ---------------------------------------------------------------------------
# Apply filters — empty selection means "no restriction" for that dimension
# ---------------------------------------------------------------------------

filtered = df.copy()

if sel_materials:
    filtered = filtered[filtered["material"].astype(str).isin(sel_materials)]

if sel_conf:
    filtered = filtered[filtered["material_confidence"].isin(sel_conf)]

if sel_machines:
    filtered = filtered[filtered["machine_folder"].astype(str).isin(sel_machines)]

if sel_tools:
    filtered = filtered[filtered["tool_number"].astype(str).isin(sel_tools)]

if sel_intents:
    filtered = filtered[filtered["feed_intent_candidate"].astype(str).isin(sel_intents)]

# ---------------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------------

def _occ_sum(frame: pd.DataFrame) -> int:
    return int(pd.to_numeric(frame.get("occurrence_count", pd.Series(dtype=float)),
                             errors="coerce").fillna(0).sum())


high_rows  = filtered[filtered["material_confidence"] == "HIGH"]
med_rows   = filtered[filtered["material_confidence"] == "MEDIUM"]
total_occ  = _occ_sum(filtered)
high_occ   = _occ_sum(high_rows)
med_occ    = _occ_sum(med_rows)
n_machines = filtered["machine_folder"].nunique() if "machine_folder" in filtered.columns else 0
n_tools    = filtered["tool_number"].nunique()    if "tool_number"    in filtered.columns else 0

if len(sel_materials) == 1:
    mat_label = sel_materials[0]
elif len(sel_materials) > 1:
    mat_label = f"{len(sel_materials)} materials"
else:
    n_unique = filtered["material"].nunique() if "material" in filtered.columns else 0
    mat_label = f"All ({n_unique})"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Material",          mat_label)
c2.metric("Proven Records",    f"{total_occ:,}")
c3.metric("Machines",          f"{n_machines:,}")
c4.metric("Tools",             f"{n_tools:,}")
c5.metric("HIGH Confidence",   f"{high_occ:,}")
c6.metric("MEDIUM Confidence", f"{med_occ:,}")

# ---------------------------------------------------------------------------
# Confidence guide (collapsed)
# ---------------------------------------------------------------------------

with st.expander("Material confidence guide", expanded=False):
    st.markdown("""
| Level | Source | Guidance |
|---|---|---|
| **HIGH** | Direct router/job link — exact program matched to a specific job order | Strongest signal. Use as-is. |
| **MEDIUM** | Router consensus — multiple repeat-order jobs reference the same program and agree on material | Usable. Verify on first use for a new job type. |
| **UNKNOWN** | No material verification — S/F inferred from shop reference table only | Research/reference only. Confirm material before use. |
""")

# ---------------------------------------------------------------------------
# Main table
# ---------------------------------------------------------------------------

_DISPLAY_COLS = [c for c in [
    "material",
    "machine_folder",
    "machine_family",
    "tool_number",
    "resolved_tool_name",
    "resolved_tool_description",
    "feed_intent_candidate",
    "s_mode",
    "S_min",
    "S_avg",
    "S_max",
    "f_mode",
    "F_min",
    "F_avg",
    "F_max",
    "occurrence_count",
    "program_count",
    "material_confidence",
    "needs_review_count",
] if c in filtered.columns]

_COL_CFG = {
    "material_confidence":      st.column_config.TextColumn("Mat. Confidence"),
    "feed_intent_candidate":    st.column_config.TextColumn("Operation"),
    "resolved_tool_name":       st.column_config.TextColumn("Tool Name"),
    "resolved_tool_description":st.column_config.TextColumn("Tool Description"),
    "machine_family":           st.column_config.TextColumn("Family"),
    "s_mode":                   st.column_config.TextColumn("S Mode"),
    "f_mode":                   st.column_config.TextColumn("F Mode"),
    "S_min":  st.column_config.NumberColumn("S Min",  format="%.0f"),
    "S_avg":  st.column_config.NumberColumn("S Avg",  format="%.0f"),
    "S_max":  st.column_config.NumberColumn("S Max",  format="%.0f"),
    "F_min":  st.column_config.NumberColumn("F Min",  format="%.5f"),
    "F_avg":  st.column_config.NumberColumn("F Avg",  format="%.5f"),
    "F_max":  st.column_config.NumberColumn("F Max",  format="%.5f"),
    "occurrence_count":   st.column_config.NumberColumn("Occurrences"),
    "program_count":      st.column_config.NumberColumn("Programs"),
    "needs_review_count": st.column_config.NumberColumn("Review Flags"),
}

row_count_caption(filtered, "S/F group")
show_table(filtered[_DISPLAY_COLS], height=580, column_config=_COL_CFG)
download_csv(filtered[_DISPLAY_COLS], "proven_sf_guide.csv", "Export CSV")

# ---------------------------------------------------------------------------
# Advanced / Debug expander — job/source/raw fields hidden by default
# ---------------------------------------------------------------------------

with st.expander("Advanced / Debug", expanded=False):
    if full_db is None or full_db.empty:
        st.info("Full proven S/F database not available.")
    else:
        adv = full_db.copy()

        # Align advanced view to current filter selections
        if sel_machines and "machine_folder" in adv.columns:
            adv = adv[adv["machine_folder"].astype(str).isin(sel_machines)]
        if sel_tools and "tool_number" in adv.columns:
            adv = adv[adv["tool_number"].astype(str).isin(
                [str(t) for t in sel_tools]
            )]
        if sel_materials and "verified_material" in adv.columns:
            adv = adv[adv["verified_material"].astype(str).isin(sel_materials)]

        adv_cols = [c for c in [
            "source_file",
            "filename",
            "machine_folder",
            "tool_number",
            "verified_material",
            "material_source",
            "material_confidence",
            "matched_job_number",
            "matched_part_number",
            "matched_drawing_number",
            "link_method",
            "link_confidence",
            "material_consensus_status",
            "candidate_job_count",
            "linked_router_file",
            "review_reason",
            "raw_line",
            "prev_line",
            "next_line",
        ] if c in adv.columns]

        if adv_cols:
            row_count_caption(adv, "cut record")
            show_table(adv[adv_cols], height=360)
            download_csv(adv[adv_cols], "proven_sf_debug.csv", "Export Debug CSV")
        else:
            st.info("No advanced columns available in full database.")
