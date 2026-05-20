"""
1_Proven_Tools.py — View proven tooling intelligence: S/F ranges, material candidates,
feed intent, and tooling reference match status per tool group.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.styling.theme import apply_theme
from src.dashboard.data_access.loader import (
    load_latest_material_candidates,
    load_latest_tooling_review,
    build_proven_tools_df,
)
from src.dashboard.components.filters import (
    machine_filter, tool_filter, smode_filter,
    confidence_filter, material_candidate_filter, feed_intent_filter,
    sidebar_text_search,
)
from src.dashboard.components.tables import show_table, download_csv, row_count_caption
from src.dashboard.components.metrics import metric_row
from src.dashboard.utils.helpers import safe_list

st.set_page_config(page_title="Proven Tools | PPE", layout="wide")
apply_theme()

st.markdown("## Proven Tools")
st.caption("Proven S/F ranges extracted from real CNC programs — suggestions only, not ground truth.")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load():
    mc = load_latest_material_candidates()
    tr = load_latest_tooling_review()
    return build_proven_tools_df(mc, tr), mc, tr

df, mc_raw, tr_raw = _load()

if df is None or df.empty:
    st.warning("No data available. Run `py run_parse.py` then `py run_match.py`.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
df = machine_filter(df, key="pt_machine")
df = smode_filter(df, key="pt_smode")
df = confidence_filter(df, key="pt_conf")
df = material_candidate_filter(df, key="pt_mat")
df = feed_intent_filter(df, key="pt_intent")
df = sidebar_text_search(
    df, ["active_t_code", "tool_number", "material_candidate_1"], label="Search", key="pt_search"
)

# ── Metrics ───────────────────────────────────────────────────────────────────
total = len(df)
high = (df["confidence_label"] == "HIGH").sum() if "confidence_label" in df.columns else 0
matched = df["match_type"].isin(
    ["exact_match", "close_match", "multiple_possible_matches"]
).sum() if "match_type" in df.columns else 0
machines = df["machine_folder"].nunique() if "machine_folder" in df.columns else 0

metric_row([
    {"label": "Tool Groups", "value": f"{total:,}"},
    {"label": "Machines", "value": str(machines)},
    {"label": "High Confidence", "value": str(int(high))},
    {"label": "With Material Match", "value": str(int(matched))},
])

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
tab_table, tab_sfm, tab_feed, tab_top = st.tabs(
    ["Table", "SFM Distribution", "Feed Distribution", "Top Combinations"]
)

with tab_table:
    row_count_caption(df, "tool group")

    display_cols = [c for c in [
        "machine_folder", "tool_number", "active_t_code", "s_mode",
        "s_mean", "s_min", "s_max", "f_mean", "f_min", "f_max",
        "record_count", "unique_program_count",
        "confidence_label", "match_type",
        "material_candidate_1", "material_candidate_2", "material_candidate_3",
        "feed_intent_candidate", "feed_intent_confidence",
        "match_status",
    ] if c in df.columns]

    col_cfg = {
        "s_mean": st.column_config.NumberColumn("S Mean", format="%.0f"),
        "s_min": st.column_config.NumberColumn("S Min", format="%.0f"),
        "s_max": st.column_config.NumberColumn("S Max", format="%.0f"),
        "f_mean": st.column_config.NumberColumn("F Mean", format="%.5f"),
        "f_min": st.column_config.NumberColumn("F Min", format="%.5f"),
        "f_max": st.column_config.NumberColumn("F Max", format="%.5f"),
        "record_count": st.column_config.NumberColumn("Records"),
        "unique_program_count": st.column_config.NumberColumn("Programs"),
    }

    show_table(df[display_cols], height=450, column_config=col_cfg)
    download_csv(df, "proven_tools_filtered.csv", "Export Filtered CSV")

    # Expandable source program viewer
    if "machine_folder" in df.columns and "active_t_code" in df.columns:
        with st.expander("Source Program Detail"):
            tool_options = (
                df[["machine_folder", "active_t_code"]]
                .dropna()
                .apply(lambda r: f"{r['machine_folder']} / {r['active_t_code']}", axis=1)
                .tolist()
            )
            if tool_options:
                selected = st.selectbox("Select tool group", tool_options, key="pt_detail_sel")
                if selected:
                    parts = selected.split(" / ", 1)
                    if len(parts) == 2:
                        row = df[
                            (df["machine_folder"].astype(str) == parts[0])
                            & (df["active_t_code"].astype(str) == parts[1])
                        ]
                        if not row.empty:
                            st.dataframe(row.T, use_container_width=True)

with tab_sfm:
    if "s_mean" in df.columns:
        s_data = pd.to_numeric(df["s_mean"], errors="coerce").dropna()
        if not s_data.empty:
            fig = px.histogram(
                s_data, nbins=30,
                title="SFM Distribution (s_mean across tool groups)",
                labels={"value": "SFM", "count": "Groups"},
                color_discrete_sequence=["#FF6B35"],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SFM data in current selection.")
    else:
        st.info("s_mean column not available.")

with tab_feed:
    if "f_mean" in df.columns:
        f_data = pd.to_numeric(df["f_mean"], errors="coerce").dropna()
        ipr = f_data[f_data < 1.0]
        ipm = f_data[f_data >= 1.0]
        col_a, col_b = st.columns(2)
        with col_a:
            if not ipr.empty:
                fig = px.histogram(
                    ipr, nbins=25,
                    title=f"IPR Feed Distribution ({len(ipr)} groups)",
                    labels={"value": "IPR", "count": "Groups"},
                    color_discrete_sequence=["#00B4D8"],
                    template="plotly_dark",
                )
                fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No IPR feed data.")
        with col_b:
            if not ipm.empty:
                fig = px.histogram(
                    ipm, nbins=25,
                    title=f"IPM Feed Distribution ({len(ipm)} groups)",
                    labels={"value": "IPM", "count": "Groups"},
                    color_discrete_sequence=["#00C851"],
                    template="plotly_dark",
                )
                fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No IPM feed data.")
    else:
        st.info("f_mean column not available.")

with tab_top:
    if "record_count" in df.columns and "active_t_code" in df.columns:
        top_df = (
            df.groupby(["machine_folder", "active_t_code"], dropna=False)["record_count"]
            .sum()
            .reset_index()
            .nlargest(20, "record_count")
        )
        top_df["label"] = top_df["machine_folder"].astype(str) + " / " + top_df["active_t_code"].astype(str)
        fig = px.bar(
            top_df, x="record_count", y="label", orientation="h",
            title="Top 20 Tool Groups by Record Count",
            labels={"record_count": "Records", "label": "Machine / T-Code"},
            color_discrete_sequence=["#FF6B35"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)
