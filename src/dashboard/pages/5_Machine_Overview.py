"""
5_Machine_Overview.py — High-level machine intelligence summary.
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
    load_latest_tooldb_reference,
    build_proven_tools_df,
)
from src.dashboard.data_access.overrides import load_tooling_overrides
from src.dashboard.data_access.tool_identity import resolve_tool_identity_df
from src.dashboard.components.tables import show_table, row_count_caption
from src.dashboard.components.metrics import metric_row
from src.dashboard.utils.helpers import safe_list

st.set_page_config(page_title="Machine Overview | PPE", layout="wide")
apply_theme()

st.markdown("## Machine Overview")
st.caption("Per-machine intelligence summary derived from proven CNC programs.")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load():
    mc = load_latest_material_candidates()
    tr = load_latest_tooling_review()
    tooldb = load_latest_tooldb_reference()
    overrides = load_tooling_overrides()
    enriched = resolve_tool_identity_df(build_proven_tools_df(mc, tr), tooldb_ref=tooldb, overrides=overrides)
    return enriched, mc, tr

mc_enriched, mc, tr = _load()

if mc is None or mc.empty:
    st.warning("No material_candidates export. Run `py run_match.py`.")
    st.stop()

# ── Machine selector ──────────────────────────────────────────────────────────
machines = safe_list(mc, "machine_folder")
if not machines:
    st.warning("No machine_folder data found.")
    st.stop()

all_label = "ALL MACHINES"
machine_sel = st.selectbox("Select Machine", [all_label] + machines, key="mo_machine")

mc_view = mc_enriched if machine_sel == all_label else mc_enriched[mc_enriched["machine_folder"] == machine_sel]
tr_view = (
    tr if (tr is None or tr.empty or machine_sel == all_label)
    else tr[tr["machine_folder"] == machine_sel]
)

# ── Top-level metrics ─────────────────────────────────────────────────────────
total_groups = len(mc_view)
unique_tools = mc_view["tool_number"].nunique() if "tool_number" in mc_view.columns else 0
total_records = int(mc_view["record_count"].sum()) if "record_count" in mc_view.columns else 0
total_programs = int(mc_view["unique_program_count"].sum()) if "unique_program_count" in mc_view.columns else 0
high_conf = int((mc_view["confidence_label"] == "HIGH").sum()) if "confidence_label" in mc_view.columns else 0

mismatches = 0
if tr_view is not None and not tr_view.empty and "match_status" in tr_view.columns:
    mismatches = int(tr_view["match_status"].isin(
        ["description_differs", "missing_from_reference"]
    ).sum())

metric_row([
    {"label": "Tool Groups", "value": f"{total_groups:,}"},
    {"label": "Unique Tools", "value": str(unique_tools)},
    {"label": "Total Records", "value": f"{total_records:,}"},
    {"label": "Programs", "value": f"{total_programs:,}"},
    {"label": "High Confidence", "value": str(high_conf)},
    {"label": "Tooling Mismatches", "value": str(mismatches)},
])
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_tools, tab_sf, tab_mat, tab_tooling = st.tabs(
    ["Top Tools", "S/F Overview", "Material Distribution", "Tooling Status"]
)

with tab_tools:
    if "record_count" in mc_view.columns and "active_t_code" in mc_view.columns:
        top = (
            mc_view.groupby("active_t_code", dropna=False)["record_count"]
            .sum()
            .nlargest(20)
            .reset_index()
        )
        top.columns = ["T-Code", "Records"]
        fig = px.bar(
            top, x="Records", y="T-Code", orientation="h",
            title=f"Top 20 Tools by Record Count — {machine_sel}",
            color_discrete_sequence=["#FF6B35"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Most Common Tools Table")
        _tool_cols = [c for c in [
            "active_t_code", "tool_number", "resolved_tool_name", "resolved_tool_source",
            "s_mode", "record_count", "unique_program_count",
        ] if c in mc_view.columns]
        show_table(
            mc_view[_tool_cols]
            .sort_values("record_count", ascending=False)
            .head(30),
            height=350,
        )

with tab_sf:
    col_a, col_b = st.columns(2)
    with col_a:
        if "s_mean" in mc_view.columns:
            s_css = mc_view[mc_view["s_mode"] == "CSS"]["s_mean"].dropna()
            s_css_n = pd.to_numeric(s_css, errors="coerce").dropna()
            if not s_css_n.empty:
                fig = px.box(
                    s_css_n, title=f"SFM Distribution (CSS) — {machine_sel}",
                    template="plotly_dark", color_discrete_sequence=["#FF6B35"],
                )
                fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CSS SFM data.")
    with col_b:
        if "f_mean" in mc_view.columns:
            f_vals = pd.to_numeric(mc_view["f_mean"], errors="coerce").dropna()
            ipr_vals = f_vals[f_vals < 1.0]
            if not ipr_vals.empty:
                fig = px.box(
                    ipr_vals, title=f"IPR Feed Distribution — {machine_sel}",
                    template="plotly_dark", color_discrete_sequence=["#00B4D8"],
                )
                fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No IPR feed data.")

    # Average S/F per machine comparison (all machines view)
    if machine_sel == all_label and "machine_folder" in mc_view.columns:
        st.markdown("#### Average SFM by Machine")
        if "s_mean" in mc_view.columns and "s_mode" in mc_view.columns:
            css_df = mc_view[mc_view["s_mode"] == "CSS"].copy()
            if not css_df.empty:
                css_df["s_mean_n"] = pd.to_numeric(css_df["s_mean"], errors="coerce")
                agg = (
                    css_df.groupby("machine_folder")["s_mean_n"]
                    .mean()
                    .reset_index()
                    .sort_values("s_mean_n", ascending=False)
                )
                agg.columns = ["Machine", "Avg SFM"]
                fig = px.bar(
                    agg, x="Machine", y="Avg SFM",
                    title="Average SFM per Machine (CSS mode)",
                    color_discrete_sequence=["#FF6B35"],
                    template="plotly_dark",
                )
                fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C")
                st.plotly_chart(fig, use_container_width=True)

with tab_mat:
    if "material_candidate_1" in mc_view.columns:
        vc = mc_view["material_candidate_1"].value_counts().reset_index()
        vc.columns = ["Material", "Groups"]
        fig = px.pie(
            vc.head(10), names="Material", values="Groups",
            title=f"Material Candidate Distribution — {machine_sel}",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#1C1C1C")
        st.plotly_chart(fig, use_container_width=True)

    if "confidence_label" in mc_view.columns:
        vc2 = mc_view["confidence_label"].value_counts().reset_index()
        vc2.columns = ["Confidence", "Count"]
        col_c, col_d = st.columns(2)
        with col_c:
            fig = px.bar(
                vc2, x="Confidence", y="Count",
                title=f"Confidence Distribution — {machine_sel}",
                color="Confidence",
                color_discrete_map={
                    "HIGH": "#00C851", "MEDIUM": "#FFB347", "LOW": "#FF6B6B", "NONE": "#555",
                },
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with tab_tooling:
    if tr_view is None or tr_view.empty:
        st.info("No tooling_review data. Run `py run_tooling_review.py`.")
    else:
        if "match_status" in tr_view.columns:
            vc = tr_view["match_status"].value_counts().reset_index()
            vc.columns = ["Status", "Count"]
            _colors = {
                "description_match": "#00C851",
                "description_differs": "#FFB347",
                "missing_from_reference": "#FF4444",
                "no_description_in_reference": "#FFDD47",
                "no_program_description": "#888888",
                "no_reference_data": "#444444",
                "needs_review": "#FF6B35",
            }
            fig = px.bar(
                vc, x="Count", y="Status", orientation="h",
                title=f"Tooling Match Status — {machine_sel}",
                color="Status", color_discrete_map=_colors,
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Tooling Reference Table")
        show_table(
            tr_view[[c for c in [
                "machine_folder", "tool_number", "active_t_code",
                "program_description", "reference_description", "match_status",
            ] if c in tr_view.columns]],
            height=350,
        )
