"""
6_Analytics.py — Shop-level machining intelligence analytics.

Future sections for AI/operation classification are stubbed but not built.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.dashboard.styling.theme import apply_theme
from src.dashboard.data_access.loader import (
    load_latest_material_candidates,
    load_latest_cuts,
)
from src.dashboard.components.tables import show_table
from src.dashboard.components.metrics import metric_row
from src.dashboard.utils.helpers import count_by

st.set_page_config(page_title="Analytics | PPE", layout="wide")
apply_theme()

st.markdown("## Analytics")
st.caption("Shop-wide machining intelligence patterns extracted from proven CNC programs.")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load():
    return load_latest_material_candidates(), load_latest_cuts()

mc, cuts = _load()

if mc is None or mc.empty:
    st.warning("No material_candidates export found. Run `py run_match.py`.")
    st.stop()

# ── Shop Metrics ──────────────────────────────────────────────────────────────
total_groups = len(mc)
total_records = int(mc["record_count"].sum()) if "record_count" in mc.columns else 0
unique_machines = mc["machine_folder"].nunique() if "machine_folder" in mc.columns else 0
unique_tools = mc["tool_number"].nunique() if "tool_number" in mc.columns else 0
total_programs = int(mc["unique_program_count"].sum()) if "unique_program_count" in mc.columns else 0

metric_row([
    {"label": "Total Records", "value": f"{total_records:,}"},
    {"label": "Tool Groups", "value": f"{total_groups:,}"},
    {"label": "Unique Tools", "value": str(unique_tools)},
    {"label": "Machines", "value": str(unique_machines)},
    {"label": "Programs", "value": f"{total_programs:,}"},
])
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_combos, tab_sf, tab_intent, tab_conf, tab_future = st.tabs([
    "Top Tool Combos", "S/F Ranges", "Feed Intent", "Confidence", "Future Sections"
])

with tab_combos:
    st.markdown("#### Top Repeated Tool Combinations (by Record Count)")
    if "active_t_code" in mc.columns and "machine_folder" in mc.columns:
        combos = (
            mc.groupby(["machine_folder", "active_t_code", "s_mode"], dropna=False)["record_count"]
            .sum()
            .reset_index()
            .nlargest(25, "record_count")
        )
        combos["combo"] = (
            combos["machine_folder"].astype(str) + " / "
            + combos["active_t_code"].astype(str) + " ["
            + combos["s_mode"].astype(str) + "]"
        )
        fig = px.bar(
            combos, x="record_count", y="combo", orientation="h",
            title="Top 25 Tool Groups by Record Count",
            labels={"record_count": "Records", "combo": ""},
            color_discrete_sequence=["#FF6B35"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
            yaxis={"categoryorder": "total ascending"},
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    if "sf_combo_count" in cuts.columns if cuts is not None else False:
        st.markdown("#### Duplicate S/F Combinations (from raw cuts)")
        dup = cuts[pd.to_numeric(cuts["sf_combo_count"], errors="coerce") > 1]
        st.metric("Records with Repeated S/F Combo", f"{len(dup):,}")

with tab_sf:
    col_a, col_b = st.columns(2)
    with col_a:
        if "s_mean" in mc.columns and "s_mode" in mc.columns:
            css_df = mc[mc["s_mode"] == "CSS"].copy()
            css_df["s_mean_n"] = pd.to_numeric(css_df["s_mean"], errors="coerce")
            fig = px.histogram(
                css_df.dropna(subset=["s_mean_n"]), x="s_mean_n",
                nbins=40,
                title="SFM Distribution (CSS mode — all machines)",
                labels={"s_mean_n": "SFM", "count": "Groups"},
                color_discrete_sequence=["#FF6B35"],
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C")
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if "f_mean" in mc.columns:
            f_vals = pd.to_numeric(mc["f_mean"], errors="coerce").dropna()
            ipr = f_vals[f_vals < 1.0]
            if not ipr.empty:
                fig = px.histogram(
                    ipr, nbins=40,
                    title="IPR Feed Distribution (all machines)",
                    labels={"value": "IPR", "count": "Groups"},
                    color_discrete_sequence=["#00B4D8"],
                    template="plotly_dark",
                )
                fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C")
                st.plotly_chart(fig, use_container_width=True)

    # SFM by machine comparison
    if "s_mean" in mc.columns and "machine_folder" in mc.columns and "s_mode" in mc.columns:
        css_all = mc[mc["s_mode"] == "CSS"].copy()
        css_all["s_mean_n"] = pd.to_numeric(css_all["s_mean"], errors="coerce")
        agg = (
            css_all.groupby("machine_folder")["s_mean_n"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )
        agg.columns = ["Machine", "Avg SFM", "Min SFM", "Max SFM"]
        st.markdown("#### SFM Range by Machine")
        show_table(agg.sort_values("Avg SFM", ascending=False))

with tab_intent:
    if "feed_intent_candidate" in mc.columns:
        vc = mc["feed_intent_candidate"].value_counts().reset_index()
        vc.columns = ["Intent", "Groups"]
        fig = px.bar(
            vc, x="Groups", y="Intent", orientation="h",
            title="Feed Intent Distribution (all machines)",
            color_discrete_sequence=["#00B4D8"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if "feed_intent_confidence" in mc.columns:
        vc2 = mc["feed_intent_confidence"].value_counts().reset_index()
        vc2.columns = ["Confidence", "Count"]
        fig = px.pie(
            vc2, names="Confidence", values="Count",
            title="Feed Intent Confidence Distribution",
            color="Confidence",
            color_discrete_map={"HIGH": "#00C851", "MEDIUM": "#FFB347", "NONE": "#555"},
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#1C1C1C")
        st.plotly_chart(fig, use_container_width=True)

with tab_conf:
    col_c, col_d = st.columns(2)
    with col_c:
        if "confidence_label" in mc.columns:
            vc = mc["confidence_label"].value_counts().reset_index()
            vc.columns = ["Confidence", "Count"]
            fig = px.bar(
                vc, x="Confidence", y="Count",
                title="Material Match Confidence Distribution",
                color="Confidence",
                color_discrete_map={
                    "HIGH": "#00C851", "MEDIUM": "#FFB347", "LOW": "#FF6B6B", "NONE": "#555",
                },
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col_d:
        if cuts is not None and "extraction_confidence" in cuts.columns:
            vc = cuts["extraction_confidence"].value_counts().reset_index()
            vc.columns = ["Confidence", "Records"]
            fig = px.pie(
                vc, names="Confidence", values="Records",
                title="Parser Extraction Confidence (raw cuts)",
                color="Confidence",
                color_discrete_map={"HIGH": "#00C851", "MEDIUM": "#FFB347", "LOW": "#FF6B6B"},
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#1C1C1C")
            st.plotly_chart(fig, use_container_width=True)

    if cuts is not None and "is_duplicate" in cuts.columns:
        dup_count = cuts["is_duplicate"].astype(str).str.lower().isin(("true", "1")).sum()
        total_cuts = len(cuts)
        st.metric(
            "Duplicate S/F Combos",
            f"{dup_count:,}",
            delta=f"{dup_count/total_cuts*100:.1f}% of records" if total_cuts else None,
        )

with tab_future:
    st.markdown(
        """
        <div class="ppe-banner ppe-banner-info">
          Future sections — not yet implemented. These will be added when the dashboard
          is integrated into the Manufacturing Command Center.
        </div>
        """,
        unsafe_allow_html=True,
    )
    placeholders = [
        ("Operation Classification", "Auto-classify cuts as roughing, finishing, grooving, drilling, etc. based on S/F signature."),
        ("Process Flow Learning", "Learn common tool sequences and flag deviations from proven patterns."),
        ("AI Recommendations", "Suggest optimal S/F ranges for new programs based on proven data."),
        ("Training Assistant", "Generate machining parameter guidance from the proven dataset."),
    ]
    for title, desc in placeholders:
        with st.expander(f"[ PLANNED ] {title}"):
            st.markdown(f"*{desc}*")
            st.caption("Not implemented in this phase.")
