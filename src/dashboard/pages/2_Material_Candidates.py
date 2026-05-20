"""
2_Material_Candidates.py — Review inferred material matches.
Approve or reject candidates. Corrections saved locally — never modifies exports.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.dashboard.styling.theme import apply_theme, confidence_badge, badge
from src.dashboard.data_access.loader import load_latest_material_candidates
from src.dashboard.data_access.overrides import (
    apply_material_overrides,
    upsert_material_override,
    load_material_overrides,
)
from src.dashboard.components.filters import (
    machine_filter, tool_filter, confidence_filter, match_type_filter,
)
from src.dashboard.components.tables import show_table, download_csv, row_count_caption
from src.dashboard.components.metrics import metric_row

st.set_page_config(page_title="Material Candidates | PPE", layout="wide")
apply_theme()

st.markdown("## Material Candidates")
st.caption(
    "Inferred material matches based on S/F reference bands. "
    "Approve or reject locally — never modifies source data."
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load():
    return load_latest_material_candidates()

mc = _load()

if mc is None or mc.empty:
    st.warning("No material_candidates export found. Run `py run_match.py`.")
    st.stop()

mc_with_ov = apply_material_overrides(mc)

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
df = mc_with_ov.copy()
df = machine_filter(df, key="mc_machine")
df = confidence_filter(df, key="mc_conf")
df = match_type_filter(df, key="mc_match")

decision_opts = ["all", "pending", "approved", "rejected"]
decision_filter = st.sidebar.selectbox("Review Decision", decision_opts, key="mc_decision")
if decision_filter != "all" and "review_decision" in df.columns:
    df = df[df["review_decision"] == decision_filter]

# ── Metrics ───────────────────────────────────────────────────────────────────
total = len(mc_with_ov)
approved = (mc_with_ov.get("review_decision", pd.Series()) == "approved").sum()
rejected = (mc_with_ov.get("review_decision", pd.Series()) == "rejected").sum()
pending = total - approved - rejected

metric_row([
    {"label": "Total Groups", "value": f"{total:,}"},
    {"label": "Approved", "value": str(int(approved))},
    {"label": "Rejected", "value": str(int(rejected))},
    {"label": "Pending Review", "value": str(int(pending))},
])
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_table, tab_dist, tab_review = st.tabs(["Candidates Table", "Distributions", "Quick Review"])

with tab_table:
    row_count_caption(df, "record")
    display_cols = [c for c in [
        "machine_folder", "tool_number", "active_t_code", "s_mode",
        "s_mean", "f_mean", "record_count",
        "material_candidate_1", "material_candidate_2", "material_candidate_3",
        "confidence_score", "confidence_label", "match_type", "reason",
        "feed_intent_candidate", "feed_intent_confidence",
        "review_decision", "reviewer_note",
    ] if c in df.columns]
    show_table(df[display_cols], height=450)
    download_csv(df, "material_candidates_filtered.csv")

with tab_dist:
    col_a, col_b = st.columns(2)
    with col_a:
        if "confidence_label" in mc_with_ov.columns:
            vc = mc_with_ov["confidence_label"].value_counts().reset_index()
            vc.columns = ["Confidence", "Count"]
            fig = px.pie(
                vc, names="Confidence", values="Count",
                title="Confidence Distribution",
                color="Confidence",
                color_discrete_map={
                    "HIGH": "#00C851", "MEDIUM": "#FFB347", "LOW": "#FF6B6B", "NONE": "#555",
                },
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#1C1C1C")
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if "match_type" in mc_with_ov.columns:
            vc = mc_with_ov["match_type"].value_counts().reset_index()
            vc.columns = ["Match Type", "Count"]
            fig = px.bar(
                vc, x="Count", y="Match Type", orientation="h",
                title="Match Type Distribution",
                color_discrete_sequence=["#FF6B35"],
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C")
            st.plotly_chart(fig, use_container_width=True)

    if "material_candidate_1" in mc_with_ov.columns:
        vc = (
            mc_with_ov[mc_with_ov["material_candidate_1"].notna()]["material_candidate_1"]
            .value_counts()
            .reset_index()
        )
        vc.columns = ["Material", "Count"]
        fig = px.bar(
            vc.head(15), x="Count", y="Material", orientation="h",
            title="Top Material Candidates",
            color_discrete_sequence=["#00B4D8"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_review:
    st.markdown(
        '<div class="ppe-banner ppe-banner-info">'
        "Select a record to approve or reject the material candidate. "
        "Saved locally to <code>data/overrides/material_overrides.csv</code>."
        "</div>",
        unsafe_allow_html=True,
    )

    pending_df = mc_with_ov[
        mc_with_ov.get("review_decision", pd.Series("pending", index=mc_with_ov.index)) == "pending"
    ] if "review_decision" in mc_with_ov.columns else mc_with_ov

    if pending_df.empty:
        st.success("All records have been reviewed.")
    else:
        key_cols = [c for c in ["machine_folder", "active_t_code", "material_candidate_1"] if c in pending_df.columns]
        pending_df["_label"] = pending_df[key_cols].astype(str).agg(" / ".join, axis=1)
        options = pending_df["_label"].tolist()

        selected_label = st.selectbox(
            f"Select record to review ({len(pending_df)} pending)", options, key="mc_sel"
        )
        if selected_label:
            row = pending_df[pending_df["_label"] == selected_label].iloc[0]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Candidate Info**")
                for f in ["material_candidate_1", "confidence_label", "match_type", "reason", "feed_intent_candidate"]:
                    if f in row.index:
                        st.text(f"{f}: {row[f]}")
            with c2:
                st.markdown("**S/F Values**")
                for f in ["s_mean", "s_min", "s_max", "f_mean"]:
                    if f in row.index:
                        st.text(f"{f}: {row[f]}")

            note = st.text_input("Note (optional)", key="mc_note")
            btn_col1, btn_col2, _ = st.columns([1, 1, 4])
            with btn_col1:
                if st.button("Approve", type="primary", key="mc_approve"):
                    upsert_material_override(
                        str(row.get("machine_folder", "")),
                        str(row.get("active_t_code", "")),
                        str(row.get("tool_number", "")),
                        "approved", note,
                    )
                    st.success("Saved — approved.")
                    st.rerun()
            with btn_col2:
                if st.button("Reject", key="mc_reject"):
                    upsert_material_override(
                        str(row.get("machine_folder", "")),
                        str(row.get("active_t_code", "")),
                        str(row.get("tool_number", "")),
                        "rejected", note,
                    )
                    st.warning("Saved — rejected.")
                    st.rerun()
