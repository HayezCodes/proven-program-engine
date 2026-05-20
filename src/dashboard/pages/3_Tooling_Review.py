"""
3_Tooling_Review.py — Review tooling mismatches and save corrections locally.
Never modifies original exports or production files.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.dashboard.styling.theme import apply_theme, match_status_badge
from src.dashboard.data_access.loader import (
    load_latest_tooling_review,
    load_latest_tooldb_reference,
)
from src.dashboard.data_access.overrides import (
    apply_tooling_overrides,
    save_tooling_overrides,
    load_tooling_overrides,
)
from src.dashboard.data_access.tool_identity import resolve_tool_identity_df
from src.dashboard.components.filters import (
    machine_filter, match_status_filter,
    tool_source_filter, needs_review_filter,
)
from src.dashboard.components.tables import show_table, download_csv, row_count_caption
from src.dashboard.components.metrics import metric_row

st.set_page_config(page_title="Tooling Review | PPE", layout="wide")
apply_theme()

st.markdown("## Tooling Review")
st.caption(
    "Compare parsed tool descriptions against the shop reference. "
    "Corrections saved to data/overrides/ — never overwrites original exports."
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load():
    tr = load_latest_tooling_review()
    tooldb = load_latest_tooldb_reference()
    return tr, tooldb

tr, tooldb_ref = _load()

if tr is None or tr.empty:
    st.warning(
        "No tooling_review export found. Run `py run_tooling_review.py` to generate it."
    )
    st.stop()

df = apply_tooling_overrides(tr)
df = resolve_tool_identity_df(df, tooldb_ref=tooldb_ref)

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
df = machine_filter(df, key="tr_machine")
df = match_status_filter(df, key="tr_status")
df = tool_source_filter(df, key="tr_tool_src")
df = needs_review_filter(df, key="tr_needs_review")
show_override_only = st.sidebar.checkbox("Overridden rows only", key="tr_ov_only")
if show_override_only and "is_overridden" in df.columns:
    df = df[df["is_overridden"]]

# ── Metrics ───────────────────────────────────────────────────────────────────
if "match_status" in tr.columns:
    counts = tr["match_status"].value_counts()
    metric_row([
        {"label": "Total", "value": f"{len(tr):,}"},
        {"label": "Matched", "value": str(int(counts.get("description_match", 0)))},
        {"label": "Differs", "value": str(int(counts.get("description_differs", 0)))},
        {"label": "Missing from Ref", "value": str(int(counts.get("missing_from_reference", 0)))},
        {"label": "Needs Review", "value": str(int(counts.get("needs_review", 0)))},
    ])
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_view, tab_edit, tab_dist = st.tabs(["View", "Edit Corrections", "Status Distribution"])

with tab_view:
    row_count_caption(df, "record")
    display_cols = [c for c in [
        "machine_folder", "tool_number", "active_t_code",
        "resolved_tool_name", "resolved_tool_source", "resolved_tool_needs_review",
        "program_description", "reference_description", "decimal_size",
        "match_status", "reference_needs_review",
        "review_action", "corrected_description", "notes",
        "is_overridden",
    ] if c in df.columns]
    show_table(df[display_cols], height=450)
    download_csv(df, "tooling_review_filtered.csv")

with tab_edit:
    st.markdown(
        '<div class="ppe-banner ppe-banner-warn">'
        "Edit <b>review_action</b>, <b>corrected_description</b>, and <b>notes</b> columns below. "
        "Click <b>Save Overrides</b> to persist changes locally."
        "</div>",
        unsafe_allow_html=True,
    )

    editable_cols = [c for c in [
        "machine_folder", "tool_number", "active_t_code",
        "program_description", "reference_description",
        "match_status",
        "review_action", "corrected_description", "notes",
    ] if c in df.columns]

    edit_df = df[editable_cols].copy()

    # Disable editing on non-review columns
    disabled_cols = [c for c in editable_cols if c not in ("review_action", "corrected_description", "notes")]

    edited = st.data_editor(
        edit_df,
        disabled=disabled_cols,
        hide_index=True,
        use_container_width=True,
        height=400,
        key="tr_editor",
    )

    if st.button("Save Overrides", type="primary", key="tr_save"):
        # Build override rows from edited data — only rows with non-empty review fields
        new_overrides = edited[
            (edited.get("review_action", pd.Series("", index=edited.index)).astype(str).str.strip() != "")
            | (edited.get("corrected_description", pd.Series("", index=edited.index)).astype(str).str.strip() != "")
        ].copy()

        if new_overrides.empty:
            st.info("No changes to save.")
        else:
            from datetime import datetime
            new_overrides["override_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            existing = load_tooling_overrides()
            # Upsert: replace existing keys, append new
            for col in ("tool_number",):
                if col in new_overrides.columns:
                    new_overrides[col] = new_overrides[col].astype(str)
                if col in existing.columns:
                    existing[col] = existing[col].astype(str)

            key_cols = ["machine_folder", "tool_number"]
            existing_key = existing[key_cols].astype(str).apply(tuple, axis=1)
            new_key = new_overrides[key_cols].astype(str).apply(tuple, axis=1)

            keep_existing = existing[~existing_key.isin(new_key)]
            combined = pd.concat(
                [keep_existing, new_overrides[existing.columns.intersection(new_overrides.columns)]],
                ignore_index=True,
            )
            save_tooling_overrides(combined)
            st.success(f"Saved {len(new_overrides)} overrides to data/overrides/tooling_overrides.csv")
            st.cache_data.clear()

with tab_dist:
    if "match_status" in df.columns:
        vc = df["match_status"].value_counts().reset_index()
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
            title="Tooling Match Status Distribution",
            template="plotly_dark",
            color="Status",
            color_discrete_map=_colors,
        )
        fig.update_layout(
            paper_bgcolor="#1C1C1C", plot_bgcolor="#1C1C1C",
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
