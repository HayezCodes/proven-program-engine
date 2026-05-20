"""
app.py — Proven Program Engine Dashboard home page.

Run from project root:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for all page imports
_PROJECT_ROOT = Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.dashboard.styling.theme import apply_theme
from src.dashboard.data_access.loader import get_export_status, load_latest_material_candidates
from src.dashboard.utils.helpers import count_by

st.set_page_config(
    page_title="Proven Program Engine",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="border-bottom:1px solid #333; padding-bottom:12px; margin-bottom:20px;">
      <span style="color:#FF6B35; font-size:1.1rem; font-weight:700; letter-spacing:0.1em;">
        PROVEN PROGRAM ENGINE
      </span>
      <span style="color:#555; font-size:0.85rem; margin-left:12px;">
        CNC Machining Intelligence Dashboard
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── System Status ─────────────────────────────────────────────────────────────
st.markdown("### System Status")

status = get_export_status()
_icons = {"found": "✓", "missing": "✗"}

col1, col2, col3, col4 = st.columns(4)
_cards = [
    ("cuts", "Cuts", col1),
    ("tool_summary", "Tool Summary", col2),
    ("material_candidates", "Material Candidates", col3),
    ("tooling_review", "Tooling Review", col4),
]
for key, label, col in _cards:
    info = status[key]
    with col:
        if info["found"]:
            st.success(f"**{label}**  \n{info['filename']}  \n{info['modified']}")
        else:
            st.warning(f"**{label}**  \nNot found — run pipeline")

# ── Quick Stats ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Quick Stats")

mc = load_latest_material_candidates()

if mc is not None and not mc.empty:
    total_groups = len(mc)
    unique_machines = mc["machine_folder"].nunique() if "machine_folder" in mc.columns else 0
    high_conf = (mc["confidence_label"] == "HIGH").sum() if "confidence_label" in mc.columns else 0
    matched = mc["match_type"].isin(
        ["exact_match", "close_match", "multiple_possible_matches"]
    ).sum() if "match_type" in mc.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tool Groups", f"{total_groups:,}")
    c2.metric("Machines", f"{unique_machines:,}")
    c3.metric("High Confidence", f"{high_conf:,}")
    c4.metric("With Material Match", f"{matched:,}")
    c5.metric("Match Rate", f"{matched/total_groups*100:.0f}%" if total_groups else "—")
else:
    st.info(
        "No material_candidates export found. "
        "Run `py run_parse.py` then `py run_match.py` to generate data."
    )

# ── Navigation Guide ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Pages")

nav_items = [
    ("1 — Proven Tools", "Review proven S/F ranges, material candidates, and feed intent by tool."),
    ("2 — Material Candidates", "Approve or reject inferred material matches."),
    ("3 — Tooling Review", "Compare parsed tool descriptions against the shop reference. Save corrections."),
    ("4 — Program Search", "Search raw cut records by program, tool, S/F value, or keyword."),
    ("5 — Machine Overview", "High-level intelligence summary per machine."),
    ("6 — Analytics", "Shop-wide patterns: top tools, S/F distributions, feed intent frequency."),
]

cols = st.columns(3)
for i, (title, desc) in enumerate(nav_items):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div style="background:#1C1C1C;border:1px solid #2D2D2D;border-left:3px solid #FF6B35;
                        border-radius:4px;padding:12px 14px;margin-bottom:10px;">
              <div style="color:#FAFAFA;font-weight:700;font-size:0.85rem;">{title}</div>
              <div style="color:#888;font-size:0.78rem;margin-top:4px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "READ ONLY against P:\\  |  All corrections saved locally to data/overrides/  |  "
    "Never modifies source CNC programs"
)
