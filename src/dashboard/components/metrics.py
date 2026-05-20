"""
metrics.py — Metric card rendering helpers.
"""

import streamlit as st


def metric_row(items: list[dict]) -> None:
    """Render a horizontal row of metric cards.

    Each item dict: {label, value, delta=None, help=None}
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                label=item.get("label", ""),
                value=item.get("value", "—"),
                delta=item.get("delta"),
                help=item.get("help"),
            )


def status_indicator(found: bool, label: str) -> str:
    """Return a text status indicator string."""
    return f"{'OK' if found else 'MISSING'}  {label}"


def export_status_display(status: dict) -> None:
    """Render export file availability as a compact status table."""
    labels = {
        "cuts": "Cuts (raw parser output)",
        "tool_summary": "Tool Summary",
        "material_candidates": "Material Candidates",
        "tooling_review": "Tooling Review",
    }

    for key, info in status.items():
        label = labels.get(key, key)
        if info["found"]:
            st.success(
                f"**{label}** — {info['filename']}  "
                f"| {info['modified']}  | {info['size_kb']} KB",
                icon="OK",
            )
        else:
            st.warning(f"**{label}** — not found. Run the appropriate pipeline step.", icon="WARN")
