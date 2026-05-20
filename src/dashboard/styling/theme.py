"""
theme.py — Dark industrial CSS for the Proven Program Engine dashboard.
"""

_CSS = """
<style>
/* ── Global ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
.ppe-metric-card {
    background: #1C1C1C;
    border: 1px solid #2D2D2D;
    border-left: 3px solid #FF6B35;
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.ppe-metric-label {
    color: #888888;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.ppe-metric-value {
    color: #FAFAFA;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.1;
}
.ppe-metric-delta {
    color: #FF6B35;
    font-size: 0.75rem;
    margin-top: 2px;
}

/* ── Status badges ───────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-high    { background: #1a3a2a; color: #00C851; border: 1px solid #00C851; }
.badge-medium  { background: #3a2a10; color: #FFB347; border: 1px solid #FFB347; }
.badge-low     { background: #3a1a1a; color: #FF6B6B; border: 1px solid #FF6B6B; }
.badge-none    { background: #2a2a2a; color: #888888; border: 1px solid #444444; }
.badge-ok      { background: #1a3a2a; color: #00C851; border: 1px solid #00C851; }
.badge-warn    { background: #3a3a10; color: #FFDD47; border: 1px solid #FFDD47; }
.badge-err     { background: #3a1010; color: #FF4444; border: 1px solid #FF4444; }
.badge-info    { background: #0d2a3a; color: #00B4D8; border: 1px solid #00B4D8; }

/* ── Section headers ─────────────────────────────────────────────────────── */
.ppe-section-header {
    border-left: 3px solid #FF6B35;
    padding-left: 10px;
    color: #FAFAFA;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 16px 0 8px 0;
}

/* ── Override / status banner ───────────────────────────────────────────── */
.ppe-banner {
    padding: 8px 14px;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-bottom: 12px;
}
.ppe-banner-warn { background: #2a2510; border: 1px solid #554400; color: #FFDD47; }
.ppe-banner-info { background: #0d1a25; border: 1px solid #004466; color: #00B4D8; }
.ppe-banner-ok   { background: #0d2010; border: 1px solid #004420; color: #00C851; }

/* ── Dataframe styling ───────────────────────────────────────────────────── */
.stDataFrame { border: 1px solid #2D2D2D !important; border-radius: 4px; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #2D2D2D;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
details {
    background: #1C1C1C !important;
    border: 1px solid #2D2D2D !important;
    border-radius: 4px !important;
}

/* ── Download button ─────────────────────────────────────────────────────── */
.stDownloadButton button {
    background: transparent !important;
    border: 1px solid #FF6B35 !important;
    color: #FF6B35 !important;
    font-size: 0.78rem !important;
}
.stDownloadButton button:hover {
    background: #FF6B35 !important;
    color: #000000 !important;
}
</style>
"""


def apply_theme() -> None:
    """Inject the dark industrial theme CSS. Call once at the top of each page."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, level: str = "info") -> str:
    """Return a coloured badge HTML span. level: high/medium/low/none/ok/warn/err/info."""
    return f'<span class="badge badge-{level.lower()}">{text}</span>'


def confidence_badge(label: str) -> str:
    """Map confidence label to badge HTML."""
    mapping = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low", "NONE": "none"}
    return badge(label, mapping.get(label.upper(), "info"))


def match_status_badge(status: str) -> str:
    """Map tooling match_status to badge HTML."""
    mapping = {
        "description_match": ("MATCH", "ok"),
        "description_differs": ("DIFFERS", "warn"),
        "missing_from_reference": ("MISSING", "err"),
        "no_description_in_reference": ("NO REF DESC", "warn"),
        "no_program_description": ("NO PROG DESC", "medium"),
        "no_reference_data": ("NO REF", "none"),
        "needs_review": ("NEEDS REVIEW", "warn"),
    }
    text, level = mapping.get(status, (status, "info"))
    return badge(text, level)
