"""
filters.py — Reusable Streamlit sidebar filter components.

Each function renders UI controls in the sidebar and returns a filtered DataFrame.
"""

from typing import Optional
import pandas as pd
import streamlit as st

from src.dashboard.utils.helpers import safe_list, filter_df, text_search


def _sidebar_multiselect(label: str, options: list, key: str) -> list:
    """Render a sidebar multiselect. Returns selected values or [] for 'all'."""
    if not options:
        return []
    all_label = f"All ({len(options)})"
    choices = [all_label] + options
    selected = st.sidebar.multiselect(label, choices, default=[all_label], key=key)
    if all_label in selected or not selected:
        return []
    return selected


def machine_filter(df: pd.DataFrame, key: str = "filter_machine") -> pd.DataFrame:
    """Sidebar filter by machine_folder."""
    machines = safe_list(df, "machine_folder")
    selected = _sidebar_multiselect("Machine", machines, key)
    return filter_df(df, machine_folder=selected) if selected else df


def tool_filter(df: pd.DataFrame, key: str = "filter_tool") -> pd.DataFrame:
    """Sidebar filter by tool_number."""
    tools = safe_list(df, "tool_number")
    selected = _sidebar_multiselect("Tool Number", tools, key)
    return filter_df(df, tool_number=selected) if selected else df


def smode_filter(df: pd.DataFrame, key: str = "filter_smode") -> pd.DataFrame:
    """Sidebar filter by spindle mode (s_mode)."""
    modes = safe_list(df, "s_mode")
    selected = _sidebar_multiselect("Spindle Mode", modes, key)
    return filter_df(df, s_mode=selected) if selected else df


def confidence_filter(df: pd.DataFrame, key: str = "filter_conf") -> pd.DataFrame:
    """Sidebar filter by confidence_label."""
    labels = safe_list(df, "confidence_label")
    selected = _sidebar_multiselect("Confidence", labels, key)
    return filter_df(df, confidence_label=selected) if selected else df


def match_type_filter(df: pd.DataFrame, key: str = "filter_match") -> pd.DataFrame:
    """Sidebar filter by match_type."""
    types = safe_list(df, "match_type")
    selected = _sidebar_multiselect("Match Type", types, key)
    return filter_df(df, match_type=selected) if selected else df


def match_status_filter(df: pd.DataFrame, key: str = "filter_status") -> pd.DataFrame:
    """Sidebar filter by match_status (tooling review)."""
    statuses = safe_list(df, "match_status")
    selected = _sidebar_multiselect("Match Status", statuses, key)
    return filter_df(df, match_status=selected) if selected else df


def material_candidate_filter(df: pd.DataFrame, key: str = "filter_mat") -> pd.DataFrame:
    """Sidebar filter by material_candidate_1."""
    mats = safe_list(df, "material_candidate_1")
    selected = _sidebar_multiselect("Material Candidate", mats, key)
    return filter_df(df, material_candidate_1=selected) if selected else df


def feed_intent_filter(df: pd.DataFrame, key: str = "filter_intent") -> pd.DataFrame:
    """Sidebar filter by feed_intent_candidate."""
    intents = safe_list(df, "feed_intent_candidate")
    selected = _sidebar_multiselect("Feed Intent", intents, key)
    return filter_df(df, feed_intent_candidate=selected) if selected else df


def sidebar_text_search(
    df: pd.DataFrame,
    columns: list[str],
    label: str = "Search",
    key: str = "search_text",
) -> pd.DataFrame:
    """Sidebar text search across given columns."""
    query = st.sidebar.text_input(label, key=key)
    return text_search(df, query, columns) if query else df
