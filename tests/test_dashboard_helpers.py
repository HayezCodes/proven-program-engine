"""
test_dashboard_helpers.py — Tests for the dashboard utilities/helpers module.
"""

import pandas as pd
import pytest

from src.dashboard.utils.helpers import (
    safe_list,
    filter_df,
    text_search,
    fmt_sfm,
    fmt_feed_ipr,
    fmt_feed_ipm,
    fmt_range,
    top_n,
    count_by,
    summarize_sf,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "machine_folder": ["655", "655", "432", "654"],
        "tool_number": [1, 2, 1, 3],
        "s_mean": [600.0, 450.0, None, 500.0],
        "f_mean": [0.016, 0.010, 15.0, 0.012],
        "confidence_label": ["HIGH", "MEDIUM", "NONE", "HIGH"],
        "material_candidate_1": ["4140", "1018", None, "4140"],
        "record_count": [10, 5, 20, 8],
    })


# ---------------------------------------------------------------------------
# safe_list
# ---------------------------------------------------------------------------

class TestSafeList:
    def test_returns_sorted_unique_values(self, sample_df):
        result = safe_list(sample_df, "machine_folder")
        assert result == ["432", "654", "655"]

    def test_excludes_nan(self, sample_df):
        result = safe_list(sample_df, "material_candidate_1")
        assert None not in result
        assert "None" not in result

    def test_empty_df_returns_empty(self):
        assert safe_list(pd.DataFrame(), "col") == []

    def test_none_df_returns_empty(self):
        assert safe_list(None, "col") == []

    def test_missing_column_returns_empty(self, sample_df):
        assert safe_list(sample_df, "nonexistent") == []

    def test_no_sort(self, sample_df):
        result = safe_list(sample_df, "machine_folder", sort=False)
        assert set(result) == {"432", "654", "655"}


# ---------------------------------------------------------------------------
# filter_df
# ---------------------------------------------------------------------------

class TestFilterDf:
    def test_single_equality_filter(self, sample_df):
        result = filter_df(sample_df, machine_folder="655")
        assert len(result) == 2
        assert all(result["machine_folder"] == "655")

    def test_list_filter(self, sample_df):
        result = filter_df(sample_df, machine_folder=["655", "432"])
        assert set(result["machine_folder"].tolist()) == {"655", "432"}

    def test_empty_list_no_filter(self, sample_df):
        result = filter_df(sample_df, machine_folder=[])
        assert len(result) == len(sample_df)

    def test_none_value_no_filter(self, sample_df):
        result = filter_df(sample_df, machine_folder=None)
        assert len(result) == len(sample_df)

    def test_multiple_filters(self, sample_df):
        result = filter_df(sample_df, machine_folder="655", confidence_label="HIGH")
        assert len(result) == 1
        assert result.iloc[0]["tool_number"] == 1

    def test_missing_column_ignored(self, sample_df):
        result = filter_df(sample_df, nonexistent_col="value")
        assert len(result) == len(sample_df)

    def test_empty_df_returns_empty(self):
        result = filter_df(pd.DataFrame(), machine_folder="655")
        assert result.empty

    def test_none_df_returns_empty(self):
        result = filter_df(None)
        assert result.empty


# ---------------------------------------------------------------------------
# text_search
# ---------------------------------------------------------------------------

class TestTextSearch:
    def test_finds_exact_match(self, sample_df):
        result = text_search(sample_df, "655", ["machine_folder"])
        assert len(result) == 2

    def test_case_insensitive(self):
        df = pd.DataFrame({"desc": ["1/2 END MILL", "WOODRUFF", "DRILL"]})
        result = text_search(df, "end mill", ["desc"])
        assert len(result) == 1

    def test_partial_match(self):
        df = pd.DataFrame({"desc": ["1/2 END MILL", "1/4 END MILL", "DRILL"]})
        result = text_search(df, "end mill", ["desc"])
        assert len(result) == 2

    def test_empty_query_returns_all(self, sample_df):
        result = text_search(sample_df, "", ["machine_folder"])
        assert len(result) == len(sample_df)

    def test_no_match_returns_empty(self, sample_df):
        result = text_search(sample_df, "ZZZNOMATCH", ["machine_folder"])
        assert result.empty

    def test_searches_multiple_columns(self):
        df = pd.DataFrame({
            "col_a": ["alpha", "beta", "gamma"],
            "col_b": ["one", "two", "three"],
        })
        result = text_search(df, "two", ["col_a", "col_b"])
        assert len(result) == 1

    def test_none_df_returns_empty(self):
        result = text_search(None, "query", ["col"])
        assert result.empty


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

class TestFmtSfm:
    def test_integer_value(self):
        assert fmt_sfm(600.0) == "600 SFM"

    def test_rounds_correctly(self):
        assert fmt_sfm(600.6) == "601 SFM"

    def test_none_returns_dash(self):
        assert fmt_sfm(None) == "—"

    def test_string_number(self):
        assert fmt_sfm("500") == "500 SFM"

    def test_non_numeric_returns_dash(self):
        assert fmt_sfm("n/a") == "—"


class TestFmtFeedIpr:
    def test_four_decimal_places(self):
        assert fmt_feed_ipr(0.016) == "0.0160 IPR"

    def test_none_returns_dash(self):
        assert fmt_feed_ipr(None) == "—"


class TestFmtFeedIpm:
    def test_two_decimal_places(self):
        assert fmt_feed_ipm(15.0) == "15.00 IPM"

    def test_none_returns_dash(self):
        assert fmt_feed_ipm(None) == "—"


class TestFmtRange:
    def test_formats_range(self):
        result = fmt_range(450.0, 600.0, "SFM")
        assert "450" in result
        assert "600" in result
        assert "SFM" in result

    def test_none_returns_dash(self):
        assert fmt_range(None, 600.0) == "—"


# ---------------------------------------------------------------------------
# top_n
# ---------------------------------------------------------------------------

class TestTopN:
    def test_returns_top_n_rows(self, sample_df):
        result = top_n(sample_df, "record_count", n=2)
        assert len(result) == 2
        assert result.iloc[0]["record_count"] >= result.iloc[1]["record_count"]

    def test_empty_df_returns_empty(self):
        assert top_n(pd.DataFrame(), "col", n=5).empty

    def test_none_returns_empty(self):
        assert top_n(None, "col").empty

    def test_n_larger_than_rows_returns_all(self, sample_df):
        result = top_n(sample_df, "record_count", n=100)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# count_by
# ---------------------------------------------------------------------------

class TestCountBy:
    def test_counts_values(self, sample_df):
        result = count_by(sample_df, "machine_folder")
        assert "machine_folder" in result.columns
        assert "count" in result.columns
        row_655 = result[result["machine_folder"] == "655"]
        assert row_655.iloc[0]["count"] == 2

    def test_empty_df_returns_empty(self):
        result = count_by(pd.DataFrame(), "col")
        assert result.empty


# ---------------------------------------------------------------------------
# summarize_sf
# ---------------------------------------------------------------------------

class TestSummarizeSf:
    def test_computes_median_s_mean(self, sample_df):
        result = summarize_sf(sample_df)
        assert "s_mean" in result

    def test_returns_empty_for_empty_df(self):
        assert summarize_sf(pd.DataFrame()) == {}

    def test_returns_empty_for_none(self):
        assert summarize_sf(None) == {}

    def test_ignores_nan(self, sample_df):
        result = summarize_sf(sample_df)
        assert result.get("s_mean") is not None
