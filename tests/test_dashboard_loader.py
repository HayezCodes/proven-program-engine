"""
test_dashboard_loader.py — Tests for the dashboard data access loader module.
"""

import time
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard.data_access.loader import (
    _find_latest,
    _load_csv,
    load_latest_cuts,
    load_latest_tool_summary,
    load_latest_material_candidates,
    load_latest_tooling_review,
    load_latest_tooldb_reference,
    get_export_status,
    build_proven_tools_df,
)


# ---------------------------------------------------------------------------
# _find_latest
# ---------------------------------------------------------------------------

class TestFindLatest:
    def test_returns_none_for_empty_dir(self, tmp_path):
        assert _find_latest(tmp_path, "cuts_*.csv") is None

    def test_returns_none_when_no_match(self, tmp_path):
        (tmp_path / "other.csv").write_text("x")
        assert _find_latest(tmp_path, "cuts_*.csv") is None

    def test_returns_single_match(self, tmp_path):
        f = tmp_path / "cuts_20260101_120000.csv"
        f.write_text("a,b\n1,2")
        assert _find_latest(tmp_path, "cuts_*.csv") == f

    def test_returns_most_recent_by_mtime(self, tmp_path):
        f1 = tmp_path / "cuts_20260101_120000.csv"
        f1.write_text("a")
        time.sleep(0.02)
        f2 = tmp_path / "cuts_20260102_120000.csv"
        f2.write_text("b")
        assert _find_latest(tmp_path, "cuts_*.csv") == f2

    def test_pattern_specificity(self, tmp_path):
        (tmp_path / "cuts_20260101.csv").write_text("a")
        (tmp_path / "tool_summary_20260101.csv").write_text("b")
        result = _find_latest(tmp_path, "tool_summary_*.csv")
        assert result is not None
        assert "tool_summary" in result.name


# ---------------------------------------------------------------------------
# _load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def test_returns_none_for_none_path(self):
        assert _load_csv(None) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _load_csv(tmp_path / "missing.csv") is None

    def test_loads_valid_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4")
        df = _load_csv(f)
        assert df is not None
        assert len(df) == 2

    def test_returns_dataframe(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\nx,y")
        result = _load_csv(f)
        assert isinstance(result, pd.DataFrame)

    def test_empty_csv_returns_empty_df(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("col1,col2\n")
        result = _load_csv(f)
        assert result is not None
        assert len(result) == 0


# ---------------------------------------------------------------------------
# load_latest_* with custom exports_dir
# ---------------------------------------------------------------------------

class TestLoadLatestCuts:
    def test_returns_none_when_no_file(self, tmp_path):
        assert load_latest_cuts(tmp_path) is None

    def test_returns_dataframe_when_file_exists(self, tmp_path):
        f = tmp_path / "cuts_20260101_120000.csv"
        f.write_text("machine_folder,tool_number,s_value\n655,1,600")
        result = load_latest_cuts(tmp_path)
        assert result is not None
        assert "machine_folder" in result.columns

    def test_picks_latest_of_multiple(self, tmp_path):
        (tmp_path / "cuts_20260101_120000.csv").write_text("col\nold")
        time.sleep(0.02)
        (tmp_path / "cuts_20260102_120000.csv").write_text("col\nnew")
        result = load_latest_cuts(tmp_path)
        assert result.iloc[0]["col"] == "new"


class TestLoadLatestToolSummary:
    def test_returns_none_when_missing(self, tmp_path):
        assert load_latest_tool_summary(tmp_path) is None

    def test_returns_df_when_present(self, tmp_path):
        f = tmp_path / "tool_summary_20260101.csv"
        f.write_text("tool_number,record_count\n1,10")
        assert load_latest_tool_summary(tmp_path) is not None


class TestLoadLatestMaterialCandidates:
    def test_returns_none_when_missing(self, tmp_path):
        assert load_latest_material_candidates(tmp_path) is None

    def test_returns_df_when_present(self, tmp_path):
        f = tmp_path / "material_candidates_20260101.csv"
        f.write_text("machine_folder,confidence_label\n655,HIGH")
        assert load_latest_material_candidates(tmp_path) is not None


class TestLoadLatestToolingReview:
    def test_returns_none_when_missing(self, tmp_path):
        assert load_latest_tooling_review(tmp_path) is None

    def test_returns_df_when_present(self, tmp_path):
        f = tmp_path / "tooling_review_20260101.csv"
        f.write_text("machine_folder,match_status\n655,description_match")
        assert load_latest_tooling_review(tmp_path) is not None


# ---------------------------------------------------------------------------
# load_latest_tooldb_reference
# ---------------------------------------------------------------------------

class TestLoadLatestTooldbReference:
    def test_returns_none_when_missing(self, tmp_path):
        assert load_latest_tooldb_reference(tmp_path) is None

    def test_returns_df_when_present(self, tmp_path):
        f = tmp_path / "tooldb_reference_20260101_120000.csv"
        f.write_text("machine_id,tool_number,tool_name\n655,1,ENDMILL")
        result = load_latest_tooldb_reference(tmp_path)
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["tool_name"] == "ENDMILL"

    def test_returns_most_recent_when_multiple(self, tmp_path):
        f1 = tmp_path / "tooldb_reference_20260101_000000.csv"
        f1.write_text("machine_id,tool_number,tool_name\n655,1,OLD")
        time.sleep(0.02)
        f2 = tmp_path / "tooldb_reference_20260102_000000.csv"
        f2.write_text("machine_id,tool_number,tool_name\n655,1,NEW")
        result = load_latest_tooldb_reference(tmp_path)
        assert result.iloc[0]["tool_name"] == "NEW"

    def test_tooldb_reference_in_export_status(self, tmp_path):
        (tmp_path / "tooldb_reference_20260101_120000.csv").write_text("a,b\n1,2")
        status = get_export_status(tmp_path)
        assert "tooldb_reference" in status
        assert status["tooldb_reference"]["found"] is True


# ---------------------------------------------------------------------------
# get_export_status
# ---------------------------------------------------------------------------

class TestGetExportStatus:
    def test_all_missing_when_empty_dir(self, tmp_path):
        status = get_export_status(tmp_path)
        for key in ("cuts", "tool_summary", "material_candidates", "tooling_review"):
            assert status[key]["found"] is False

    def test_found_when_file_present(self, tmp_path):
        (tmp_path / "cuts_20260101_120000.csv").write_text("a,b\n1,2")
        status = get_export_status(tmp_path)
        assert status["cuts"]["found"] is True
        assert status["cuts"]["filename"] == "cuts_20260101_120000.csv"

    def test_has_required_keys(self, tmp_path):
        status = get_export_status(tmp_path)
        for key in ("cuts", "tool_summary"):
            assert "found" in status[key]
            assert "filename" in status[key]
            assert "size_kb" in status[key]

    def test_size_kb_nonzero_when_found(self, tmp_path):
        f = tmp_path / "cuts_20260101.csv"
        f.write_text("a,b\n" + "1,2\n" * 100)
        status = get_export_status(tmp_path)
        assert status["cuts"]["size_kb"] > 0


# ---------------------------------------------------------------------------
# build_proven_tools_df
# ---------------------------------------------------------------------------

class TestBuildProvenToolsDf:
    def test_returns_empty_for_none(self):
        result = build_proven_tools_df(None)
        assert result.empty

    def test_returns_empty_for_empty_df(self):
        result = build_proven_tools_df(pd.DataFrame())
        assert result.empty

    def test_returns_df_without_tooling_review(self):
        mc = pd.DataFrame({
            "machine_folder": ["655"],
            "tool_number": [1],
            "active_t_code": ["T01"],
            "s_mean": [600.0],
        })
        result = build_proven_tools_df(mc)
        assert len(result) == 1

    def test_merges_tooling_review(self):
        mc = pd.DataFrame({
            "machine_folder": ["655", "655"],
            "tool_number": [1, 2],
            "active_t_code": ["T01", "T02"],
            "s_mean": [600.0, 500.0],
        })
        tr = pd.DataFrame({
            "machine_folder": ["655"],
            "tool_number": [1],
            "match_status": ["description_match"],
            "reference_description": ["1/2"],
            "decimal_size": [0.5],
        })
        result = build_proven_tools_df(mc, tr)
        assert len(result) == 2
        row_t1 = result[result["tool_number"] == 1].iloc[0]
        assert row_t1["match_status"] == "description_match"

    def test_unmatched_rows_have_null_ref_cols(self):
        mc = pd.DataFrame({
            "machine_folder": ["655"],
            "tool_number": [99],
            "active_t_code": ["T99"],
            "s_mean": [400.0],
        })
        tr = pd.DataFrame({
            "machine_folder": ["655"],
            "tool_number": [1],
            "match_status": ["description_match"],
            "reference_description": ["1/2"],
            "decimal_size": [0.5],
        })
        result = build_proven_tools_df(mc, tr)
        assert result.iloc[0]["match_status"] is None or pd.isna(result.iloc[0]["match_status"])
