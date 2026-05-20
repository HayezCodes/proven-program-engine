"""
test_exports.py — Tests for the exports module (parser summary + tool summary).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.exports import (
    build_parser_summary,
    build_tool_summary,
    export_parser_summary,
    export_tool_summary,
)
from src.parser import parse_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(
    tool_num="01", t_code="T0101", machine="421", s_mode="RPM",
    s_val=450.0, f_val=0.015, confidence="HIGH", program_id=1,
    s_type="SPINDLE", f_mode="IPR",
) -> dict:
    return {
        "tool_number": tool_num,
        "active_t_code": t_code,
        "machine_folder": machine,
        "s_mode": s_mode,
        "s_value": s_val,
        "s_type": s_type,
        "f_value": f_val,
        "f_mode": f_mode,
        "extraction_confidence": confidence,
        "program_id": program_id,
        "block_skip": False,
        "is_duplicate": False,
        "sf_combo_count": 1,
    }


@pytest.fixture
def sample_records():
    return [
        _make_record("01", "T0101", "421", "RPM", 450.0, 0.015, "HIGH",   1),
        _make_record("01", "T0101", "421", "RPM", 450.0, 0.015, "HIGH",   2),
        _make_record("02", "T0202", "421", "CSS", 600.0, 0.006, "HIGH",   1),
        _make_record("03", "T0303", "432", "RPM", 1200.0, 0.004, "MEDIUM", 3),
        _make_record("",   "",      "432", "",    None,  0.010, "LOW",    4, s_type="", f_mode="UNKNOWN"),
    ]


@pytest.fixture
def real_file_records(sample_nc_file):
    """Records parsed from the real sample .NC fixture."""
    from src.parser import _annotate_duplicates
    recs = parse_file(sample_nc_file, program_id=1)
    _annotate_duplicates(recs)
    return recs


# ---------------------------------------------------------------------------
# build_parser_summary
# ---------------------------------------------------------------------------

class TestBuildParserSummary:
    def test_returns_dict(self, sample_records):
        result = build_parser_summary(sample_records)
        assert isinstance(result, dict)

    def test_total_records_correct(self, sample_records):
        result = build_parser_summary(sample_records)
        assert result["total_records"] == len(sample_records)

    def test_total_s_values_correct(self, sample_records):
        result = build_parser_summary(sample_records)
        s_count = sum(1 for r in sample_records if r.get("s_value") is not None)
        assert result["total_s_values"] == s_count

    def test_total_f_values_correct(self, sample_records):
        result = build_parser_summary(sample_records)
        f_count = sum(1 for r in sample_records if r.get("f_value") is not None)
        assert result["total_f_values"] == f_count

    def test_records_missing_t_code_correct(self, sample_records):
        result = build_parser_summary(sample_records)
        orphan_count = sum(1 for r in sample_records if not r.get("active_t_code"))
        assert result["records_missing_t_code"] == orphan_count

    def test_unique_tool_numbers_correct(self, sample_records):
        result = build_parser_summary(sample_records)
        unique = len({r["tool_number"] for r in sample_records if r.get("tool_number")})
        assert result["unique_tool_numbers"] == unique

    def test_confidence_distribution_present(self, sample_records):
        result = build_parser_summary(sample_records)
        assert "confidence_distribution" in result
        assert isinstance(result["confidence_distribution"], dict)

    def test_s_mode_distribution_present(self, sample_records):
        result = build_parser_summary(sample_records)
        assert "s_mode_distribution" in result

    def test_f_mode_distribution_present(self, sample_records):
        result = build_parser_summary(sample_records)
        assert "f_mode_distribution" in result

    def test_top_tools_is_list(self, sample_records):
        result = build_parser_summary(sample_records)
        assert isinstance(result["top_tools_by_record_count"], list)

    def test_empty_records_returns_minimal_dict(self):
        result = build_parser_summary([])
        assert result["total_records"] == 0

    def test_summary_is_json_serialisable(self, sample_records):
        result = build_parser_summary(sample_records)
        serialised = json.dumps(result)
        assert len(serialised) > 0

    def test_real_file_produces_summary(self, real_file_records):
        result = build_parser_summary(real_file_records)
        assert result["total_records"] > 0
        assert result["total_s_values"] > 0
        assert result["total_f_values"] > 0


# ---------------------------------------------------------------------------
# build_tool_summary
# ---------------------------------------------------------------------------

class TestBuildToolSummary:
    def test_returns_dataframe(self, sample_records):
        df = build_tool_summary(sample_records)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, sample_records):
        df = build_tool_summary(sample_records)
        required = {
            "tool_number", "active_t_code", "machine_folder", "s_mode",
            "s_count", "s_mean", "s_min", "s_max",
            "f_count", "f_mean", "f_min", "f_max",
            "record_count", "unique_program_count",
        }
        assert required.issubset(df.columns)

    def test_groups_by_tool_machine_mode(self, sample_records):
        df = build_tool_summary(sample_records)
        # T0101/421/RPM and T0202/421/CSS should be distinct groups
        t01_rpm = df[(df["active_t_code"] == "T0101") & (df["s_mode"] == "RPM")]
        t02_css = df[(df["active_t_code"] == "T0202") & (df["s_mode"] == "CSS")]
        assert len(t01_rpm) == 1
        assert len(t02_css) == 1

    def test_s_mean_computed_correctly(self, sample_records):
        df = build_tool_summary(sample_records)
        t01 = df[df["active_t_code"] == "T0101"].iloc[0]
        assert t01["s_mean"] == 450.0

    def test_f_min_max_computed(self, sample_records):
        df = build_tool_summary(sample_records)
        t01 = df[df["active_t_code"] == "T0101"].iloc[0]
        assert t01["f_min"] == 0.015
        assert t01["f_max"] == 0.015

    def test_occurrence_count_summed(self, sample_records):
        df = build_tool_summary(sample_records)
        t01 = df[df["active_t_code"] == "T0101"].iloc[0]
        assert t01["record_count"] == 2  # two T0101 records in sample_records

    def test_unique_program_count_counted(self, sample_records):
        df = build_tool_summary(sample_records)
        t01 = df[df["active_t_code"] == "T0101"].iloc[0]
        assert t01["unique_program_count"] == 2  # program_ids 1 and 2

    def test_orphan_records_excluded(self, sample_records):
        df = build_tool_summary(sample_records)
        assert "" not in df["active_t_code"].values

    def test_empty_records_returns_empty_df(self):
        df = build_tool_summary([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_real_file_produces_tool_summary(self, real_file_records):
        df = build_tool_summary(real_file_records)
        assert len(df) > 0
        assert "T0101" in df["active_t_code"].values


# ---------------------------------------------------------------------------
# export functions
# ---------------------------------------------------------------------------

class TestExportFunctions:
    def test_export_parser_summary_creates_json_file(self, sample_records, tmp_path):
        path = export_parser_summary(sample_records, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_exported_json_is_valid(self, sample_records, tmp_path):
        path = export_parser_summary(sample_records, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["total_records"] == len(sample_records)

    def test_export_tool_summary_creates_csv_file(self, sample_records, tmp_path):
        path = export_tool_summary(sample_records, tmp_path)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_exported_tool_summary_readable(self, sample_records, tmp_path):
        path = export_tool_summary(sample_records, tmp_path)
        df = pd.read_csv(path)
        assert len(df) > 0

    def test_timestamp_suffix_in_filename(self, sample_records, tmp_path):
        path = export_parser_summary(sample_records, tmp_path, timestamp="20260101_120000")
        assert "20260101_120000" in path.name
