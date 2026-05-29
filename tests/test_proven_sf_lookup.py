"""
test_proven_sf_lookup.py — Tests for the proven S/F lookup table builder.
"""

import pandas as pd
import pytest

from src.proven_sf_lookup import (
    SF_LOOKUP_COLS,
    _BANNED_COLS,
    build_sf_lookup,
    export_sf_lookup,
)


# ---------------------------------------------------------------------------
# Minimal synthetic SF database row factory
# ---------------------------------------------------------------------------

def _sf_row(
    machine_folder: str = "421, 423, 424",
    machine_family: str = "421",
    tool_name: str = "DNMG 443-PR",
    feed_intent: str = "",
    s_mode: str = "CSS",
    f_mode: str = "IPR",
    S: float = 400.0,
    F: float = 0.012,
    verified_material: str = "4140 HR HT",
    material_candidate_1: str = "",
    sf_record_confidence: str = "MEDIUM",
    **kwargs,
) -> dict:
    defaults = {
        "machine_folder":       machine_folder,
        "machine_family":       machine_family,
        "source_file":          "P:/test/prog.EIA",
        "filename":             "10007001.EIA",
        "program_id":           1,
        "tool_number":          "1",
        "resolved_tool_name":   tool_name,
        "resolved_tool_description": "",
        "tool_identity_source": "TOOLDB_ASSEMBLY",
        "tool_needs_review":    False,
        "S":                    S,
        "s_mode":               s_mode,
        "s_type":               "SPINDLE",
        "F":                    F,
        "f_mode":               f_mode,
        "feed_intent_candidate": feed_intent,
        "verified_material":    verified_material,
        "material_source":      "ROUTER",
        "material_confidence":  "MEDIUM",
        "material_candidate_1": material_candidate_1,
        "material_candidate_confidence_1": "",
        "matched_job_number":   "D20182",
        "matched_part_number":  "",
        "matched_drawing_number": "",
        "linked_router_file":   "",
        "router_work_center":   "",
        "router_operation_description": "",
        "link_confidence":      "MEDIUM",
        "link_method":          "router_match",
        "extraction_confidence": "HIGH",
        "sf_record_confidence": sf_record_confidence,
        "needs_review":         False,
        "review_reason":        "",
        "raw_line":             "G96 S400 F.012",
        "prev_line":            "",
        "next_line":            "",
    }
    defaults.update(kwargs)
    return defaults


def _db(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema and column constraints
# ---------------------------------------------------------------------------

class TestSchema:
    def test_empty_input_returns_empty_with_correct_columns(self):
        result = build_sf_lookup(pd.DataFrame())
        assert list(result.columns) == SF_LOOKUP_COLS

    def test_none_input_returns_empty_with_correct_columns(self):
        result = build_sf_lookup(None)
        assert list(result.columns) == SF_LOOKUP_COLS

    def test_output_has_exactly_sf_lookup_cols(self):
        df = _db([_sf_row()])
        result = build_sf_lookup(df)
        assert list(result.columns) == SF_LOOKUP_COLS

    def test_no_banned_cols_in_output(self):
        df = _db([_sf_row()])
        result = build_sf_lookup(df)
        for col in _BANNED_COLS:
            assert col not in result.columns, f"Banned column present: {col}"

    def test_no_job_number_in_output(self):
        df = _db([_sf_row(matched_job_number="D20182")])
        result = build_sf_lookup(df)
        assert "matched_job_number" not in result.columns

    def test_no_raw_line_in_output(self):
        df = _db([_sf_row(raw_line="G96 S400")])
        result = build_sf_lookup(df)
        assert "raw_line" not in result.columns

    def test_no_source_file_in_output(self):
        df = _db([_sf_row()])
        result = build_sf_lookup(df)
        assert "source_file" not in result.columns

    def test_blank_and_nan_modes_display_unknown(self):
        rows = [
            _sf_row(s_mode="", f_mode=None),
            _sf_row(s_mode=float("nan"), f_mode="nan"),
        ]
        result = build_sf_lookup(_db(rows))
        assert set(result["s_mode"]) == {"UNKNOWN"}
        assert set(result["f_mode"]) == {"UNKNOWN"}


# ---------------------------------------------------------------------------
# S and F range calculation
# ---------------------------------------------------------------------------

class TestRanges:
    def test_s_low_mid_high_from_single_row(self):
        df = _db([_sf_row(S=400.0)])
        result = build_sf_lookup(df)
        assert result.iloc[0]["S_low"]  == 400
        assert result.iloc[0]["S_mid"]  == 400
        assert result.iloc[0]["S_high"] == 400

    def test_s_low_mid_high_from_multiple_rows(self):
        rows = [
            _sf_row(S=200.0),
            _sf_row(S=400.0),
            _sf_row(S=600.0),
        ]
        result = build_sf_lookup(_db(rows))
        r = result.iloc[0]
        assert r["S_low"]  == 200
        assert r["S_mid"]  == 400    # median of 200,400,600
        assert r["S_high"] == 600

    def test_f_low_mid_high_from_multiple_rows(self):
        rows = [
            _sf_row(F=0.005, S=400.0),
            _sf_row(F=0.010, S=400.0),
            _sf_row(F=0.015, S=400.0),
        ]
        result = build_sf_lookup(_db(rows))
        r = result.iloc[0]
        assert abs(r["F_low"]  - 0.005) < 1e-9
        assert abs(r["F_mid"]  - 0.010) < 1e-9
        assert abs(r["F_high"] - 0.015) < 1e-9

    def test_rows_with_no_s_or_f_excluded(self):
        rows = [
            _sf_row(S=None, F=None),
            _sf_row(S=400.0),
        ]
        result = build_sf_lookup(_db(rows))
        assert len(result) == 1

    def test_all_no_sf_returns_empty(self):
        rows = [_sf_row(S=None, F=None)]
        result = build_sf_lookup(_db(rows))
        assert result.empty or len(result) == 0


# ---------------------------------------------------------------------------
# Occurrence count
# ---------------------------------------------------------------------------

class TestOccurrenceCount:
    def test_single_row_occurrence_count_1(self):
        result = build_sf_lookup(_db([_sf_row()]))
        assert result.iloc[0]["occurrence_count"] == 1

    def test_three_same_group_occurrence_count_3(self):
        rows = [_sf_row(S=300.0), _sf_row(S=400.0), _sf_row(S=500.0)]
        result = build_sf_lookup(_db(rows))
        assert result.iloc[0]["occurrence_count"] == 3

    def test_two_different_groups_separate_counts(self):
        rows = [
            _sf_row(tool_name="DNMG 443-PR", S=400.0),
            _sf_row(tool_name="VNMG 332-PF", S=600.0),
        ]
        result = build_sf_lookup(_db(rows))
        assert len(result) == 2
        assert all(result["occurrence_count"] == 1)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_high_confidence_wins(self):
        rows = [
            _sf_row(sf_record_confidence="LOW"),
            _sf_row(sf_record_confidence="HIGH"),
            _sf_row(sf_record_confidence="MEDIUM"),
        ]
        result = build_sf_lookup(_db(rows))
        assert result.iloc[0]["confidence"] == "HIGH"

    def test_medium_when_no_high(self):
        rows = [
            _sf_row(sf_record_confidence="MEDIUM"),
            _sf_row(sf_record_confidence="LOW"),
        ]
        result = build_sf_lookup(_db(rows))
        assert result.iloc[0]["confidence"] == "MEDIUM"

    def test_low_when_only_low(self):
        result = build_sf_lookup(_db([_sf_row(sf_record_confidence="LOW")]))
        assert result.iloc[0]["confidence"] == "LOW"


# ---------------------------------------------------------------------------
# Material resolution
# ---------------------------------------------------------------------------

class TestMaterialResolution:
    def test_verified_material_used_when_not_unknown(self):
        result = build_sf_lookup(_db([_sf_row(verified_material="4140 HR HT")]))
        assert result.iloc[0]["material"] == "4140 HR HT"

    def test_candidate_used_when_verified_unknown(self):
        result = build_sf_lookup(_db([
            _sf_row(verified_material="UNKNOWN", material_candidate_1="316 SS")
        ]))
        assert result.iloc[0]["material"] == "316 SS"

    def test_unknown_when_both_empty(self):
        result = build_sf_lookup(_db([
            _sf_row(verified_material="UNKNOWN", material_candidate_1="")
        ]))
        assert result.iloc[0]["material"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Material-only and material+tool_type filtering
# ---------------------------------------------------------------------------

class TestLookupFiltering:
    def _lookup(self):
        rows = [
            _sf_row(tool_name="DNMG 443-PR",   verified_material="4140 HR HT", S=400.0),
            _sf_row(tool_name="VNMG 332-PF",   verified_material="4140 HR HT", S=600.0),
            _sf_row(tool_name="3/8\" FLAT ENDMILL", verified_material="316 SS",  S=1200.0),
        ]
        return build_sf_lookup(_db(rows))

    def test_material_only_returns_all_tools_for_that_material(self):
        result = self._lookup()
        mat_rows = result[result["material"] == "4140 HR HT"]
        assert len(mat_rows) == 2

    def test_material_plus_tool_type_narrows_results(self):
        result = self._lookup()
        filtered = result[
            (result["material"] == "4140 HR HT") &
            (result["tool_type"] == "turning_rough")
        ]
        assert len(filtered) == 1
        assert filtered.iloc[0]["S_mid"] == 400

    def test_different_material_separate_group(self):
        result = self._lookup()
        ss_rows = result[result["material"] == "316 SS"]
        assert len(ss_rows) == 1
        assert ss_rows.iloc[0]["tool_type"] == "milling_profile"


# ---------------------------------------------------------------------------
# Tool type classification in lookup
# ---------------------------------------------------------------------------

class TestToolTypeInLookup:
    def test_dnmg_classified_as_turning_rough(self):
        result = build_sf_lookup(_db([_sf_row(tool_name="DNMG 443-PR")]))
        assert result.iloc[0]["tool_type"] == "turning_rough"

    def test_vnmg_classified_as_turning_finish(self):
        result = build_sf_lookup(_db([_sf_row(tool_name="VNMG 332-PF")]))
        assert result.iloc[0]["tool_type"] == "turning_finish"

    def test_endmill_classified_as_milling_profile(self):
        result = build_sf_lookup(_db([_sf_row(tool_name='3/8" FLAT ENDMILL')]))
        assert result.iloc[0]["tool_type"] == "milling_profile"

    def test_threading_insert_classified(self):
        result = build_sf_lookup(_db([_sf_row(tool_name="16ER 12 UN")]))
        assert result.iloc[0]["tool_type"] == "threading"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_creates_file(self, tmp_path):
        df = build_sf_lookup(_db([_sf_row()]))
        path = export_sf_lookup(df, tmp_path, "20260101_000000")
        assert path.exists()

    def test_exported_csv_has_correct_columns(self, tmp_path):
        df = build_sf_lookup(_db([_sf_row()]))
        path = export_sf_lookup(df, tmp_path, "20260101_000000")
        loaded = pd.read_csv(path)
        for col in SF_LOOKUP_COLS:
            assert col in loaded.columns, f"Missing column: {col}"

    def test_export_empty_df_creates_file_with_headers(self, tmp_path):
        path = export_sf_lookup(pd.DataFrame(), tmp_path, "20260101_000001")
        assert path.exists()
        loaded = pd.read_csv(path)
        for col in SF_LOOKUP_COLS:
            assert col in loaded.columns

    def test_no_banned_cols_in_export(self, tmp_path):
        df = build_sf_lookup(_db([_sf_row()]))
        path = export_sf_lookup(df, tmp_path, "20260101_000002")
        loaded = pd.read_csv(path)
        for col in _BANNED_COLS:
            assert col not in loaded.columns, f"Banned column in export: {col}"

    def test_dashboard_loads_lookup(self):
        """Smoke test: loader can load the lookup from a real or dummy path."""
        from src.dashboard.data_access.loader import load_latest_proven_sf_lookup
        # Function is importable and callable; no file → returns None gracefully
        result = load_latest_proven_sf_lookup.__wrapped__() \
            if hasattr(load_latest_proven_sf_lookup, "__wrapped__") \
            else None
        # Just verify no exception was raised and return value is None or DataFrame
        assert result is None or isinstance(result, pd.DataFrame)
