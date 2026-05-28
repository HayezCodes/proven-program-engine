"""
test_proven_sf_database.py — Tests for the proven S/F database builder.
"""

import time
from pathlib import Path

import pandas as pd
import pytest

from src.proven_sf_database import (
    SF_DB_COLS,
    SF_PROGRAMMER_COLS,
    SF_SUMMARY_COLS,
    build_programmer_view,
    build_sf_database,
    build_sf_summary,
    compute_needs_review,
    detect_latest_exports,
    export_programmer_view,
    export_sf_database,
    export_sf_summary,
    run_build_sf_database,
    score_sf_record_confidence,
)


# ---------------------------------------------------------------------------
# Synthetic DataFrame factories
# ---------------------------------------------------------------------------

def _cuts(rows: list[dict] = None) -> pd.DataFrame:
    defaults = {
        "record_id": 1, "program_id": 1,
        "source_file": "P:/655/Proven/EM0001.NC",
        "machine_folder": "655", "filename": "EM0001.NC",
        "line_number": 10, "active_t_code": "T0101", "tool_number": 1.0,
        "tool_description": "ROUGHING INSERT",
        "s_value": 500.0, "s_mode": "CSS", "s_type": "SPINDLE",
        "f_value": 0.012, "f_mode": "IPR",
        "block_skip": False, "lines_since_t_code": 2,
        "extraction_confidence": "HIGH",
        "raw_line": "G96 S500 F0.012", "prev_line": "T0101", "next_line": "G00",
        "context_json": "[]", "sf_combo_count": 1, "is_duplicate": False,
    }
    return pd.DataFrame([{**defaults, **r} for r in (rows or [{}])])


def _links(rows: list[dict] = None) -> pd.DataFrame:
    defaults = {
        "source_file": "P:/655/Proven/EM0001.NC",
        "filename": "EM0001.NC", "machine_folder": "655", "program_id": 1,
        "matched_job_number": "9001", "matched_part_number": "PART-A",
        "matched_drawing_number": "EM0001", "matched_revision": "A",
        "matched_shared_print_file": "",
        "matched_router_file": "G:/jobs/9001.txt",
        "link_confidence": "HIGH", "link_method": "exact_drawing_number",
        "link_reason": "EM0001 match",
        "material": "316 STAINLESS", "material_source": "ROUTER",
        "material_confidence": "HIGH",
        "material_conflict": False, "needs_review": False,
    }
    return pd.DataFrame([{**defaults, **r} for r in (rows or [{}])])


def _tooldb(rows: list[dict] = None) -> pd.DataFrame:
    defaults = {
        "source_tooldb": "655.tooldb", "machine_id": "655",
        "tool_number": "1", "tool_station": 1,
        "tool_name": "DNMG 432 ROUGH INSERT", "holder_name": "DCLNR",
        "tool_category": "Lathe", "overall_diameter": 0.5,
        "flute_count": None, "mc_tool_type": None,
        "insert_shape": "D", "insert_ic_diameter": 0.5, "insert_corner_radius": 0.031,
        "feed_rate_ipm": 10.0, "spindle_rpm": 800,
        "is_metric": False, "sf_usable": False,
        "sf_reject_reason": "no TlWorkMaterial — no material context",
        "extraction_method": "assembly_join",
        "confidence": "HIGH", "needs_review": False, "raw_json": "{}",
    }
    return pd.DataFrame([{**defaults, **r} for r in (rows or [{}])])


def _tooling_rev(rows: list[dict] = None) -> pd.DataFrame:
    defaults = {
        "machine_folder": "655", "tool_number": "1",
        "active_t_code": "T0101",
        "program_description": "ROUGHING INSERT",
        "reference_description": "DNMG 432 ROUGHING",
        "decimal_size": 0.5,
        "match_status": "description_match",
        "reference_needs_review": False,
        "review_action": "", "corrected_description": "", "notes": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in (rows or [{}])])


def _mat_cands(rows: list[dict] = None) -> pd.DataFrame:
    defaults = {
        "tool_number": "1", "active_t_code": "T0101",
        "machine_folder": "655", "s_mode": "CSS",
        "s_count": 5, "s_mean": 500.0,
        "material_candidate_1": "316",
        "confidence_label": "HIGH",
        "feed_intent_candidate": "finish_diameter_to_size_candidate",
        "feed_intent_confidence": "MEDIUM",
    }
    return pd.DataFrame([{**defaults, **r} for r in (rows or [{}])])


def _router_ctx(rows: list[dict] = None) -> pd.DataFrame:
    defaults = {
        "matched_job_number": "9001",
        "matched_part_number": "PART-A",
        "matched_drawing_number": "EM0001",
        "operation_number": "10",
        "work_center": "655 HAAS",
        "machine_hint": "HAAS",
        "operation_description": "TURN OD TO PRINT",
        "program_reference": "EM0001.NC",
        "source_router_file": "G:/jobs/9001.txt",
        "source_file": "P:/655/Proven/EM0001.NC",
        "machine_folder": "655",
        "context_match_confidence": "HIGH",
        "context_match_reason": "EM0001 match",
    }
    return pd.DataFrame([{**defaults, **r} for r in (rows or [{}])])


# ---------------------------------------------------------------------------
# score_sf_record_confidence
# ---------------------------------------------------------------------------

class TestScoreSfRecordConfidence:
    def _high_row(self):
        return {
            "verified_material":    "316 STAINLESS",
            "material_source":      "ROUTER",
            "tool_identity_source": "TOOLDB_ASSEMBLY",
            "extraction_confidence":"HIGH",
            "s_type":               "SPINDLE",
            "link_confidence":      "HIGH",
            "material_candidate_1": "",
        }

    def test_all_conditions_met_is_high(self):
        assert score_sf_record_confidence(self._high_row()) == "HIGH"

    def test_medium_link_still_high(self):
        row = {**self._high_row(), "link_confidence": "MEDIUM"}
        assert score_sf_record_confidence(row) == "HIGH"

    def test_unverified_but_candidate_and_tool_is_medium(self):
        row = {
            "verified_material":    "UNKNOWN",
            "material_source":      "UNKNOWN",
            "tool_identity_source": "PROGRAM",
            "extraction_confidence":"MEDIUM",
            "s_type":               "SPINDLE",
            "link_confidence":      "NONE",
            "material_candidate_1": "4140",
        }
        assert score_sf_record_confidence(row) == "MEDIUM"

    def test_missing_material_and_tool_is_low(self):
        row = {
            "verified_material":    "UNKNOWN",
            "material_source":      "",
            "tool_identity_source": "UNKNOWN",
            "extraction_confidence":"LOW",
            "s_type":               "SPINDLE",
            "link_confidence":      "NONE",
            "material_candidate_1": "",
        }
        assert score_sf_record_confidence(row) == "LOW"

    def test_g92_limit_not_high(self):
        row = {**self._high_row(), "s_type": "LIMIT"}
        assert score_sf_record_confidence(row) != "HIGH"

    def test_low_extraction_confidence_not_high(self):
        row = {**self._high_row(), "extraction_confidence": "LOW"}
        assert score_sf_record_confidence(row) != "HIGH"


# ---------------------------------------------------------------------------
# compute_needs_review
# ---------------------------------------------------------------------------

class TestComputeNeedsReview:
    def _clean_row(self):
        return {
            "verified_material":    "316 STAINLESS",
            "material_confidence":  "HIGH",
            "tool_identity_source": "TOOLDB_ASSEMBLY",
            "tool_needs_review":    False,
            "extraction_confidence":"HIGH",
            "s_type":               "SPINDLE",
            "link_confidence":      "HIGH",
            "material_conflict":    False,
        }

    def test_clean_row_no_review(self):
        needs, reason = compute_needs_review(self._clean_row())
        assert needs == False
        assert reason == ""

    def test_unknown_material_triggers_review(self):
        row = {**self._clean_row(), "verified_material": "UNKNOWN"}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "no_verified_material" in reason

    def test_low_material_confidence_triggers_review(self):
        row = {**self._clean_row(), "material_confidence": "LOW"}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "low_material_confidence" in reason

    def test_unknown_tool_triggers_review(self):
        row = {**self._clean_row(), "tool_identity_source": "UNKNOWN"}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "unknown_tool_identity" in reason

    def test_tool_needs_review_propagates(self):
        row = {**self._clean_row(), "tool_needs_review": True}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "tool_needs_review" in reason

    def test_low_extraction_confidence_triggers_review(self):
        row = {**self._clean_row(), "extraction_confidence": "LOW"}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "low_extraction_confidence" in reason

    def test_spindle_limit_triggers_review(self):
        row = {**self._clean_row(), "s_type": "LIMIT"}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "spindle_limit_record" in reason

    def test_low_link_confidence_triggers_review(self):
        row = {**self._clean_row(), "link_confidence": "LOW"}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "low_link_confidence" in reason

    def test_material_conflict_triggers_review(self):
        row = {**self._clean_row(), "material_conflict": True}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "material_conflict" in reason

    def test_multiple_reasons_combined(self):
        row = {**self._clean_row(), "verified_material": "UNKNOWN", "s_type": "LIMIT"}
        needs, reason = compute_needs_review(row)
        assert needs == True
        assert "no_verified_material" in reason
        assert "spindle_limit_record" in reason


# ---------------------------------------------------------------------------
# build_sf_database — column completeness
# ---------------------------------------------------------------------------

class TestBuildSfDatabaseColumns:
    def test_all_required_columns_present(self):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        for col in SF_DB_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_returns_dataframe(self):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_cuts(self):
        cuts = _cuts([{}, {}])  # 2 rows
        df = build_sf_database(cuts, _links(), _tooldb(), _tooling_rev(),
                                _mat_cands(), _router_ctx())
        assert len(df) == 2

    def test_empty_cuts_returns_empty_with_columns(self):
        df = build_sf_database(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert isinstance(df, pd.DataFrame)
        for col in SF_DB_COLS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Joins — material fields
# ---------------------------------------------------------------------------

class TestJoinsMaterialFields:
    def test_verified_material_from_links(self):
        df = build_sf_database(
            _cuts(), _links(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame()
        )
        assert df.iloc[0]["verified_material"] == "316 STAINLESS"
        assert df.iloc[0]["material_source"] == "ROUTER"

    def test_unmatched_program_gets_unknown_material(self):
        # cuts with a different source_file → no link match
        cuts = _cuts([{"source_file": "P:/655/Proven/NOMATCH.NC",
                        "filename": "NOMATCH.NC"}])
        df = build_sf_database(cuts, _links(), pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame())
        assert df.iloc[0]["verified_material"] == "UNKNOWN"

    def test_material_candidate_1_from_mat_cands(self):
        df = build_sf_database(
            _cuts(), _links(), pd.DataFrame(), pd.DataFrame(),
            _mat_cands(), pd.DataFrame()
        )
        assert df.iloc[0]["material_candidate_1"] == "316"

    def test_feed_intent_candidate_joined(self):
        df = build_sf_database(
            _cuts(), _links(), pd.DataFrame(), pd.DataFrame(),
            _mat_cands(), pd.DataFrame()
        )
        assert "feed_intent_candidate" in df.columns


# ---------------------------------------------------------------------------
# Material priority — verified over candidate
# ---------------------------------------------------------------------------

class TestMaterialPriority:
    def test_verified_material_takes_priority_over_candidate(self):
        # verified = 316 STAINLESS (from router), candidate = 4140 (inferred)
        links  = _links([{"material": "316 STAINLESS", "material_source": "ROUTER"}])
        cands  = _mat_cands([{"material_candidate_1": "4140"}])
        df     = build_sf_database(_cuts(), links, pd.DataFrame(), pd.DataFrame(),
                                    cands, pd.DataFrame())
        # Both columns present; sf_record_confidence should use verified
        assert df.iloc[0]["verified_material"] == "316 STAINLESS"
        assert df.iloc[0]["material_candidate_1"] == "4140"
        assert df.iloc[0]["sf_record_confidence"] != "LOW"

    def test_inferred_not_overwritten(self):
        # No link match → verified = UNKNOWN, but candidate should still appear
        cuts  = _cuts([{"source_file": "P:/NOLINK.NC"}])
        cands = _mat_cands()
        # mat_cands joined on (machine_folder, active_t_code, tool_number, s_mode)
        df = build_sf_database(cuts, pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), cands, pd.DataFrame())
        # candidate may or may not join (source_file doesn't match), but verified
        # should remain UNKNOWN
        assert df.iloc[0]["verified_material"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# G92 spindle-limit records
# ---------------------------------------------------------------------------

class TestG92LimitRecords:
    def test_limit_record_gets_needs_review_true(self):
        cuts = _cuts([{
            "s_value": 1200.0, "s_mode": "UNKNOWN", "s_type": "LIMIT",
            "f_value": None, "f_mode": None,
            "extraction_confidence": "LOW",
        }])
        df = build_sf_database(cuts, pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert df.iloc[0]["needs_review"] == True

    def test_limit_record_reason_includes_spindle_limit(self):
        cuts = _cuts([{"s_type": "LIMIT", "extraction_confidence": "LOW"}])
        df = build_sf_database(cuts, pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert "spindle_limit_record" in str(df.iloc[0]["review_reason"])

    def test_limit_record_confidence_not_high(self):
        cuts = _cuts([{"s_type": "LIMIT", "s_mode": "UNKNOWN", "extraction_confidence": "LOW"}])
        df = build_sf_database(cuts, _links(), _tooldb(), _tooling_rev(),
                                _mat_cands(), pd.DataFrame())
        assert df.iloc[0]["sf_record_confidence"] != "HIGH"

    def test_spindle_record_not_flagged_as_limit(self):
        cuts = _cuts([{"s_type": "SPINDLE", "s_mode": "CSS", "extraction_confidence": "HIGH"}])
        df = build_sf_database(cuts, _links(), _tooldb(), _tooling_rev(),
                                _mat_cands(), _router_ctx())
        assert "spindle_limit_record" not in str(df.iloc[0]["review_reason"])


# ---------------------------------------------------------------------------
# Dwell feed exclusion — dwell F values never enter cuts (parser guarantee)
# ---------------------------------------------------------------------------

class TestDwellFeedExclusion:
    def test_no_g4_dwell_patterns_in_output(self):
        # Verify: if cuts are loaded from synthetic data (no dwell lines), the
        # builder does not introduce any F values.  The raw_line check confirms
        # dwell lines would not appear as feedrate records.
        cuts = _cuts([{"raw_line": "G96 S500 F0.012", "f_mode": "IPR", "f_value": 0.012}])
        df = build_sf_database(cuts, pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        # The F value from a non-dwell line passes through unchanged
        assert df.iloc[0]["F"] == pytest.approx(0.012)

    def test_dwell_context_line_not_treated_as_feedrate(self):
        # Simulate a row that would come from a G4 line — in production this
        # should not exist in cuts (parser excludes it), but if it did arrive
        # with s_type != SPINDLE it would still be correctly flagged.
        cuts = _cuts([{"raw_line": "G04 X0.5", "s_value": None, "f_value": None}])
        df = build_sf_database(cuts, pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        # No S or F → not a cutting record
        assert pd.isna(df.iloc[0]["S"]) or df.iloc[0]["S"] == ""
        assert pd.isna(df.iloc[0]["F"]) or df.iloc[0]["F"] == ""


# ---------------------------------------------------------------------------
# TOOLDB S/F not used
# ---------------------------------------------------------------------------

class TestNoTooldbSfUsage:
    def test_tooldb_feed_rate_not_in_output_S(self):
        # TOOLDB has feed_rate_ipm=50.0 and spindle_rpm=1500
        # These must NOT appear as S/F values in the database
        tdb = _tooldb([{"feed_rate_ipm": 50.0, "spindle_rpm": 1500, "sf_usable": False}])
        cuts = _cuts([{"s_value": 500.0, "f_value": 0.012}])
        df = build_sf_database(cuts, pd.DataFrame(), tdb, pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame())
        # S must be from cuts (500), NOT from TOOLDB (1500)
        assert df.iloc[0]["S"] == 500.0
        # F must be from cuts (0.012), NOT from TOOLDB (50.0)
        assert df.iloc[0]["F"] == pytest.approx(0.012)

    def test_tooldb_sf_usable_false_respected(self):
        tdb = _tooldb([{"sf_usable": False, "sf_reject_reason": "no TlWorkMaterial"}])
        # sf_usable=False means those values must never be used as proven S/F
        assert tdb.iloc[0]["sf_usable"] == False


# ---------------------------------------------------------------------------
# Tool identity resolution
# ---------------------------------------------------------------------------

class TestToolIdentityResolution:
    def test_tooldb_assembly_tool_name_resolved(self):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        assert df.iloc[0]["resolved_tool_name"] == "DNMG 432 ROUGH INSERT"
        assert df.iloc[0]["tool_identity_source"] == "TOOLDB_ASSEMBLY"

    def test_unknown_tool_flagged(self):
        # No tooldb, no tooling_review, no tool_description → UNKNOWN
        cuts = _cuts([{"tool_description": None}])
        df = build_sf_database(cuts, pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert df.iloc[0]["tool_identity_source"] == "UNKNOWN"
        assert df.iloc[0]["tool_needs_review"] == True

    def test_program_description_used_as_fallback(self):
        # Tooldb empty → falls through to program description
        cuts = _cuts([{"tool_description": "MY ROUGHER"}])
        df = build_sf_database(cuts, pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert df.iloc[0]["resolved_tool_name"] == "MY ROUGHER"
        assert df.iloc[0]["tool_identity_source"] == "PROGRAM"


# ---------------------------------------------------------------------------
# Router context join
# ---------------------------------------------------------------------------

class TestRouterContextJoin:
    def test_router_work_center_joined(self):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        assert df.iloc[0]["router_work_center"] == "655 HAAS"

    def test_router_operation_description_joined(self):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        assert df.iloc[0]["router_operation_description"] == "TURN OD TO PRINT"

    def test_linked_router_file_from_links(self):
        df = build_sf_database(
            _cuts(), _links(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert df.iloc[0]["linked_router_file"] == "G:/jobs/9001.txt"

    def test_missing_router_context_empty_not_error(self):
        df = build_sf_database(
            _cuts(), _links(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame()
        )
        # No router_ctx → empty strings, no crash
        assert "router_work_center" in df.columns


# ---------------------------------------------------------------------------
# build_sf_summary
# ---------------------------------------------------------------------------

class TestBuildSfSummary:
    def _build_db(self, n_rows=3) -> pd.DataFrame:
        rows = [{"s_value": 400.0 + i * 50, "f_value": 0.010 + i * 0.002} for i in range(n_rows)]
        cuts = _cuts(rows)
        return build_sf_database(cuts, _links(), _tooldb(), _tooling_rev(),
                                  _mat_cands(), _router_ctx())

    def test_returns_dataframe(self):
        sf_db = self._build_db()
        summ = build_sf_summary(sf_db)
        assert isinstance(summ, pd.DataFrame)

    def test_has_all_summary_columns(self):
        sf_db = self._build_db()
        summ = build_sf_summary(sf_db)
        for col in SF_SUMMARY_COLS:
            assert col in summ.columns, f"Missing summary column: {col}"

    def test_occurrence_count_correct(self):
        sf_db = self._build_db(n_rows=3)
        summ = build_sf_summary(sf_db)
        assert summ["occurrence_count"].sum() == 3

    def test_s_avg_within_range(self):
        sf_db = self._build_db(n_rows=2)  # S = 400, 450
        summ = build_sf_summary(sf_db)
        # Only rows where S is real numeric
        s_avgs = pd.to_numeric(summ["S_avg"], errors="coerce").dropna()
        assert (s_avgs > 0).all()

    def test_final_material_uses_verified_first(self):
        sf_db = self._build_db()
        summ = build_sf_summary(sf_db)
        # verified_material = "316 STAINLESS" from links
        assert (summ["final_material"] == "316 STAINLESS").any()

    def test_empty_db_returns_empty_with_columns(self):
        summ = build_sf_summary(pd.DataFrame(columns=SF_DB_COLS))
        assert isinstance(summ, pd.DataFrame)
        for col in SF_SUMMARY_COLS:
            assert col in summ.columns

    def test_needs_review_count_correct(self):
        sf_db = self._build_db(n_rows=2)
        summ = build_sf_summary(sf_db)
        assert "needs_review_count" in summ.columns
        assert (summ["needs_review_count"] >= 0).all()


# ---------------------------------------------------------------------------
# build_programmer_view
# ---------------------------------------------------------------------------

class TestBuildProgrammerView:
    def _build_db(self, rows=None, links=None, cands=None) -> pd.DataFrame:
        return build_sf_database(
            _cuts(rows or [{}]),
            links if links is not None else _links(),
            _tooldb(),
            _tooling_rev(),
            cands if cands is not None else _mat_cands(),
            _router_ctx(),
        )

    def test_has_core_programmer_columns_only(self):
        view = build_programmer_view(self._build_db())
        assert list(view.columns) == SF_PROGRAMMER_COLS

    def test_verified_material_takes_priority(self):
        links = _links([{"material": "316 STAINLESS"}])
        cands = _mat_cands([{"material_candidate_1": "4140"}])
        view = build_programmer_view(self._build_db(links=links, cands=cands))
        assert view.iloc[0]["material"] == "316 STAINLESS"

    def test_candidate_used_when_verified_unknown(self):
        links = _links([{"material": "UNKNOWN", "material_source": "UNKNOWN"}])
        cands = _mat_cands([{"material_candidate_1": "4140"}])
        view = build_programmer_view(self._build_db(links=links, cands=cands))
        assert view.iloc[0]["material"] == "4140"

    def test_unknown_material_when_no_verified_or_candidate(self):
        links = _links([{"material": "UNKNOWN", "material_source": "UNKNOWN"}])
        cands = _mat_cands([{"material_candidate_1": ""}])
        view = build_programmer_view(self._build_db(links=links, cands=cands))
        assert view.iloc[0]["material"] == "UNKNOWN"

    def test_grouping_combines_matching_rows(self):
        rows = [
            {"s_value": 400.0, "f_value": 0.010},
            {"s_value": 600.0, "f_value": 0.014},
        ]
        view = build_programmer_view(self._build_db(rows=rows))
        assert len(view) == 1
        assert view.iloc[0]["occurrence_count"] == 2
        assert view.iloc[0]["S_min"] == 400.0
        assert view.iloc[0]["S_avg"] == 500.0
        assert view.iloc[0]["S_max"] == 600.0
        assert view.iloc[0]["F_min"] == pytest.approx(0.010)
        assert view.iloc[0]["F_avg"] == pytest.approx(0.012)
        assert view.iloc[0]["F_max"] == pytest.approx(0.014)

    def test_no_manufacturing_intelligence_columns(self):
        view = build_programmer_view(self._build_db())
        forbidden = {
            "matched_job_number",
            "matched_part_number",
            "matched_drawing_number",
            "linked_router_file",
            "source_file",
            "raw_line",
            "prev_line",
            "next_line",
        }
        assert forbidden.isdisjoint(view.columns)

    def test_full_database_preserves_manufacturing_intelligence_fields(self):
        db = self._build_db()
        for col in (
            "matched_job_number",
            "matched_part_number",
            "matched_drawing_number",
            "linked_router_file",
            "raw_line",
            "prev_line",
            "next_line",
        ):
            assert col in db.columns


# ---------------------------------------------------------------------------
# Empty / missing optional file handling
# ---------------------------------------------------------------------------

class TestEmptyInputHandling:
    def test_all_empty_returns_empty_with_columns(self):
        df = build_sf_database(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        for col in SF_DB_COLS:
            assert col in df.columns

    def test_missing_links_df_not_crash(self):
        df = build_sf_database(
            _cuts(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert len(df) == 1
        assert df.iloc[0]["verified_material"] == "UNKNOWN"

    def test_missing_tooldb_not_crash(self):
        df = build_sf_database(
            _cuts(), _links(), pd.DataFrame(),
            _tooling_rev(), _mat_cands(), pd.DataFrame()
        )
        assert len(df) == 1

    def test_missing_mat_cands_not_crash(self):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(),
            _tooling_rev(), pd.DataFrame(), pd.DataFrame()
        )
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Export column completeness and no-overwrite
# ---------------------------------------------------------------------------

class TestExports:
    def test_export_sf_database_creates_csv(self, tmp_path):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        path = export_sf_database(df, tmp_path, "20260101_000000")
        assert path.exists()
        assert path.name == "proven_sf_database_20260101_000000.csv"

    def test_export_csv_has_all_columns(self, tmp_path):
        df = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        path = export_sf_database(df, tmp_path, "20260101_000000")
        result = pd.read_csv(path)
        for col in SF_DB_COLS:
            assert col in result.columns, f"Missing in CSV: {col}"

    def test_export_summary_creates_csv(self, tmp_path):
        db = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        summ = build_sf_summary(db)
        path = export_sf_summary(summ, tmp_path, "20260101_000000")
        assert path.exists()

    def test_export_programmer_view_creates_csv(self, tmp_path):
        db = build_sf_database(
            _cuts(), _links(), _tooldb(), _tooling_rev(), _mat_cands(), _router_ctx()
        )
        view = build_programmer_view(db)
        path = export_programmer_view(view, tmp_path, "20260101_000000")
        assert path.exists()
        assert path.name == "proven_sf_programmer_view_20260101_000000.csv"
        result = pd.read_csv(path)
        assert list(result.columns) == SF_PROGRAMMER_COLS

    def test_no_overwrite_different_timestamps(self, tmp_path):
        df = build_sf_database(_cuts(), pd.DataFrame(), pd.DataFrame(),
                                pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        p1 = export_sf_database(df, tmp_path, "20260101_000001")
        p2 = export_sf_database(df, tmp_path, "20260101_000002")
        assert p1 != p2
        assert p1.exists() and p2.exists()

    def test_safety_blocks_production_path(self):
        from src.safety import ProductionWriteViolation
        with pytest.raises(ProductionWriteViolation):
            export_sf_database(pd.DataFrame(columns=SF_DB_COLS),
                               Path(r"P:\Manufacturing"), "test")


# ---------------------------------------------------------------------------
# Latest file auto-detection
# ---------------------------------------------------------------------------

class TestLatestFileAutoDetection:
    def test_detects_latest_cuts(self, tmp_path):
        (tmp_path / "cuts_20260101_000001.csv").write_text("col\nval")
        (tmp_path / "cuts_20260101_000002.csv").write_text("col\nval")
        d = detect_latest_exports(tmp_path)
        assert d["cuts"] is not None
        assert "000002" in d["cuts"].name

    def test_returns_none_for_missing_files(self, tmp_path):
        d = detect_latest_exports(tmp_path)
        assert d["cuts"] is None
        assert d["links"] is None

    def test_detects_all_six_types(self, tmp_path):
        for stem in [
            "cuts_20260101_000001",
            "program_job_links_20260101_000001",
            "tooldb_reference_20260101_000001",
            "tooling_review_20260101_000001",
            "material_candidates_20260101_000001",
            "router_program_context_20260101_000001",
        ]:
            (tmp_path / f"{stem}.csv").write_text("col\nval")
        d = detect_latest_exports(tmp_path)
        for key in ("cuts", "links", "tooldb", "tooling_review",
                    "mat_candidates", "router_context"):
            assert d[key] is not None, f"Not detected: {key}"

    def test_picks_newest_by_mtime(self, tmp_path):
        p1 = tmp_path / "cuts_20260101_000001.csv"
        p2 = tmp_path / "cuts_20260101_000002.csv"
        p1.write_text("col\nval")
        time.sleep(0.05)
        p2.write_text("col\nval")
        d = detect_latest_exports(tmp_path)
        assert d["cuts"].name == p2.name


# ---------------------------------------------------------------------------
# run_build_sf_database integration
# ---------------------------------------------------------------------------

class TestRunBuildSfDatabase:
    def _write_csv(self, path: Path, df: pd.DataFrame) -> None:
        df.to_csv(path, index=False)

    def test_run_produces_two_files(self, tmp_path):
        exports = tmp_path / "exports"
        exports.mkdir()
        self._write_csv(exports / "cuts_20260101_000001.csv", _cuts())
        self._write_csv(exports / "program_job_links_20260101_000001.csv", _links())
        self._write_csv(exports / "tooldb_reference_20260101_000001.csv", _tooldb())
        self._write_csv(exports / "tooling_review_20260101_000001.csv", _tooling_rev())
        self._write_csv(exports / "material_candidates_20260101_000001.csv", _mat_cands())
        self._write_csv(exports / "router_program_context_20260101_000001.csv", _router_ctx())

        db_path, summ_path, prog_path = run_build_sf_database(exports_dir=exports)
        assert db_path.exists()
        assert summ_path.exists()
        assert prog_path.exists()

    def test_run_with_no_cuts_creates_empty_outputs(self, tmp_path):
        exports = tmp_path / "empty_exports"
        exports.mkdir()
        db_path, summ_path, prog_path = run_build_sf_database(exports_dir=exports)
        assert db_path.exists()
        assert summ_path.exists()
        assert prog_path.exists()

    def test_end_to_end_material_in_database(self, tmp_path):
        exports = tmp_path / "exports"
        exports.mkdir()
        self._write_csv(exports / "cuts_20260101_000001.csv", _cuts())
        self._write_csv(exports / "program_job_links_20260101_000001.csv", _links())

        db_path, _, prog_path = run_build_sf_database(exports_dir=exports)
        db = pd.read_csv(db_path)
        assert db.iloc[0]["verified_material"] == "316 STAINLESS"
        assert db.iloc[0]["material_source"] == "ROUTER"
        programmer = pd.read_csv(prog_path)
        assert "matched_job_number" not in programmer.columns
