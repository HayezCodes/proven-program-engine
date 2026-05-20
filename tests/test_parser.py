"""
test_parser.py — Tests for the parser module (Phase 1 + Phase 2).
"""

import json
from pathlib import Path

import pytest

from src.parser import (
    _annotate_duplicates,
    _extract_tool_number,
    _find_tool_description,
    _get_context,
    _score_confidence,
    _strip_comments,
    parse_file,
)


# ---------------------------------------------------------------------------
# _strip_comments
# ---------------------------------------------------------------------------

class TestStripComments:
    def test_removes_parenthetical_comment(self):
        assert _strip_comments("G97 S450 M03 (ROUGH SPEED)") == "G97 S450 M03"

    def test_removes_semicolon_comment(self):
        assert _strip_comments("G97 S450 ; RPM mode") == "G97 S450"

    def test_leaves_clean_code_unchanged(self):
        assert _strip_comments("G96 S600") == "G96 S600"

    def test_empty_line_returns_empty(self):
        assert _strip_comments("") == ""

    def test_comment_only_line_returns_empty(self):
        assert _strip_comments("(TOOL 01 - DNMG)") == ""

    def test_multiple_comments_removed(self):
        result = _strip_comments("T0101 (tool) G97 (mode) S450")
        assert "tool" not in result
        assert "mode" not in result
        assert "S450" in result


# ---------------------------------------------------------------------------
# _extract_tool_number
# ---------------------------------------------------------------------------

class TestExtractToolNumber:
    def test_four_digit_t0101(self):
        assert _extract_tool_number("T0101") == "01"

    def test_four_digit_t0202(self):
        assert _extract_tool_number("T0202") == "02"

    def test_two_digit_t01(self):
        assert _extract_tool_number("T01") == "01"

    def test_single_digit_t1(self):
        assert _extract_tool_number("T1") == "1"

    def test_empty_string_returns_empty(self):
        assert _extract_tool_number("") == ""


# ---------------------------------------------------------------------------
# _get_context
# ---------------------------------------------------------------------------

class TestGetContext:
    def test_center_line_marked_with_arrow(self):
        lines = ["A", "B", "C", "D", "E"]
        ctx = _get_context(lines, 2, window=1)
        assert any(">>>" in line for line in ctx)
        center = [l for l in ctx if ">>>" in l]
        assert len(center) == 1
        assert "C" in center[0]

    def test_window_does_not_exceed_file_bounds(self):
        lines = ["X", "Y"]
        ctx = _get_context(lines, 0, window=5)
        assert len(ctx) == 2

    def test_context_includes_line_numbers(self):
        lines = ["A", "B", "C"]
        ctx = _get_context(lines, 1, window=1)
        assert any("L" in line for line in ctx)


# ---------------------------------------------------------------------------
# _find_tool_description
# ---------------------------------------------------------------------------

class TestFindToolDescription:
    def test_returns_same_line_comment_if_present(self):
        lines = ["(PREV COMMENT)", "T0101 (DNMG ROUGHER)", "G97 S450"]
        desc = _find_tool_description(lines, 1, "DNMG ROUGHER")
        assert desc == "DNMG ROUGHER"

    def test_looks_back_to_preceding_comment(self):
        lines = ["(T01 - DNMG 432)", "T0101", "G97 S450"]
        desc = _find_tool_description(lines, 1, "")
        assert desc == "T01 - DNMG 432"

    def test_stops_at_code_line(self):
        lines = ["G97 S450", "(IGNORED COMMENT)", "T0101", "G97 S600"]
        # The preceding comment is after a code line — should still find it
        desc = _find_tool_description(lines, 2, "")
        assert desc == "IGNORED COMMENT"

    def test_returns_empty_if_no_comment_available(self):
        lines = ["G97 S450", "T0101"]
        desc = _find_tool_description(lines, 1, "")
        assert desc == ""


# ---------------------------------------------------------------------------
# _score_confidence
# ---------------------------------------------------------------------------

class TestScoreConfidence:
    def test_high_when_explicit_spindle_mode_with_s(self):
        assert _score_confidence("T0101", 450.0, None, True) == "HIGH"

    def test_high_when_both_s_and_f_on_line(self):
        assert _score_confidence("T0101", 450.0, 0.015, False) == "HIGH"

    def test_medium_when_only_s_no_mode_on_line(self):
        assert _score_confidence("T0101", 450.0, None, False) == "MEDIUM"

    def test_medium_when_only_f(self):
        assert _score_confidence("T0202", None, 0.015, False) == "MEDIUM"

    def test_low_when_no_active_t_code(self):
        assert _score_confidence("", 450.0, None, False) == "LOW"

    def test_low_when_t_code_is_none(self):
        assert _score_confidence(None, 450.0, 0.015, True) == "LOW"


# ---------------------------------------------------------------------------
# _annotate_duplicates
# ---------------------------------------------------------------------------

class TestAnnotateDuplicates:
    def _make_record(self, tool: str, s, f) -> dict:
        return {
            "tool_number": tool,
            "s_value": s,
            "f_value": f,
        }

    def test_first_occurrence_not_marked_duplicate(self):
        records = [self._make_record("01", 450.0, 0.015)]
        _annotate_duplicates(records)
        assert records[0]["is_duplicate"] is False

    def test_second_occurrence_marked_duplicate(self):
        records = [
            self._make_record("01", 450.0, 0.015),
            self._make_record("01", 450.0, 0.015),
        ]
        _annotate_duplicates(records)
        assert records[0]["is_duplicate"] is False
        assert records[1]["is_duplicate"] is True

    def test_combo_count_reflects_total_occurrences(self):
        records = [
            self._make_record("01", 450.0, 0.015),
            self._make_record("01", 450.0, 0.015),
            self._make_record("01", 450.0, 0.015),
        ]
        _annotate_duplicates(records)
        assert all(r["sf_combo_count"] == 3 for r in records)

    def test_different_tools_not_conflated(self):
        records = [
            self._make_record("01", 450.0, 0.015),
            self._make_record("02", 450.0, 0.015),
        ]
        _annotate_duplicates(records)
        assert records[0]["sf_combo_count"] == 1
        assert records[1]["sf_combo_count"] == 1
        assert records[1]["is_duplicate"] is False

    def test_none_values_in_combo_handled(self):
        records = [
            self._make_record("01", 450.0, None),
            self._make_record("01", 450.0, None),
        ]
        _annotate_duplicates(records)
        assert records[1]["is_duplicate"] is True
        assert records[0]["sf_combo_count"] == 2


# ---------------------------------------------------------------------------
# parse_file — Phase 1 extractions (unchanged behaviour)
# ---------------------------------------------------------------------------

class TestParseFile:
    def test_returns_list_of_records(self, sample_nc_file):
        assert len(parse_file(sample_nc_file)) > 0

    def test_extracts_s_value_450(self, sample_nc_file):
        vals = {r["s_value"] for r in parse_file(sample_nc_file) if r["s_value"] is not None}
        assert 450.0 in vals

    def test_extracts_s_value_600(self, sample_nc_file):
        vals = {r["s_value"] for r in parse_file(sample_nc_file) if r["s_value"] is not None}
        assert 600.0 in vals

    def test_extracts_s_value_1200(self, sample_nc_file):
        vals = {r["s_value"] for r in parse_file(sample_nc_file) if r["s_value"] is not None}
        assert 1200.0 in vals

    def test_extracts_f_value_0_015(self, sample_nc_file):
        vals = {r["f_value"] for r in parse_file(sample_nc_file) if r["f_value"] is not None}
        assert 0.015 in vals

    def test_extracts_f_value_0_006(self, sample_nc_file):
        vals = {r["f_value"] for r in parse_file(sample_nc_file) if r["f_value"] is not None}
        assert 0.006 in vals

    def test_extracts_f_value_0_004(self, sample_nc_file):
        vals = {r["f_value"] for r in parse_file(sample_nc_file) if r["f_value"] is not None}
        assert 0.004 in vals

    def test_g97_mode_is_rpm(self, sample_nc_file):
        assert any(r["s_mode"] == "RPM" for r in parse_file(sample_nc_file))

    def test_g96_mode_is_css(self, sample_nc_file):
        assert any(r["s_mode"] == "CSS" for r in parse_file(sample_nc_file))

    def test_t0202_associated_with_css_s600(self, sample_nc_file):
        match = [
            r for r in parse_file(sample_nc_file)
            if r["active_t_code"] == "T0202" and r["s_value"] == 600.0 and r["s_mode"] == "CSS"
        ]
        assert len(match) > 0

    def test_t0101_associated_with_rpm_s450(self, sample_nc_file):
        match = [
            r for r in parse_file(sample_nc_file)
            if r["active_t_code"] == "T0101" and r["s_value"] == 450.0 and r["s_mode"] == "RPM"
        ]
        assert len(match) > 0

    def test_context_json_is_valid_list(self, sample_nc_file):
        for rec in parse_file(sample_nc_file):
            ctx = json.loads(rec["context_json"])
            assert isinstance(ctx, list) and len(ctx) > 0

    def test_context_contains_target_line_marker(self, sample_nc_file):
        for rec in parse_file(sample_nc_file):
            ctx = json.loads(rec["context_json"])
            assert sum(1 for l in ctx if ">>>" in l) == 1

    def test_record_ids_are_sequential(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        assert [r["record_id"] for r in records] == list(range(1, len(records) + 1))

    def test_s_not_extracted_from_comments(self, tmp_path):
        nc = tmp_path / "test.NC"
        nc.write_text("T0101\n(DO NOT USE S999)\nG97 S450 M03\n")
        vals = {r["s_value"] for r in parse_file(nc) if r["s_value"] is not None}
        assert 999.0 not in vals
        assert 450.0 in vals

    def test_nonexistent_file_returns_empty_list(self, tmp_path):
        assert parse_file(tmp_path / "ghost.NC") == []

    def test_program_id_propagated(self, sample_nc_file):
        assert all(r["program_id"] == 42 for r in parse_file(sample_nc_file, program_id=42))

    def test_all_required_phase2_fields_present(self, sample_nc_file):
        required = {
            "record_id", "program_id", "source_file", "machine_folder",
            "filename", "line_number", "active_t_code", "tool_number",
            "tool_description", "s_value", "s_mode", "s_type",
            "f_value", "f_mode", "block_skip", "lines_since_t_code",
            "extraction_confidence", "raw_line", "prev_line", "next_line",
            "context_json",
        }
        for rec in parse_file(sample_nc_file):
            assert required.issubset(rec.keys()), f"Missing: {required - rec.keys()}"


# ---------------------------------------------------------------------------
# Phase 2 — G4 dwell, G92 limit, block skip, feed mode, confidence
# ---------------------------------------------------------------------------

class TestDwellSkip:
    def test_f_not_extracted_from_g4_line(self, sample_nc_file):
        """G04 X0.5 on a dwell line — F inside it should not appear (no F in sample G4)."""
        records = parse_file(sample_nc_file)
        f_vals = {r["f_value"] for r in records if r["f_value"] is not None}
        # The sample has G04 X0.5 (no F on that line) so this mainly tests
        # that non-dwell F values are still extracted.
        assert 0.004 in f_vals

    def test_f_skipped_on_explicit_g4_f_line(self, tmp_path):
        prog = "T0101\nG97 S450 M03\nG4 F3\nG01 Z-1.0 F0.010\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        f_vals = {r["f_value"] for r in records if r["f_value"] is not None}
        assert 3.0 not in f_vals        # dwell time must not appear as feedrate
        assert 0.010 in f_vals          # real feedrate must still appear

    def test_g04_variant_also_skipped(self, tmp_path):
        prog = "T0101\nG97 S450 M03\nG04 F5\nG01 Z-1.0 F0.008\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        f_vals = {r["f_value"] for r in records if r["f_value"] is not None}
        assert 5.0 not in f_vals
        assert 0.008 in f_vals


class TestSpindleLimit:
    def test_g92_s_tagged_as_limit(self, sample_advanced_nc_file):
        records = parse_file(sample_advanced_nc_file)
        limit_recs = [r for r in records if r.get("s_type") == "LIMIT"]
        assert len(limit_recs) > 0

    def test_g92_s_value_is_extracted(self, sample_advanced_nc_file):
        records = parse_file(sample_advanced_nc_file)
        limit_vals = {r["s_value"] for r in records if r.get("s_type") == "LIMIT"}
        assert 1200.0 in limit_vals

    def test_normal_s_tagged_as_spindle(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        spindle_recs = [r for r in records if r.get("s_type") == "SPINDLE"]
        assert len(spindle_recs) > 0

    def test_g92_tagged_independently_of_normal_s(self, tmp_path):
        prog = "G92 S3000\nT0101\nG97 S600 M03\nG01 Z-1.0 F0.010\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        types = {r["s_type"] for r in records if r.get("s_type")}
        assert "LIMIT" in types
        assert "SPINDLE" in types


class TestFeedMode:
    def test_g95_sets_ipr_mode(self, sample_advanced_nc_file):
        records = parse_file(sample_advanced_nc_file)
        ipr_recs = [r for r in records if r.get("f_mode") == "IPR"]
        assert len(ipr_recs) > 0

    def test_g94_sets_ipm_mode(self, sample_advanced_nc_file):
        records = parse_file(sample_advanced_nc_file)
        ipm_recs = [r for r in records if r.get("f_mode") == "IPM"]
        assert len(ipm_recs) > 0

    def test_f_mode_persists_until_changed(self, tmp_path):
        prog = "T0101\nG95\nG97 S450 M03\nG01 Z-1.0 F0.010\nG01 Z-2.0 F0.012\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        f_recs = [r for r in records if r.get("f_value") is not None]
        assert all(r["f_mode"] == "IPR" for r in f_recs)

    def test_unknown_f_mode_before_any_g94_g95(self, tmp_path):
        prog = "T0101\nG97 S450 M03\nG01 Z-1.0 F0.010\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        f_recs = [r for r in records if r.get("f_value") is not None]
        assert all(r["f_mode"] == "UNKNOWN" for r in f_recs)


class TestBlockSkip:
    def test_block_skip_lines_flagged(self, sample_advanced_nc_file):
        records = parse_file(sample_advanced_nc_file)
        skip_recs = [r for r in records if r.get("block_skip") is True]
        assert len(skip_recs) > 0

    def test_non_skip_lines_not_flagged(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        assert all(r.get("block_skip") is False for r in records)

    def test_s_value_extracted_from_block_skip_line(self, sample_advanced_nc_file):
        records = parse_file(sample_advanced_nc_file)
        skip_s = [r for r in records if r.get("block_skip") and r.get("s_value") is not None]
        assert len(skip_s) > 0


class TestExtractionConfidence:
    def test_high_confidence_on_explicit_spindle_set(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        high = [r for r in records if r["extraction_confidence"] == "HIGH"]
        assert len(high) > 0

    def test_medium_confidence_on_f_only_lines(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        med = [r for r in records if r["extraction_confidence"] == "MEDIUM"]
        assert len(med) > 0

    def test_low_confidence_on_orphan_records(self, tmp_path):
        # S/F before any T code → LOW
        prog = "G97 S1200 M03\nG01 Z-1.0 F0.010\nT0101\nG97 S450 M03\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        low = [r for r in records if r["extraction_confidence"] == "LOW"]
        assert len(low) > 0

    def test_confidence_field_only_contains_valid_values(self, sample_nc_file):
        valid = {"HIGH", "MEDIUM", "LOW"}
        for rec in parse_file(sample_nc_file):
            assert rec["extraction_confidence"] in valid


class TestPrevNextLines:
    def test_prev_line_captured(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        assert all("prev_line" in r for r in records)

    def test_next_line_captured(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        assert all("next_line" in r for r in records)

    def test_first_line_prev_is_empty_or_non_sf(self, tmp_path):
        prog = "G97 S450 M03\nG01 Z-1.0 F0.010\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        first_rec = records[0]
        # prev_line of line 1 should be empty string (no line before it)
        assert first_rec["prev_line"] == ""


class TestToolDescriptionLookback:
    def test_description_captured_from_preceding_comment(self, sample_lookback_nc_file):
        records = parse_file(sample_lookback_nc_file)
        descs = {r["tool_description"] for r in records if r.get("tool_description")}
        assert any("1/2 DRILL BIT" in d for d in descs)

    def test_description_on_t_code_line_preferred(self, tmp_path):
        prog = "(PRECEDING COMMENT)\nT0101 (ON-LINE DESC)\nG97 S450 M03\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        descs = {r["tool_description"] for r in records if r.get("tool_description")}
        assert "ON-LINE DESC" in descs


class TestActiveToolTracking:
    def test_t_code_inherited_by_subsequent_lines(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        # All records with F value for T0101 should carry that T code
        t01_f = [r for r in records if r["active_t_code"] == "T0101" and r["f_value"] is not None]
        assert len(t01_f) > 0

    def test_tool_changes_update_active_t_code(self, sample_nc_file):
        records = parse_file(sample_nc_file)
        t_codes = {r["active_t_code"] for r in records if r.get("active_t_code")}
        assert "T0101" in t_codes
        assert "T0202" in t_codes
        assert "T0303" in t_codes

    def test_lines_since_t_code_is_zero_on_t_code_line(self, tmp_path):
        prog = "T0101 G97 S450 M03\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        t_line_recs = [r for r in records if r["active_t_code"] == "T0101" and r["s_value"] == 450.0]
        assert t_line_recs and t_line_recs[0]["lines_since_t_code"] == 0

    def test_lines_since_t_code_increases(self, tmp_path):
        prog = "T0101\nG97 S450 M03\nG01 Z-1.0 F0.010\nM30\n"
        nc = tmp_path / "t.NC"
        nc.write_text(prog)
        records = parse_file(nc)
        assert records[0]["lines_since_t_code"] == 1   # S line is 1 line after T
        f_recs = [r for r in records if r["f_value"] is not None]
        assert f_recs[0]["lines_since_t_code"] > 1

    def test_op1_file_parsed_same_as_nc(self, sample_op1_file):
        records = parse_file(sample_op1_file)
        assert len(records) > 0
        s_vals = {r["s_value"] for r in records if r["s_value"] is not None}
        assert 900.0 in s_vals
