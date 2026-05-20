"""
test_tooling_matcher.py — Tests for the tooling_matcher module.
"""

import pytest
import pandas as pd

from src.tooling_matcher import (
    match_tool_usage,
    export_tooling_review,
    _extract_machine_id,
    _most_common,
    _determine_match_status,
)


# ---------------------------------------------------------------------------
# _extract_machine_id
# ---------------------------------------------------------------------------

class TestExtractMachineId:
    def test_plain_folder_number(self):
        assert _extract_machine_id("655") == "655"

    def test_folder_with_name_suffix(self):
        assert _extract_machine_id("655 Haas VF3") == "655"

    def test_four_digit(self):
        assert _extract_machine_id("1234") == "1234"

    def test_none_returns_none(self):
        assert _extract_machine_id(None) is None

    def test_no_leading_number_returns_none(self):
        assert _extract_machine_id("Haas Machine") is None

    def test_float_returns_none(self):
        assert _extract_machine_id(float("nan")) is None


# ---------------------------------------------------------------------------
# _most_common
# ---------------------------------------------------------------------------

class TestMostCommon:
    def test_single_value(self):
        assert _most_common(["1/2"]) == "1/2"

    def test_majority_wins(self):
        assert _most_common(["1/2", "1/2", "3/8"]) == "1/2"

    def test_ignores_none(self):
        assert _most_common([None, "1/2", None]) == "1/2"

    def test_ignores_blank_string(self):
        assert _most_common(["", "DRILL", ""]) == "DRILL"

    def test_all_none_returns_none(self):
        assert _most_common([None, None]) is None

    def test_empty_list_returns_none(self):
        assert _most_common([]) is None


# ---------------------------------------------------------------------------
# _determine_match_status
# ---------------------------------------------------------------------------

class TestDetermineMatchStatus:
    def test_no_reference_data(self):
        assert _determine_match_status("1/2", None, False) == "no_reference_data"

    def test_missing_from_reference(self):
        assert _determine_match_status("1/2", None, True) == "missing_from_reference"

    def test_needs_review(self):
        ref = {"description": "1/2", "needs_review": True}
        assert _determine_match_status("1/2", ref, True) == "needs_review"

    def test_no_description_in_reference(self):
        ref = {"description": None, "needs_review": False}
        assert _determine_match_status("1/2", ref, True) == "no_description_in_reference"

    def test_no_program_description(self):
        ref = {"description": "1/2", "needs_review": False}
        assert _determine_match_status(None, ref, True) == "no_program_description"

    def test_description_match_case_insensitive(self):
        ref = {"description": "1/2", "needs_review": False}
        assert _determine_match_status("1/2", ref, True) == "description_match"

    def test_description_match_strips_whitespace(self):
        ref = {"description": "WOODRUFF", "needs_review": False}
        assert _determine_match_status("  WOODRUFF  ", ref, True) == "description_match"

    def test_description_differs(self):
        ref = {"description": "1/2", "needs_review": False}
        assert _determine_match_status("3/8", ref, True) == "description_differs"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REFERENCE = [
    {
        "machine_id": "655",
        "machine_label": "655 Haas Endmill Sizes",
        "tool_number": 1,
        "tool_number_raw": "1",
        "description": "1/2",
        "decimal_size": 0.5,
        "needs_review": False,
    },
    {
        "machine_id": "655",
        "machine_label": "655 Haas Endmill Sizes",
        "tool_number": 17,
        "tool_number_raw": "17",
        "description": "WOODRUFF",
        "decimal_size": None,
        "needs_review": False,
    },
    {
        "machine_id": "655",
        "machine_label": "655 Haas Endmill Sizes",
        "tool_number": 3,
        "tool_number_raw": "3",
        "description": None,
        "decimal_size": None,
        "needs_review": True,
    },
    {
        "machine_id": "432",
        "machine_label": "432 MAZAK LATHE Endmill Sizes",
        "tool_number": 1,
        "tool_number_raw": "10.6/11.6",
        "description": "1/2",
        "decimal_size": 0.5,
        "needs_review": True,
    },
    {
        "machine_id": "654",
        "machine_label": "654 OKUMA Tool List",
        "tool_number": 1,
        "tool_number_raw": "1",
        "description": "1/2",
        "decimal_size": 0.5,
        "needs_review": False,
    },
]


def _make_cuts(**kwargs) -> pd.DataFrame:
    defaults = {
        "machine_folder": "655",
        "tool_number": 1,
        "active_t_code": "T01",
        "tool_description": "1/2",
        "s_value": 5000.0,
        "f_value": 0.016,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# match_tool_usage — basic output shape
# ---------------------------------------------------------------------------

class TestMatchToolUsageShape:
    def test_returns_dataframe(self):
        cuts = _make_cuts()
        result = match_tool_usage(cuts, REFERENCE)
        assert isinstance(result, pd.DataFrame)

    def test_empty_cuts_returns_empty(self):
        result = match_tool_usage(pd.DataFrame(), REFERENCE)
        assert result.empty

    def test_output_has_required_columns(self):
        cuts = _make_cuts()
        result = match_tool_usage(cuts, REFERENCE)
        for col in [
            "machine_folder", "tool_number", "active_t_code",
            "program_description", "reference_description",
            "decimal_size", "match_status", "reference_needs_review",
            "review_action", "corrected_description", "notes",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_unique_tool(self):
        cuts = pd.DataFrame([
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": "1/2"},
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": "1/2"},
        ])
        result = match_tool_usage(cuts, REFERENCE)
        assert len(result) == 1

    def test_two_tools_give_two_rows(self):
        cuts = pd.DataFrame([
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": "1/2"},
            {"machine_folder": "655", "tool_number": 17, "active_t_code": "T17",
             "tool_description": "WOODRUFF"},
        ])
        result = match_tool_usage(cuts, REFERENCE)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# match_tool_usage — match_status values
# ---------------------------------------------------------------------------

class TestMatchStatus:
    def test_description_match(self):
        cuts = _make_cuts(machine_folder="655", tool_number=1, tool_description="1/2")
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "description_match"

    def test_description_differs(self):
        cuts = _make_cuts(machine_folder="655", tool_number=1, tool_description="3/8")
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "description_differs"

    def test_no_program_description(self):
        cuts = _make_cuts(machine_folder="655", tool_number=1, tool_description=None)
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "no_program_description"

    def test_missing_from_reference(self):
        cuts = _make_cuts(machine_folder="655", tool_number=99, tool_description="SOME TOOL")
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "missing_from_reference"

    def test_no_reference_data_for_unknown_machine(self):
        cuts = _make_cuts(machine_folder="417", tool_number=1, tool_description="1/2")
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "no_reference_data"

    def test_needs_review_for_reference_needs_review(self):
        cuts = _make_cuts(machine_folder="655", tool_number=3, tool_description="3/8")
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "needs_review"

    def test_mazak_always_needs_review(self):
        cuts = _make_cuts(machine_folder="432", tool_number=1, tool_description="1/2")
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "needs_review"

    def test_no_description_in_reference(self):
        # T3 in 655 Haas has needs_review=True, but test a synthetic ref with no desc
        ref = [{
            "machine_id": "655", "machine_label": "655 Haas", "tool_number": 7,
            "tool_number_raw": "7", "description": None, "decimal_size": None,
            "needs_review": False,
        }]
        cuts = _make_cuts(machine_folder="655", tool_number=7, tool_description="1/4")
        result = match_tool_usage(cuts, ref)
        assert result.iloc[0]["match_status"] == "no_description_in_reference"

    def test_description_match_case_insensitive(self):
        cuts = _make_cuts(machine_folder="655", tool_number=17, tool_description="woodruff")
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "description_match"


# ---------------------------------------------------------------------------
# match_tool_usage — reference fields propagated
# ---------------------------------------------------------------------------

class TestReferenceFieldsPropagated:
    def test_reference_description_populated(self):
        cuts = _make_cuts(machine_folder="655", tool_number=1)
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["reference_description"] == "1/2"

    def test_decimal_size_populated(self):
        cuts = _make_cuts(machine_folder="655", tool_number=1)
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["decimal_size"] == 0.5

    def test_reference_needs_review_true_for_mazak(self):
        cuts = _make_cuts(machine_folder="432", tool_number=1, tool_description="1/2")
        result = match_tool_usage(cuts, REFERENCE)
        assert bool(result.iloc[0]["reference_needs_review"]) is True

    def test_reference_needs_review_false_for_haas_t1(self):
        cuts = _make_cuts(machine_folder="655", tool_number=1)
        result = match_tool_usage(cuts, REFERENCE)
        assert bool(result.iloc[0]["reference_needs_review"]) is False

    def test_reference_fields_none_when_missing(self):
        cuts = _make_cuts(machine_folder="655", tool_number=99)
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["reference_description"] is None
        assert result.iloc[0]["decimal_size"] is None


# ---------------------------------------------------------------------------
# match_tool_usage — review columns are blank
# ---------------------------------------------------------------------------

class TestReviewColumnsBlank:
    def test_review_action_blank(self):
        cuts = _make_cuts()
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["review_action"] == ""

    def test_corrected_description_blank(self):
        cuts = _make_cuts()
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["corrected_description"] == ""

    def test_notes_blank(self):
        cuts = _make_cuts()
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["notes"] == ""


# ---------------------------------------------------------------------------
# match_tool_usage — most common description aggregation
# ---------------------------------------------------------------------------

class TestDescriptionAggregation:
    def test_picks_most_common_description(self):
        cuts = pd.DataFrame([
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": "1/2"},
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": "1/2"},
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": "1/4"},
        ])
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["program_description"] == "1/2"

    def test_none_descriptions_ignored(self):
        cuts = pd.DataFrame([
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": None},
            {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
             "tool_description": "1/2"},
        ])
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["program_description"] == "1/2"


# ---------------------------------------------------------------------------
# match_tool_usage — no tool_description column
# ---------------------------------------------------------------------------

class TestNoProgramDescription:
    def test_works_without_tool_description_column(self):
        cuts = pd.DataFrame([{
            "machine_folder": "655",
            "tool_number": 1,
            "active_t_code": "T01",
        }])
        result = match_tool_usage(cuts, REFERENCE)
        assert result.iloc[0]["match_status"] == "no_program_description"
        assert result.iloc[0]["program_description"] is None


# ---------------------------------------------------------------------------
# export_tooling_review
# ---------------------------------------------------------------------------

class TestExportToolingReview:
    def test_creates_csv(self, tmp_path):
        cuts = _make_cuts()
        path = export_tooling_review(cuts, REFERENCE, tmp_path, timestamp="20260101_120000")
        assert path.exists()
        assert path.suffix == ".csv"

    def test_filename_prefix(self, tmp_path):
        cuts = _make_cuts()
        path = export_tooling_review(cuts, REFERENCE, tmp_path, timestamp="20260101_120000")
        assert path.name.startswith("tooling_review_")

    def test_timestamp_in_filename(self, tmp_path):
        cuts = _make_cuts()
        path = export_tooling_review(cuts, REFERENCE, tmp_path, timestamp="20260101_120000")
        assert "20260101_120000" in path.name

    def test_csv_has_match_status(self, tmp_path):
        cuts = _make_cuts()
        path = export_tooling_review(cuts, REFERENCE, tmp_path, timestamp="20260101_120000")
        df = pd.read_csv(path)
        assert "match_status" in df.columns

    def test_csv_has_review_columns(self, tmp_path):
        cuts = _make_cuts()
        path = export_tooling_review(cuts, REFERENCE, tmp_path, timestamp="20260101_120000")
        df = pd.read_csv(path)
        for col in ["review_action", "corrected_description", "notes"]:
            assert col in df.columns

    def test_auto_timestamp_creates_file(self, tmp_path):
        cuts = _make_cuts()
        path = export_tooling_review(cuts, REFERENCE, tmp_path)
        assert path.exists()
