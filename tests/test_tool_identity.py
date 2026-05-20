"""Tests for src/dashboard/data_access/tool_identity.py"""

import pandas as pd
import pytest

from src.dashboard.data_access.tool_identity import (
    _normalize_machine_id,
    _to_tool_str,
    _is_usable,
    resolve_tool_identity,
    resolve_tool_identity_df,
    SOURCE_OVERRIDE,
    SOURCE_TOOLDB_ASSEMBLY,
    SOURCE_TOOLDB_FALLBACK,
    SOURCE_PROGRAM,
    SOURCE_EXCEL,
    SOURCE_UNKNOWN,
)

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_normalize_plain_number():
    assert _normalize_machine_id("655") == "655"


def test_normalize_folder_with_text():
    assert _normalize_machine_id("655 Haas") == "655"


def test_normalize_float():
    assert _normalize_machine_id(655.0) == "655"


def test_normalize_none():
    assert _normalize_machine_id(None) == ""


def test_normalize_nan():
    assert _normalize_machine_id(float("nan")) == ""


def test_normalize_no_digits():
    assert _normalize_machine_id("Haas") == "Haas"


def test_to_tool_str_int():
    assert _to_tool_str(1) == "1"


def test_to_tool_str_float():
    assert _to_tool_str(1.0) == "1"


def test_to_tool_str_string():
    assert _to_tool_str("5") == "5"


def test_to_tool_str_none():
    assert _to_tool_str(None) == ""


def test_to_tool_str_nan():
    assert _to_tool_str(float("nan")) == ""


def test_is_usable_plain():
    assert _is_usable("ENDMILL") is True


def test_is_usable_empty():
    assert _is_usable("") is False


def test_is_usable_none():
    assert _is_usable(None) is False


def test_is_usable_nan_float():
    assert _is_usable(float("nan")) is False


def test_is_usable_nan_string():
    assert _is_usable("nan") is False


# ---------------------------------------------------------------------------
# resolve_tool_identity — priority chain
# ---------------------------------------------------------------------------

_ASSEMBLY_REC = {
    "extraction_method": "assembly_join",
    "tool_name": "1/2 ENDMILL",
    "holder_name": "CAT40 ER32",
}

_FALLBACK_REC = {
    "extraction_method": "lathe_fallback",
    "tool_name": "DCMT 3|2.5|1-MF",
    "holder_name": "",
}

_OVERRIDE = {"corrected_description": "MY CORRECTED NAME", "notes": "verified 2025"}


def test_override_beats_tooldb():
    result = resolve_tool_identity(
        tooldb_records=[_ASSEMBLY_REC],
        override=_OVERRIDE,
    )
    assert result["resolved_tool_name"] == "MY CORRECTED NAME"
    assert result["resolved_tool_source"] == SOURCE_OVERRIDE


def test_override_beats_program():
    result = resolve_tool_identity(
        parsed_description="PROGRAM TOOL",
        override=_OVERRIDE,
    )
    assert result["resolved_tool_source"] == SOURCE_OVERRIDE


def test_override_confidence_high():
    result = resolve_tool_identity(override=_OVERRIDE)
    assert result["resolved_tool_confidence"] == "HIGH"
    assert result["resolved_tool_needs_review"] is False


def test_override_notes_in_description():
    result = resolve_tool_identity(override=_OVERRIDE)
    assert result["resolved_tool_description"] == "verified 2025"


def test_override_empty_falls_through_to_tooldb():
    result = resolve_tool_identity(
        tooldb_records=[_ASSEMBLY_REC],
        override={"corrected_description": "", "notes": ""},
    )
    assert result["resolved_tool_source"] == SOURCE_TOOLDB_ASSEMBLY


def test_tooldb_assembly_beats_fallback():
    result = resolve_tool_identity(
        tooldb_records=[_ASSEMBLY_REC, _FALLBACK_REC],
    )
    assert result["resolved_tool_source"] == SOURCE_TOOLDB_ASSEMBLY
    assert result["resolved_tool_name"] == "1/2 ENDMILL"


def test_tooldb_assembly_beats_program():
    result = resolve_tool_identity(
        parsed_description="PROGRAM TOOL",
        tooldb_records=[_ASSEMBLY_REC],
    )
    assert result["resolved_tool_source"] == SOURCE_TOOLDB_ASSEMBLY


def test_tooldb_assembly_confidence_high():
    result = resolve_tool_identity(tooldb_records=[_ASSEMBLY_REC])
    assert result["resolved_tool_confidence"] == "HIGH"


def test_tooldb_assembly_not_needs_review():
    result = resolve_tool_identity(tooldb_records=[_ASSEMBLY_REC])
    assert result["resolved_tool_needs_review"] is False


def test_tooldb_assembly_holder_in_description():
    result = resolve_tool_identity(tooldb_records=[_ASSEMBLY_REC])
    assert result["resolved_tool_description"] == "CAT40 ER32"


def test_tooldb_fallback_beats_program():
    result = resolve_tool_identity(
        parsed_description="PROGRAM TOOL",
        tooldb_records=[_FALLBACK_REC],
    )
    assert result["resolved_tool_source"] == SOURCE_TOOLDB_FALLBACK


def test_tooldb_fallback_confidence_medium():
    result = resolve_tool_identity(tooldb_records=[_FALLBACK_REC])
    assert result["resolved_tool_confidence"] == "MEDIUM"


def test_tooldb_fallback_needs_review_true():
    result = resolve_tool_identity(tooldb_records=[_FALLBACK_REC])
    assert result["resolved_tool_needs_review"] is True


def test_tooldb_fallback_source_label():
    result = resolve_tool_identity(tooldb_records=[_FALLBACK_REC])
    assert result["resolved_tool_source"] == SOURCE_TOOLDB_FALLBACK


def test_tooldb_empty_name_falls_through():
    rec = {"extraction_method": "assembly_join", "tool_name": "", "holder_name": ""}
    result = resolve_tool_identity(
        parsed_description="PROGRAM TOOL",
        tooldb_records=[rec],
    )
    assert result["resolved_tool_source"] == SOURCE_PROGRAM


def test_program_beats_excel():
    result = resolve_tool_identity(
        parsed_description="PROGRAM TOOL",
        reference_description="EXCEL TOOL",
    )
    assert result["resolved_tool_source"] == SOURCE_PROGRAM
    assert result["resolved_tool_name"] == "PROGRAM TOOL"


def test_program_confidence_medium():
    result = resolve_tool_identity(parsed_description="ENDMILL")
    assert result["resolved_tool_confidence"] == "MEDIUM"
    assert result["resolved_tool_needs_review"] is False


def test_excel_beats_unknown():
    result = resolve_tool_identity(reference_description="EXCEL TOOL")
    assert result["resolved_tool_source"] == SOURCE_EXCEL
    assert result["resolved_tool_name"] == "EXCEL TOOL"


def test_excel_confidence_medium():
    result = resolve_tool_identity(reference_description="EXCEL TOOL")
    assert result["resolved_tool_confidence"] == "MEDIUM"


def test_unknown_when_all_empty():
    result = resolve_tool_identity()
    assert result["resolved_tool_source"] == SOURCE_UNKNOWN
    assert result["resolved_tool_name"] == "UNKNOWN TOOL"
    assert result["resolved_tool_confidence"] == "LOW"
    assert result["resolved_tool_needs_review"] is True


def test_unknown_needs_review_true():
    result = resolve_tool_identity()
    assert result["resolved_tool_needs_review"] is True


# ---------------------------------------------------------------------------
# resolve_tool_identity_df — DataFrame-level
# ---------------------------------------------------------------------------

def _make_df(**rows) -> pd.DataFrame:
    """Build a minimal DataFrame from column→list-of-values kwargs."""
    return pd.DataFrame(rows)


def _tooldb_ref_df(machine_id, tool_number, tool_name, extraction_method="assembly_join") -> pd.DataFrame:
    return pd.DataFrame([{
        "machine_id": machine_id,
        "tool_number": tool_number,
        "tool_name": tool_name,
        "holder_name": "",
        "extraction_method": extraction_method,
        "confidence": "HIGH" if extraction_method == "assembly_join" else "MEDIUM",
        "needs_review": False if extraction_method == "assembly_join" else True,
    }])


def test_resolve_df_adds_resolved_columns():
    df = _make_df(machine_folder=["655"], tool_number=[1])
    result = resolve_tool_identity_df(df)
    for col in ("resolved_tool_name", "resolved_tool_source",
                "resolved_tool_confidence", "resolved_tool_needs_review"):
        assert col in result.columns


def test_resolve_df_tooldb_match():
    df = _make_df(machine_folder=["655"], tool_number=[1])
    tooldb = _tooldb_ref_df("655", 1, "1/2 ENDMILL")
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    assert result.iloc[0]["resolved_tool_name"] == "1/2 ENDMILL"
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_TOOLDB_ASSEMBLY


def test_resolve_df_machine_id_normalized():
    """machine_folder='655 Haas' should match machine_id='655' in tooldb."""
    df = _make_df(machine_folder=["655 Haas"], tool_number=[1])
    tooldb = _tooldb_ref_df("655", 1, "ENDMILL")
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_TOOLDB_ASSEMBLY


def test_resolve_df_fallback_to_program():
    """No TOOLDB match → use program_description."""
    df = _make_df(
        machine_folder=["655"], tool_number=[99],
        program_description=["DRILL"],
    )
    result = resolve_tool_identity_df(df)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_PROGRAM
    assert result.iloc[0]["resolved_tool_name"] == "DRILL"


def test_resolve_df_override_beats_tooldb():
    df = _make_df(
        machine_folder=["655"], tool_number=[1],
        corrected_description=["MY OVERRIDE"],
    )
    tooldb = _tooldb_ref_df("655", 1, "TOOLDB NAME")
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_OVERRIDE
    assert result.iloc[0]["resolved_tool_name"] == "MY OVERRIDE"


def test_resolve_df_excel_reference_description():
    """reference_description column used as EXCEL source."""
    df = _make_df(
        machine_folder=["655"], tool_number=[99],
        reference_description=["EXCEL DRILL"],
    )
    result = resolve_tool_identity_df(df)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_EXCEL


def test_resolve_df_unknown_when_all_empty():
    df = _make_df(machine_folder=["655"], tool_number=[99])
    result = resolve_tool_identity_df(df)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_UNKNOWN


def test_resolve_df_lathe_fallback_needs_review():
    df = _make_df(machine_folder=["432"], tool_number=[8])
    tooldb = _tooldb_ref_df("432", 8, "DCMT INSERT", extraction_method="lathe_fallback")
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_TOOLDB_FALLBACK
    assert result.iloc[0]["resolved_tool_needs_review"] is True


def test_resolve_df_assembly_not_needs_review():
    df = _make_df(machine_folder=["655"], tool_number=[1])
    tooldb = _tooldb_ref_df("655", 1, "ENDMILL", extraction_method="assembly_join")
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    assert result.iloc[0]["resolved_tool_needs_review"] is False


def test_resolve_df_python_bool_needs_review():
    """needs_review should be Python bool, not numpy.bool_."""
    df = _make_df(machine_folder=["432"], tool_number=[8])
    tooldb = _tooldb_ref_df("432", 8, "INSERT", extraction_method="lathe_fallback")
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    val = result.iloc[0]["resolved_tool_needs_review"]
    assert val is True  # Python bool identity check


def test_resolve_df_external_overrides_lookup():
    """External overrides DataFrame takes priority over TOOLDB."""
    df = _make_df(machine_folder=["655"], tool_number=[1])
    tooldb = _tooldb_ref_df("655", 1, "TOOLDB NAME")
    overrides = pd.DataFrame([{
        "machine_folder": "655",
        "tool_number": 1,
        "corrected_description": "EXTERNAL OVERRIDE",
        "notes": "",
    }])
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb, overrides=overrides)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_OVERRIDE
    assert result.iloc[0]["resolved_tool_name"] == "EXTERNAL OVERRIDE"


def test_resolve_df_empty_df_returns_empty():
    result = resolve_tool_identity_df(pd.DataFrame())
    assert result.empty


def test_resolve_df_none_df_returns_none():
    result = resolve_tool_identity_df(None)
    assert result is None


def test_resolve_df_multiple_rows():
    df = _make_df(
        machine_folder=["655", "432", "655"],
        tool_number=[1, 8, 99],
        program_description=["", "", "SPARE DRILL"],
    )
    tooldb = pd.concat([
        _tooldb_ref_df("655", 1, "ENDMILL", "assembly_join"),
        _tooldb_ref_df("432", 8, "INSERT", "lathe_fallback"),
    ], ignore_index=True)
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_TOOLDB_ASSEMBLY
    assert result.iloc[1]["resolved_tool_source"] == SOURCE_TOOLDB_FALLBACK
    assert result.iloc[2]["resolved_tool_source"] == SOURCE_PROGRAM


def test_resolve_df_tool_number_float_matches_int():
    """tool_number=1.0 in df should match tool_number=1 in tooldb."""
    df = _make_df(machine_folder=["655"], tool_number=[1.0])
    tooldb = _tooldb_ref_df("655", 1, "ENDMILL")
    result = resolve_tool_identity_df(df, tooldb_ref=tooldb)
    assert result.iloc[0]["resolved_tool_source"] == SOURCE_TOOLDB_ASSEMBLY


def test_resolve_df_original_columns_preserved():
    """resolve_tool_identity_df should not drop existing columns."""
    df = _make_df(
        machine_folder=["655"], tool_number=[1],
        record_count=[42], s_mean=[350.0],
    )
    result = resolve_tool_identity_df(df)
    assert "record_count" in result.columns
    assert result.iloc[0]["record_count"] == 42
