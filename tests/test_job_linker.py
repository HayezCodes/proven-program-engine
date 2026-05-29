"""
test_job_linker.py — Tests for Phase 6B program–job linker.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.job_linker import (
    _MATERIAL_BACKFILL_COLS,
    _PROG_JOB_LINK_COLS,
    _ROUTER_CONTEXT_COLS,
    build_indexes,
    build_material_backfill,
    build_program_job_links,
    build_router_program_context,
    detect_latest_exports,
    export_material_backfill,
    export_program_job_links,
    export_router_program_context,
    filename_tokens,
    resolve_material,
    run_job_link,
)

# ---------------------------------------------------------------------------
# Synthetic test data builders
# ---------------------------------------------------------------------------

def _manifest(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "program_id": 0, "source_file": "", "filename": "",
        "machine_folder": "", "included": True,
        "relative_path": "", "extension": "", "modified_datetime": "",
        "file_size_bytes": 0, "skip_reason": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _job_meta(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "source_file": "", "filename": "", "modified_datetime": "",
        "job_number": "", "part_number": "", "drawing_number": "",
        "revision": "", "material": "", "work_centers": "",
        "operation_count": 0, "extraction_confidence": "LOW",
        "extraction_notes": "",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _prints(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "source_file": "", "filename": "", "modified_datetime": "",
        "part_number": "", "drawing_number": "", "revision": "",
        "material": "", "file_size_bytes": 0,
        "extraction_confidence": "LOW", "extraction_notes": "",
    }
    if not rows:
        return pd.DataFrame(columns=list(defaults.keys()))
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _router_ops(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "source_file": "", "job_number": "", "operation_sequence": 1,
        "operation_number": "10", "work_center": "", "machine": "",
        "operation_description": "", "operation_notes": "",
    }
    if not rows:
        return pd.DataFrame(columns=list(defaults.keys()))
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _cuts(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "record_id": 0, "program_id": 0, "source_file": "",
        "machine_folder": "", "filename": "", "line_number": 1,
        "active_t_code": "T0101", "tool_number": "01",
        "tool_description": "", "s_value": 500.0, "s_mode": "CSS",
        "s_type": "SPINDLE", "f_value": 0.012, "f_mode": "IPR",
        "block_skip": False, "lines_since_t_code": 1,
        "extraction_confidence": "HIGH", "raw_line": "G96 S500 F0.012",
        "prev_line": "", "next_line": "", "context_json": "[]",
        "sf_combo_count": 1, "is_duplicate": False,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# filename_tokens
# ---------------------------------------------------------------------------

class TestFilenameTokens:
    def test_em_prefix_produces_stem_and_numeric(self):
        toks = filename_tokens("EM10986.NC")
        assert "EM10986" in toks or "em10986" in toks
        assert "10986" in toks

    def test_pure_numeric_strips_leading_zeros(self):
        toks = filename_tokens("0262.TAP")
        assert "0262" in toks
        assert "262" in toks

    def test_hyphenated_em_includes_prefix(self):
        toks = filename_tokens("EM0170-HEX.NC")
        assert any("0170" in t or "EM0170" in t for t in toks)

    def test_returns_list_not_empty(self):
        assert len(filename_tokens("PART001.NC")) > 0

    def test_deduplicates_tokens(self):
        toks = filename_tokens("0100.NC")
        assert len(toks) == len(set(toks))

    def test_eight_digit_produces_prefix_slices(self):
        toks = filename_tokens("10007001.EIA")
        assert "10007" in toks

    def test_no_extension_stem_only(self):
        toks = filename_tokens("NOEXT")
        assert "NOEXT" in toks or "noext" in toks


# ---------------------------------------------------------------------------
# resolve_material
# ---------------------------------------------------------------------------

class TestResolveMaterial:
    def test_job_material_returns_router_source(self):
        mat, src, conf, conflict = resolve_material(
            [{"material": "4140 ALLOY STEEL"}], []
        )
        assert mat == "4140 ALLOY STEEL"
        assert src == "ROUTER"
        assert conf == "HIGH"
        assert conflict == False

    def test_print_material_when_no_job_material(self):
        mat, src, conf, conflict = resolve_material(
            [{"material": ""}], [{"material": "316 SS"}]
        )
        assert mat == "316 SS"
        assert src == "SHARED_PRINT"
        assert conf == "MEDIUM"

    def test_no_material_returns_unknown(self):
        mat, src, conf, conflict = resolve_material([], [])
        assert mat == "UNKNOWN"
        assert src == "UNKNOWN"
        assert conf == "NONE"
        assert conflict == False

    def test_conflict_detected_when_job_and_print_differ(self):
        mat, src, conf, conflict = resolve_material(
            [{"material": "4140"}], [{"material": "316"}]
        )
        assert conflict == True
        assert mat == "4140"  # job takes priority

    def test_no_conflict_when_materials_match(self):
        mat, src, conf, conflict = resolve_material(
            [{"material": "4140"}], [{"material": "4140"}]
        )
        assert conflict == False

    def test_empty_job_material_falls_through_to_print(self):
        mat, src, conf, conflict = resolve_material(
            [{"material": ""}], [{"material": "TITANIUM"}]
        )
        assert mat == "TITANIUM"
        assert src == "SHARED_PRINT"


# ---------------------------------------------------------------------------
# Exact job number match
# ---------------------------------------------------------------------------

class TestExactJobNumberMatch:
    def test_links_program_by_job_number(self):
        manifest  = _manifest([{"program_id": 1, "source_file": "P:/655/0262.NC",
                                 "filename": "0262.NC", "machine_folder": "655"}])
        jobs      = _job_meta([{"source_file": "G:/jobs/t.txt", "job_number": "262",
                                 "part_number": "PART-A", "material": "4140"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert len(df) == 1
        assert df.iloc[0]["link_method"] == "exact_job_number"
        assert df.iloc[0]["link_confidence"] == "HIGH"
        assert df.iloc[0]["matched_job_number"] == "262"

    def test_material_populated_from_job(self):
        manifest  = _manifest([{"program_id": 1, "source_file": "P:/1/0100.NC",
                                 "filename": "0100.NC", "machine_folder": "655"}])
        jobs      = _job_meta([{"job_number": "100", "material": "316 STAINLESS"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["material"] == "316 STAINLESS"
        assert df.iloc[0]["material_source"] == "ROUTER"
        assert df.iloc[0]["material_confidence"] == "HIGH"

    def test_excluded_programs_not_linked(self):
        manifest = _manifest([
            {"program_id": 1, "filename": "0262.NC", "included": True},
            {"program_id": 2, "filename": "0263.NC", "included": False},
        ])
        jobs = _job_meta([{"job_number": "262"}, {"job_number": "263"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert len(df) == 1
        assert df.iloc[0]["filename"] == "0262.NC"


# ---------------------------------------------------------------------------
# Exact drawing number match
# ---------------------------------------------------------------------------

class TestExactDrawingNumberMatch:
    def test_em_filename_matches_drawing_number(self):
        manifest = _manifest([{"program_id": 1, "source_file": "P:/655/EM10986.NC",
                                "filename": "EM10986.NC", "machine_folder": "655"}])
        jobs     = _job_meta([{"job_number": "5001", "drawing_number": "EM10986",
                                "material": "4140"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["link_method"] == "exact_drawing_number"
        assert df.iloc[0]["link_confidence"] == "HIGH"
        assert df.iloc[0]["matched_drawing_number"] == "EM10986"

    def test_drawing_takes_priority_over_job_number(self):
        # Program EM10986.NC should match on drawing_number, not job_number
        manifest = _manifest([{"program_id": 1, "filename": "EM10986.NC"}])
        jobs     = _job_meta([
            {"job_number": "10986",  "drawing_number": "",       "material": "WRONG"},
            {"job_number": "9999",   "drawing_number": "EM10986","material": "CORRECT"},
        ])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["link_method"] == "exact_drawing_number"
        assert df.iloc[0]["material"] == "CORRECT"

    def test_case_insensitive_drawing_match(self):
        manifest = _manifest([{"program_id": 1, "filename": "em10986.NC"}])
        jobs     = _job_meta([{"drawing_number": "EM10986", "material": "4140"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["link_method"] == "exact_drawing_number"


# ---------------------------------------------------------------------------
# Exact part number match
# ---------------------------------------------------------------------------

class TestExactPartNumberMatch:
    def test_links_by_part_number(self):
        manifest = _manifest([{"program_id": 1, "filename": "SS316FLANGE.NC"}])
        jobs     = _job_meta([{"part_number": "SS316FLANGE", "material": "316 SS"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["link_method"] == "exact_part_number"
        assert df.iloc[0]["link_confidence"] == "HIGH"

    def test_part_number_match_with_numeric_normalization(self):
        manifest = _manifest([{"program_id": 1, "filename": "0825.NC"}])
        # part_number "825" normalizes to "825"; "0825" also normalizes to "825"
        jobs     = _job_meta([{"part_number": "825", "material": "17-4 PH"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        # Either job_number, drawing_number, or part_number match — accept any that links
        assert df.iloc[0]["link_confidence"] in ("HIGH", "MEDIUM")
        assert df.iloc[0]["material"] == "17-4 PH"


# ---------------------------------------------------------------------------
# Shared print bridge match
# ---------------------------------------------------------------------------

class TestSharedPrintBridgeMatch:
    def test_bridge_via_drawing_number(self):
        manifest = _manifest([{"program_id": 1, "source_file": "P:/655/EM0825.NC",
                                "filename": "EM0825.NC", "machine_folder": "655"}])
        prints   = _prints([{"source_file": "G:/prints/EM0825.pdf",
                              "drawing_number": "EM0825", "part_number": "PART-825",
                              "material": "4140", "revision": "B"}])
        df = build_program_job_links(manifest, _job_meta([]), prints, _router_ops([]))
        assert df.iloc[0]["link_method"] == "shared_print_bridge"
        assert df.iloc[0]["link_confidence"] == "MEDIUM"
        assert df.iloc[0]["matched_shared_print_file"] != ""
        assert "4140" in df.iloc[0]["material"]

    def test_bridge_then_job_resolved(self):
        manifest = _manifest([{"program_id": 1, "filename": "EM0825.NC"}])
        jobs     = _job_meta([{"job_number": "9001", "drawing_number": "EM0825",
                                "material": "316 STAINLESS"}])
        prints   = _prints([{"drawing_number": "EM0825", "material": "316 SS"}])
        df = build_program_job_links(manifest, jobs, prints, _router_ops([]))
        # Should match drawing via job first (higher priority than print bridge)
        assert df.iloc[0]["link_method"] == "exact_drawing_number"
        assert df.iloc[0]["matched_job_number"] == "9001"

    def test_bridge_material_source_is_shared_print_when_no_job(self):
        manifest = _manifest([{"program_id": 1, "filename": "EM0001.NC"}])
        prints   = _prints([{"drawing_number": "EM0001", "material": "MONEL K-500"}])
        df = build_program_job_links(manifest, _job_meta([]), prints, _router_ops([]))
        assert df.iloc[0]["material_source"] == "SHARED_PRINT"
        assert df.iloc[0]["material_confidence"] == "MEDIUM"


# ---------------------------------------------------------------------------
# Router match
# ---------------------------------------------------------------------------

class TestRouterMatch:
    def test_token_in_operation_description_links(self):
        manifest = _manifest([{"program_id": 1, "filename": "SPEC10007.NC"}])
        jobs     = _job_meta([{"source_file": "G:/j.txt", "job_number": "5555",
                                "material": "4140"}])
        ops      = _router_ops([{"job_number": "5555",
                                  "operation_description": "TURN SPEC10007 PROFILE"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] == "router_match"
        assert df.iloc[0]["link_confidence"] == "LOW"
        assert df.iloc[0]["needs_review"] == True

    def test_token_shorter_than_5_chars_not_matched(self):
        manifest = _manifest([{"program_id": 1, "filename": "001.NC"}])
        jobs     = _job_meta([{"job_number": "9999", "material": "4140"}])
        ops      = _router_ops([{"job_number": "9999",
                                  "operation_description": "CONTAINS 001 SOMEWHERE"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        # "001" has 3 chars < 5 → should not trigger router_match
        assert df.iloc[0]["link_method"] != "router_match"

    def test_ambiguous_router_description_no_match(self):
        manifest = _manifest([{"program_id": 1, "filename": "SPEC10007.NC"}])
        jobs     = _job_meta([
            {"job_number": "1111", "material": "4140"},
            {"job_number": "2222", "material": "316"},
        ])
        ops = _router_ops([
            {"job_number": "1111", "operation_description": "TURN SPEC10007 PROFILE"},
            {"job_number": "2222", "operation_description": "MILL SPEC10007 SLOT"},
        ])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        # Two different jobs in description → router_match not fired; should be no_match
        assert df.iloc[0]["link_method"] not in ("router_match",)


# ---------------------------------------------------------------------------
# Machine context assist
# ---------------------------------------------------------------------------

class TestMachineContextAssist:
    def test_disambiguates_multiple_job_hits(self):
        manifest = _manifest([{"program_id": 1, "filename": "0262.NC",
                                "machine_folder": "655"}])
        jobs = _job_meta([
            {"source_file": "G:/j1.txt", "job_number": "262",
             "part_number": "PART-A", "material": "4140"},
            {"source_file": "G:/j2.txt", "job_number": "262",
             "part_number": "PART-B", "material": "316"},
        ])
        ops = _router_ops([
            {"job_number": "262", "work_center": "655 HAAS",
             "operation_description": "MILL FEATURES",
             "source_file": "G:/j1.txt"},
        ])
        # Only j1 has ops with machine 655 → should disambiguate to j1
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] == "machine_context_assist"
        assert df.iloc[0]["link_confidence"] == "HIGH"

    def test_ambiguous_when_machine_context_fails(self):
        manifest = _manifest([{"program_id": 1, "filename": "0262.NC",
                                "machine_folder": "655"}])
        jobs = _job_meta([
            {"job_number": "262", "part_number": "A", "material": "4140"},
            {"job_number": "262", "part_number": "B", "material": "316"},
        ])
        # No router ops → can't disambiguate
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["link_method"] == "ambiguous_match"
        assert df.iloc[0]["link_confidence"] == "MEDIUM"
        assert df.iloc[0]["needs_review"] == True


# ---------------------------------------------------------------------------
# Ambiguous match handling
# ---------------------------------------------------------------------------

class TestAmbiguousMatch:
    def test_multiple_matches_flag_needs_review(self):
        manifest = _manifest([{"program_id": 1, "filename": "0500.NC",
                                "machine_folder": "417"}])
        jobs = _job_meta([
            {"job_number": "500", "material": "4140"},
            {"job_number": "500", "material": "316"},
        ])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["link_method"] == "ambiguous_match"
        assert df.iloc[0]["needs_review"] == True
        assert df.iloc[0]["material_conflict"] == True

    def test_ambiguous_has_medium_confidence(self):
        manifest = _manifest([{"program_id": 1, "filename": "0500.NC"}])
        jobs = _job_meta([{"job_number": "500"}, {"job_number": "500"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        assert df.iloc[0]["link_confidence"] == "MEDIUM"


# ---------------------------------------------------------------------------
# Material conflict handling
# ---------------------------------------------------------------------------

class TestMaterialConflict:
    def test_conflict_when_job_and_print_differ(self):
        manifest = _manifest([{"program_id": 1, "filename": "EM0001.NC"}])
        jobs     = _job_meta([{"drawing_number": "EM0001", "material": "4140 STEEL"}])
        prints   = _prints([{"drawing_number": "EM0001", "material": "316 SS"}])
        df = build_program_job_links(manifest, jobs, prints, _router_ops([]))
        assert df.iloc[0]["material_conflict"] == True
        assert df.iloc[0]["needs_review"] == True
        assert df.iloc[0]["material"] == "4140 STEEL"  # job wins

    def test_no_conflict_when_materials_agree(self):
        manifest = _manifest([{"program_id": 1, "filename": "EM0001.NC"}])
        jobs     = _job_meta([{"drawing_number": "EM0001", "material": "316 SS"}])
        prints   = _prints([{"drawing_number": "EM0001", "material": "316 SS"}])
        df = build_program_job_links(manifest, jobs, prints, _router_ops([]))
        assert df.iloc[0]["material_conflict"] == False

    def test_unknown_not_treated_as_conflict(self):
        manifest = _manifest([{"program_id": 1, "filename": "EM0001.NC"}])
        jobs     = _job_meta([{"drawing_number": "EM0001", "material": "4140"}])
        prints   = _prints([{"drawing_number": "EM0001", "material": ""}])
        df = build_program_job_links(manifest, jobs, prints, _router_ops([]))
        assert df.iloc[0]["material_conflict"] == False


# ---------------------------------------------------------------------------
# No match handling
# ---------------------------------------------------------------------------

class TestNoMatch:
    def test_unmatched_program_gets_none_confidence(self):
        manifest = _manifest([{"program_id": 1, "filename": "XYZABC.NC"}])
        df = build_program_job_links(manifest, _job_meta([]), _prints([]), _router_ops([]))
        assert df.iloc[0]["link_method"] == "no_match"
        assert df.iloc[0]["link_confidence"] == "NONE"

    def test_unmatched_material_is_unknown(self):
        manifest = _manifest([{"program_id": 1, "filename": "XYZABC.NC"}])
        df = build_program_job_links(manifest, _job_meta([]), _prints([]), _router_ops([]))
        assert df.iloc[0]["material"] == "UNKNOWN"
        assert df.iloc[0]["material_source"] == "UNKNOWN"
        assert df.iloc[0]["material_confidence"] == "NONE"
        assert df.iloc[0]["needs_review"] == False

    def test_empty_manifest_returns_empty_df(self):
        df = build_program_job_links(
            _manifest([]), _job_meta([]), _prints([]), _router_ops([])
        )
        assert df.empty or len(df) == 0

    def test_all_dfs_empty_returns_empty(self):
        df = build_program_job_links(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        assert isinstance(df, pd.DataFrame)
        for col in _PROG_JOB_LINK_COLS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Material backfill builder
# ---------------------------------------------------------------------------

class TestBuildMaterialBackfill:
    def _links(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "source_file":           "P:/655/EM10986.NC",
            "filename":              "EM10986.NC",
            "machine_folder":        "655",
            "program_id":            1,
            "matched_job_number":    "5001",
            "matched_part_number":   "PART-A",
            "matched_drawing_number":"EM10986",
            "matched_revision":      "A",
            "matched_shared_print_file": "",
            "matched_router_file":   "G:/j.txt",
            "link_confidence":       "HIGH",
            "link_method":           "exact_drawing_number",
            "link_reason":           "matched EM10986",
            "material":              "4140 ALLOY STEEL",
            "material_source":       "ROUTER",
            "material_confidence":   "HIGH",
            "material_conflict":     False,
            "needs_review":          False,
        }], columns=_PROG_JOB_LINK_COLS)

    def _cuts_row(self) -> pd.DataFrame:
        return _cuts([{
            "source_file":    "P:/655/EM10986.NC",
            "machine_folder": "655",
            "tool_number":    "01",
            "tool_description": "ROUGHING INSERT",
            "s_value": 500.0, "s_mode": "CSS", "s_type": "SPINDLE",
            "f_value": 0.012, "f_mode": "IPR",
        }])

    def test_returns_dataframe(self):
        df = build_material_backfill(self._cuts_row(), self._links())
        assert isinstance(df, pd.DataFrame)

    def test_has_all_required_columns(self):
        df = build_material_backfill(self._cuts_row(), self._links())
        for col in _MATERIAL_BACKFILL_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_verified_material_populated(self):
        df = build_material_backfill(self._cuts_row(), self._links())
        assert df.iloc[0]["verified_material"] == "4140 ALLOY STEEL"
        assert df.iloc[0]["material_source"] == "ROUTER"
        assert df.iloc[0]["material_confidence"] == "HIGH"

    def test_sv_fv_renamed_to_S_F(self):
        df = build_material_backfill(self._cuts_row(), self._links())
        assert "S" in df.columns
        assert "F" in df.columns
        assert df.iloc[0]["S"] == 500.0
        assert df.iloc[0]["F"] == 0.012

    def test_tool_description_renamed(self):
        df = build_material_backfill(self._cuts_row(), self._links())
        assert "resolved_tool_description" in df.columns
        assert df.iloc[0]["resolved_tool_description"] == "ROUGHING INSERT"

    def test_unmatched_cut_gets_unknown_material(self):
        cuts = _cuts([{"source_file": "P:/417/NOLINK.NC", "machine_folder": "417"}])
        links = self._links()  # links only has EM10986
        df = build_material_backfill(cuts, links)
        row = df[df["source_file"] == "P:/417/NOLINK.NC"].iloc[0]
        assert row["verified_material"] == "UNKNOWN"
        assert row["link_method"] == "no_match"

    def test_empty_cuts_returns_empty_with_columns(self):
        df = build_material_backfill(pd.DataFrame(), self._links())
        assert isinstance(df, pd.DataFrame)
        for col in _MATERIAL_BACKFILL_COLS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Router program context builder
# ---------------------------------------------------------------------------

class TestBuildRouterProgramContext:
    def _links(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "source_file":           "P:/655/EM0825.NC",
            "filename":              "EM0825.NC",
            "machine_folder":        "655",
            "program_id":            1,
            "matched_job_number":    "9001",
            "matched_part_number":   "PART-825",
            "matched_drawing_number":"EM0825",
            "matched_revision":      "A",
            "matched_shared_print_file": "",
            "matched_router_file":   "G:/j.txt",
            "link_confidence":       "HIGH",
            "link_method":           "exact_drawing_number",
            "link_reason":           "matched EM0825",
            "material":              "316",
            "material_source":       "ROUTER",
            "material_confidence":   "HIGH",
            "material_conflict":     False,
            "needs_review":          False,
        }], columns=_PROG_JOB_LINK_COLS)

    def _router(self) -> pd.DataFrame:
        return _router_ops([
            {"source_file": "G:/j.txt", "job_number": "9001",
             "operation_number": "10", "work_center": "655 HAAS",
             "machine": "HAAS", "operation_description": "MILL PROFILE"},
            {"source_file": "G:/j.txt", "job_number": "9001",
             "operation_number": "20", "work_center": "INSPECT",
             "machine": "INSPECT", "operation_description": "CMM CHECK"},
        ])

    def test_returns_one_row_per_program_op_pair(self):
        df = build_router_program_context(self._links(), self._router())
        assert len(df) == 2

    def test_has_all_required_columns(self):
        df = build_router_program_context(self._links(), self._router())
        for col in _ROUTER_CONTEXT_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_program_reference_populated(self):
        df = build_router_program_context(self._links(), self._router())
        assert (df["program_reference"] == "EM0825.NC").all()

    def test_machine_hint_populated(self):
        df = build_router_program_context(self._links(), self._router())
        hints = df["machine_hint"].tolist()
        assert "HAAS" in hints

    def test_no_job_links_returns_empty(self):
        links = pd.DataFrame([{
            **{c: "" for c in _PROG_JOB_LINK_COLS},
            "matched_job_number": "",
            "link_method": "no_match",
        }], columns=_PROG_JOB_LINK_COLS)
        df = build_router_program_context(links, self._router())
        assert len(df) == 0

    def test_empty_router_returns_empty(self):
        df = build_router_program_context(self._links(), pd.DataFrame())
        assert isinstance(df, pd.DataFrame)
        for col in _ROUTER_CONTEXT_COLS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Export column completeness
# ---------------------------------------------------------------------------

class TestExportColumnCompleteness:
    def test_program_job_links_csv_has_all_columns(self, tmp_path):
        manifest = _manifest([{"program_id": 1, "filename": "EM0001.NC"}])
        jobs     = _job_meta([{"drawing_number": "EM0001", "material": "4140"}])
        df = build_program_job_links(manifest, jobs, _prints([]), _router_ops([]))
        path = export_program_job_links(df, tmp_path, "20260101_000000")
        result = pd.read_csv(path)
        for col in _PROG_JOB_LINK_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_material_backfill_csv_has_all_columns(self, tmp_path):
        links = pd.DataFrame([{**{c: "" for c in _PROG_JOB_LINK_COLS}}], columns=_PROG_JOB_LINK_COLS)
        cuts  = _cuts([{"source_file": "P:/x.NC"}])
        df    = build_material_backfill(cuts, links)
        path  = export_material_backfill(df, tmp_path, "20260101_000000")
        result = pd.read_csv(path)
        for col in _MATERIAL_BACKFILL_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_router_context_csv_has_all_columns(self, tmp_path):
        df   = pd.DataFrame(columns=_ROUTER_CONTEXT_COLS)
        path = export_router_program_context(df, tmp_path, "20260101_000000")
        result = pd.read_csv(path)
        for col in _ROUTER_CONTEXT_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_empty_links_csv_has_all_columns(self, tmp_path):
        path = export_program_job_links(
            pd.DataFrame(columns=_PROG_JOB_LINK_COLS), tmp_path, "20260101_000000"
        )
        result = pd.read_csv(path)
        for col in _PROG_JOB_LINK_COLS:
            assert col in result.columns


# ---------------------------------------------------------------------------
# No-overwrite behaviour
# ---------------------------------------------------------------------------

class TestNoOverwrite:
    def test_different_timestamps_produce_different_files(self, tmp_path):
        df = pd.DataFrame(columns=_PROG_JOB_LINK_COLS)
        p1 = export_program_job_links(df, tmp_path, "20260101_000001")
        p2 = export_program_job_links(df, tmp_path, "20260101_000002")
        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    def test_backfill_no_overwrite(self, tmp_path):
        df = pd.DataFrame(columns=_MATERIAL_BACKFILL_COLS)
        p1 = export_material_backfill(df, tmp_path, "20260101_000001")
        p2 = export_material_backfill(df, tmp_path, "20260101_000002")
        assert p1 != p2

    def test_router_context_no_overwrite(self, tmp_path):
        df = pd.DataFrame(columns=_ROUTER_CONTEXT_COLS)
        p1 = export_router_program_context(df, tmp_path, "20260101_000001")
        p2 = export_router_program_context(df, tmp_path, "20260101_000002")
        assert p1 != p2


# ---------------------------------------------------------------------------
# Latest file auto-detection
# ---------------------------------------------------------------------------

class TestLatestFileAutoDetection:
    def test_detects_latest_manifest(self, tmp_path):
        (tmp_path / "manifest_20260101_000001.csv").write_text("a,b\n1,2")
        (tmp_path / "manifest_20260101_000002.csv").write_text("a,b\n1,2")
        detected = detect_latest_exports(tmp_path)
        assert detected["manifest"] is not None
        assert "000002" in detected["manifest"].name

    def test_returns_none_when_no_file(self, tmp_path):
        detected = detect_latest_exports(tmp_path)
        assert detected["manifest"] is None
        assert detected["cuts"] is None
        assert detected["job_metadata"] is None

    def test_detects_all_six_types(self, tmp_path):
        for stem in [
            "manifest_20260101_000001",
            "cuts_20260101_000001",
            "tool_summary_20260101_000001",
            "job_metadata_20260101_000001",
            "shared_print_index_20260101_000001",
            "router_operations_20260101_000001",
        ]:
            (tmp_path / f"{stem}.csv").write_text("col\nval")
        detected = detect_latest_exports(tmp_path)
        for key in ("manifest", "cuts", "tool_summary", "job_metadata", "shared_print", "router_ops"):
            assert detected[key] is not None, f"Key not detected: {key}"

    def test_picks_newer_of_two_same_pattern(self, tmp_path):
        import time
        p1 = tmp_path / "cuts_20260101_000001.csv"
        p2 = tmp_path / "cuts_20260101_000002.csv"
        p1.write_text("a,b\n1,2")
        time.sleep(0.05)
        p2.write_text("a,b\n3,4")
        detected = detect_latest_exports(tmp_path)
        assert detected["cuts"].name == p2.name


# ---------------------------------------------------------------------------
# run_job_link pipeline integration
# ---------------------------------------------------------------------------

class TestRunJobLinkIntegration:
    def _write_csv(self, path: Path, df: pd.DataFrame) -> Path:
        df.to_csv(path, index=False)
        return path

    def test_run_produces_three_files(self, tmp_path):
        exports = tmp_path / "exports"
        exports.mkdir()
        m = self._write_csv(exports / "manifest_20260101_000001.csv",
                            _manifest([{"program_id": 1, "filename": "EM0001.NC",
                                        "source_file": "P:/655/EM0001.NC",
                                        "machine_folder": "655"}]))
        j = self._write_csv(exports / "job_metadata_20260101_000001.csv",
                            _job_meta([{"drawing_number": "EM0001", "material": "4140"}]))
        c = self._write_csv(exports / "cuts_20260101_000001.csv",
                            _cuts([{"source_file": "P:/655/EM0001.NC",
                                    "machine_folder": "655"}]))
        lp, bp, cp = run_job_link(exports_dir=exports)
        assert lp.exists()
        assert bp.exists()
        assert cp.exists()

    def test_run_with_no_inputs_creates_empty_exports(self, tmp_path):
        exports = tmp_path / "empty_exports"
        exports.mkdir()
        lp, bp, cp = run_job_link(exports_dir=exports)
        assert lp.exists()
        assert bp.exists()
        assert cp.exists()

    def test_link_accuracy_end_to_end(self, tmp_path):
        exports = tmp_path / "exports"
        exports.mkdir()
        self._write_csv(exports / "manifest_20260101_000001.csv",
                        _manifest([{"program_id": 1, "filename": "EM0825.NC",
                                    "source_file": "P:/655/EM0825.NC",
                                    "machine_folder": "655"}]))
        self._write_csv(exports / "job_metadata_20260101_000001.csv",
                        _job_meta([{"job_number": "9001", "drawing_number": "EM0825",
                                    "material": "316 STAINLESS"}]))
        self._write_csv(exports / "cuts_20260101_000001.csv",
                        _cuts([{"source_file": "P:/655/EM0825.NC"}]))
        self._write_csv(exports / "shared_print_index_20260101_000001.csv",
                        _prints([]))
        self._write_csv(exports / "router_operations_20260101_000001.csv",
                        _router_ops([]))

        lp, bp, _ = run_job_link(exports_dir=exports)
        links = pd.read_csv(lp)
        backfill = pd.read_csv(bp)

        assert links.iloc[0]["link_method"] == "exact_drawing_number"
        assert links.iloc[0]["material"] == "316 STAINLESS"
        assert backfill.iloc[0]["verified_material"] == "316 STAINLESS"


# ---------------------------------------------------------------------------
# Phase 6D: 4-digit numeric router token matching
# ---------------------------------------------------------------------------

class TestRouterMatch4DigitTokens:
    """
    Phase 6D: Validate that 4-digit pure-numeric tokens (e.g. 0582, 1025)
    are now allowed through Strategy 5 while preserving all guards for
    shorter tokens, non-numeric 4-char tokens, and ambiguous matches.
    """

    def _job(self, job_number: str, material: str = "4140") -> pd.DataFrame:
        return _job_meta([{"source_file": f"G:/j_{job_number}.txt",
                           "job_number": job_number, "material": material}])

    def _ops(self, rows: list[dict]) -> pd.DataFrame:
        return _router_ops(rows)

    def test_4digit_numeric_single_job_links_via_router(self):
        """0582.OP1 → token '0582' in router for exactly one job → router_match."""
        manifest = _manifest([{"program_id": 1, "filename": "0582.OP1",
                                "source_file": "P:/417/0582.OP1",
                                "machine_folder": "417,426"}])
        jobs = self._job("D21635", "1045 HR")
        ops  = self._ops([{"job_number": "D21635",
                            "operation_description": "right end program number: 0582.op2"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] == "router_match"
        assert df.iloc[0]["link_confidence"] == "LOW"
        assert df.iloc[0]["matched_job_number"] == "D21635"
        assert df.iloc[0]["needs_review"] == True

    def test_4digit_numeric_material_populated_from_router_job(self):
        """Material from the matched job is carried through."""
        manifest = _manifest([{"program_id": 1, "filename": "1486.OP1",
                                "source_file": "P:/417/1486.OP1",
                                "machine_folder": "417,426"}])
        jobs = self._job("D23022", "316 STAINLESS")
        ops  = self._ops([{"job_number": "D23022",
                            "operation_description": "first operation left end program number: 1486.op1"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] == "router_match"
        assert df.iloc[0]["material"] == "316 STAINLESS"

    def test_4digit_numeric_multiple_jobs_conflicting_materials(self):
        """Token '1025' in router for two jobs with different materials.
        Phase 6E Strategy 6 now catches this as composite_material_consensus
        with conflicting_materials — no material assigned, needs_review=True."""
        manifest = _manifest([{"program_id": 1, "filename": "1025.OP1",
                                "source_file": "P:/417/1025.OP1",
                                "machine_folder": "417,426"}])
        jobs = _job_meta([
            {"job_number": "D20144", "material": "4140"},
            {"job_number": "D20993", "material": "316"},
        ])
        ops = self._ops([
            {"job_number": "D20144", "operation_description": "second op program 1025.op2"},
            {"job_number": "D20993", "operation_description": "program number: 1025.op1"},
        ])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["link_method"] == "composite_material_consensus"
        assert row["material"] == "UNKNOWN"
        assert row["material_consensus_status"] == "conflicting_materials"
        assert row["needs_review"] == True

    def test_3digit_token_still_not_matched_by_router(self):
        """3-char token '118' must not trigger router_match even with a single-job hit."""
        manifest = _manifest([{"program_id": 1, "filename": "118.NC",
                                "source_file": "P:/417/118.NC",
                                "machine_folder": "417,426"}])
        jobs = self._job("D24076")
        ops  = self._ops([{"job_number": "D24076",
                            "operation_description": "program 118 left end"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] != "router_match"

    def test_4char_nonnumeric_token_not_matched_by_router(self):
        """4-char non-numeric token 'ABCD' must not pass the numeric guard."""
        manifest = _manifest([{"program_id": 1, "filename": "ABCD.NC"}])
        jobs = self._job("D99999")
        ops  = self._ops([{"job_number": "D99999",
                            "operation_description": "ABCD sequence operation"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] != "router_match"

    def test_5digit_existing_behaviour_unchanged(self):
        """5-digit tokens continue to work exactly as before."""
        manifest = _manifest([{"program_id": 1, "filename": "10007.NC"}])
        jobs = self._job("D20182", "4140 HR HT")
        ops  = self._ops([{"job_number": "D20182",
                            "operation_description": "left end program number: 10007"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] == "router_match"
        assert df.iloc[0]["matched_job_number"] == "D20182"

    def test_leading_zero_4digit_token_links(self):
        """Leading-zero 4-digit stem like 0582 produces token '0582' which is all digits."""
        manifest = _manifest([{"program_id": 1, "filename": "0118.OP1",
                                "source_file": "P:/417/0118.OP1",
                                "machine_folder": "417,426"}])
        jobs = self._job("D24076")
        ops  = self._ops([{"job_number": "D24076",
                            "operation_description": "program number 0118.op1"}])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        assert df.iloc[0]["link_method"] == "router_match"
        assert df.iloc[0]["matched_job_number"] == "D24076"


# ---------------------------------------------------------------------------
# Phase 6E: Composite material consensus strategy
# ---------------------------------------------------------------------------

class TestCompositeMaterialConsensus:
    """
    Phase 6E: Validates Strategy 6 — composite_material_consensus.
    Multiple router jobs reference the same program; material is verified
    by consensus without assigning a specific job number.
    """

    def _multi_job_ops(self, stem: str, jobs: list[str]) -> pd.DataFrame:
        return _router_ops([
            {"job_number": jn,
             "operation_description": f"left end program number: {stem}"}
            for jn in jobs
        ])

    def test_multiple_jobs_same_material_assigns_consensus(self):
        """3 jobs all specify '4140 HR HT' → consensus_material, material assigned."""
        manifest = _manifest([{"program_id": 1, "filename": "10007001.EIA",
                                "source_file": "P:/421/10007001.EIA",
                                "machine_folder": "421, 423, 424"}])
        jobs = _job_meta([
            {"job_number": "D20182", "material": "4140 HR HT"},
            {"job_number": "D20464", "material": "4140 HR HT"},
            {"job_number": "D21690", "material": "4140 HR HT"},
        ])
        ops = self._multi_job_ops("10007001", ["D20182", "D20464", "D21690"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["link_method"] == "composite_material_consensus"
        assert row["link_confidence"] == "MEDIUM"
        assert row["material"] == "4140 HR HT"
        assert row["material_source"] == "ROUTER_CONSENSUS"
        assert row["material_confidence"] == "MEDIUM"
        assert row["material_consensus_status"] == "consensus_material"
        assert row["needs_review"] == False
        assert row["material_conflict"] == False

    def test_multiple_jobs_conflicting_materials_no_assignment(self):
        """Jobs disagree on material → conflicting_materials, material stays UNKNOWN."""
        manifest = _manifest([{"program_id": 1, "filename": "10033001.EIA",
                                "source_file": "P:/421/10033001.EIA",
                                "machine_folder": "421, 423, 424"}])
        jobs = _job_meta([
            {"job_number": "D20372", "material": "4140 HR HT"},
            {"job_number": "D20509", "material": "316 STAINLESS"},
        ])
        ops = self._multi_job_ops("10033001", ["D20372", "D20509"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["link_method"] == "composite_material_consensus"
        assert row["material"] == "UNKNOWN"
        assert row["material_source"] == "UNKNOWN"
        assert row["material_conflict"] == True
        assert row["needs_review"] == True
        assert row["material_consensus_status"] == "conflicting_materials"

    def test_multiple_jobs_all_unknown_material_stays_unknown(self):
        """All candidate jobs have no material → insufficient_material_data."""
        manifest = _manifest([{"program_id": 1, "filename": "10064001.EIA",
                                "source_file": "P:/421/10064001.EIA",
                                "machine_folder": "421, 423, 424"}])
        jobs = _job_meta([
            {"job_number": "D20067", "material": ""},
            {"job_number": "D20629", "material": ""},
            {"job_number": "D20899", "material": ""},
        ])
        ops = self._multi_job_ops("10064001", ["D20067", "D20629", "D20899"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["link_method"] == "composite_material_consensus"
        assert row["material"] == "UNKNOWN"
        assert row["material_confidence"] == "NONE"
        assert row["material_consensus_status"] == "insufficient_material_data"
        assert row["needs_review"] == False

    def test_one_known_material_plus_unknowns_stays_unknown(self):
        """Only 1 of 3 jobs has a known material → rule requires >= 2; stays UNKNOWN."""
        manifest = _manifest([{"program_id": 1, "filename": "10071001.EIA",
                                "source_file": "P:/421/10071001.EIA",
                                "machine_folder": "421, 423, 424"}])
        jobs = _job_meta([
            {"job_number": "D21126", "material": "4140 HR HT"},
            {"job_number": "D24415", "material": ""},
            {"job_number": "D24500", "material": ""},
        ])
        ops = self._multi_job_ops("10071001", ["D21126", "D24415", "D24500"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["material_consensus_status"] == "insufficient_material_data"
        assert row["material"] == "UNKNOWN"

    def test_two_known_same_material_plus_unknowns_assigns_consensus(self):
        """2 of 4 jobs agree on material, 2 are unknown → >= 2 threshold met → consensus."""
        manifest = _manifest([{"program_id": 1, "filename": "10007002.EIA",
                                "source_file": "P:/421/10007002.EIA",
                                "machine_folder": "421, 423, 424"}])
        jobs = _job_meta([
            {"job_number": "D20182", "material": "4140 HR HT"},
            {"job_number": "D20464", "material": "4140 HR HT"},
            {"job_number": "D21690", "material": ""},
            {"job_number": "D23073", "material": ""},
        ])
        ops = self._multi_job_ops("10007002", ["D20182", "D20464", "D21690", "D23073"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["material_consensus_status"] == "consensus_material"
        assert row["material"] == "4140 HR HT"
        assert row["material_confidence"] == "MEDIUM"

    def test_matched_job_number_is_not_single_job(self):
        """matched_job_number must never hold a single specific job for a consensus match."""
        manifest = _manifest([{"program_id": 1, "filename": "10033002.EIA",
                                "source_file": "P:/421/10033002.EIA",
                                "machine_folder": "421, 423, 424"}])
        jobs = _job_meta([
            {"job_number": "D20372", "material": "316 STAINLESS"},
            {"job_number": "D20509", "material": "316 STAINLESS"},
        ])
        ops = self._multi_job_ops("10033002", ["D20372", "D20509"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["link_method"] == "composite_material_consensus"
        # Job number must not be a single specific D-number
        assert row["matched_job_number"] not in ("D20372", "D20509")
        assert row["matched_job_number"] == "MULTIPLE"

    def test_material_source_is_router_consensus(self):
        """material_source must be ROUTER_CONSENSUS for a successful consensus."""
        manifest = _manifest([{"program_id": 1, "filename": "10264001.EIA"}])
        jobs = _job_meta([
            {"job_number": "D21001", "material": "17-4 PH"},
            {"job_number": "D21500", "material": "17-4 PH"},
        ])
        ops = self._multi_job_ops("10264001", ["D21001", "D21500"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["material_source"] == "ROUTER_CONSENSUS"

    def test_material_confidence_is_medium_for_consensus(self):
        """Consensus material must be MEDIUM confidence, not HIGH."""
        manifest = _manifest([{"program_id": 1, "filename": "10303001.EIA"}])
        jobs = _job_meta([
            {"job_number": "D22000", "material": "1144 CF"},
            {"job_number": "D22500", "material": "1144 CF"},
        ])
        ops = self._multi_job_ops("10303001", ["D22000", "D22500"])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        row = df.iloc[0]
        assert row["material_confidence"] == "MEDIUM"
        assert row["link_confidence"] == "MEDIUM"

    def test_new_columns_present_in_all_link_records(self):
        """candidate_job_count, candidate_materials, material_consensus_status
        must appear for every program, including non-consensus ones."""
        manifest = _manifest([
            {"program_id": 1, "filename": "EM0001.NC"},
            {"program_id": 2, "filename": "10007001.EIA"},
        ])
        jobs = _job_meta([
            {"drawing_number": "EM0001", "material": "4140"},
            {"job_number": "D20182", "material": "316"},
            {"job_number": "D20464", "material": "316"},
        ])
        ops = _router_ops([
            {"job_number": "D20182",
             "operation_description": "program number: 10007001"},
            {"job_number": "D20464",
             "operation_description": "program number: 10007001"},
        ])
        df = build_program_job_links(manifest, jobs, _prints([]), ops)
        for col in ("candidate_job_count", "candidate_materials", "material_consensus_status"):
            assert col in df.columns, f"Missing column: {col}"
        em_row   = df[df["filename"] == "EM0001.NC"].iloc[0]
        comp_row = df[df["filename"] == "10007001.EIA"].iloc[0]
        assert em_row["material_consensus_status"]   == "not_applicable"
        assert comp_row["material_consensus_status"] == "consensus_material"
        assert comp_row["candidate_job_count"]       == 2

    def test_new_columns_present_in_material_backfill(self):
        """The three new columns must propagate through material_backfill."""
        manifest = _manifest([{"program_id": 1, "filename": "10007001.EIA",
                                "source_file": "P:/421/10007001.EIA",
                                "machine_folder": "421, 423, 424"}])
        jobs = _job_meta([
            {"job_number": "D20182", "material": "4140 HR HT"},
            {"job_number": "D20464", "material": "4140 HR HT"},
        ])
        ops  = self._multi_job_ops("10007001", ["D20182", "D20464"])
        cuts = _cuts([{"source_file": "P:/421/10007001.EIA",
                       "machine_folder": "421, 423, 424"}])
        links_df = build_program_job_links(manifest, jobs, _prints([]), ops)
        bf_df    = build_material_backfill(cuts, links_df)
        for col in ("candidate_job_count", "candidate_materials", "material_consensus_status"):
            assert col in bf_df.columns, f"Missing backfill column: {col}"
        row = bf_df.iloc[0]
        assert row["material_consensus_status"] == "consensus_material"
        assert row["candidate_job_count"] == 2
