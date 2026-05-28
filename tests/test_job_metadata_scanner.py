"""
test_job_metadata_scanner.py — Tests for Phase 6A job metadata scanner.
"""

from pathlib import Path

import pandas as pd
import pytest

import os
import time

from src.job_metadata_scanner import (
    JOB_FOLDER_EXTENSIONS,
    PARSE_ACTIONS,
    SHARED_PRINT_EXTENSIONS,
    _DEFAULT_LOOKBACK_DAYS,
    _JOB_META_COLS,
    _PRINT_INDEX_COLS,
    _ROUTER_OPS_COLS,
    build_job_metadata_record,
    build_shared_print_record,
    classify_content,
    classify_filename,
    compute_scan_metrics,
    decide_parse_action,
    export_job_metadata,
    export_router_operations,
    export_shared_print_index,
    extract_drawing_number,
    extract_file_text,
    extract_job_number,
    extract_material,
    extract_material_details,
    extract_part_number,
    extract_revision,
    extract_routing_operations,
    extract_work_centers,
    normalize_material,
    sample_pdf_first_page,
    scan_job_folders,
    scan_shared_prints,
    score_confidence,
)

# ---------------------------------------------------------------------------
# Synthetic document text fixtures
# ---------------------------------------------------------------------------

TRAVELER_TEXT = """\
ACME MACHINE SHOP
TRAVELER / WORK ORDER

JOB NO: 24567
PART NO: 4140-SHAFT-001
DWG NO: D-4140-001
REV: B
MATERIAL: 4140 ALLOY STEEL

ROUTING:
10  CNC LATHE       TURN OD TO 2.500 DIA
20  VMC              MILL KEYWAY .250 X .125
30  GRIND            GRIND OD TO FINAL DIM
40  INSPECT          VERIFY ALL DIMS TO PRINT

NOTES: Run 5 pieces.
"""

ROUTER_TEXT = """\
WORK ORDER: 98765
P/N: SS316-FLANGE-002
DRAWING NO: F-316-002
REV: C
MAT: 316 STAINLESS STEEL

OPERATION ROUTING

10  432 MAZAK       FACE AND TURN PROFILE
20  655 HAAS         MILL BOLT CIRCLE
30  655 HAAS         MILL SLOTS
40  INSPECT          CMM INSPECTION

"""

PRINT_TEXT = """\
PART NUMBER: ABC-1234-X
DWG NO: D-1234-REV-A
REVISION: A
MATERIAL: 17-4 PH STAINLESS

TOLERANCES: .XXX = ±.005  .XX = ±.010
"""

SPARSE_TEXT = """\
This document has very little structured data.
Some notes about a job.
"""

OP_LABEL_TEXT = """\
JOB NO: 33100
PART NO: PART-001

OP 10 - TURN OD ON LATHE
OP 20 - MILL FEATURES ON VMC
OP 30 - INSPECT TO PRINT
"""

DRMS_TRAVELER_TEXT = """\
EMPOWER MANUFACTURING
Job: D26304
Part: L1D30855093 Customer PO: 4520343717 Quote: Q-260398
"A" Shaft, 4140 HR HT Line: 30
Rev: AA Drawing: L1D30855093A

Routing Comments
WC/Vendor Sch Start
Oper/Serv Operation Key Sch End Description Setup Run Rate Run
4|17 L6ATH2E 254 6221541z ~ 1.00 0.60 Parts/Hr 1.67
3 Center Both Ends.
Material: 6-1/2" Diameter 4140 HR HT
417 LATHE Length: 55.400 +.062 -.000 Long
4|13 L6ATH2E 254 6223543 ~ 0.50 1.50 Parts/Hr 0.67
5 Face Pulley End Of Shaft to Remove Drive Blade Marks Both Ends.
413 LATHE Drill, Tap, Counter Bore, Re-Center.
6|55 M6ILL2254 6225545+ ~ 1.00 2.00 Parts/Hr 0.50
6 Program Number:
655 MILL Keyways 5/16 and Above In Width Shall Be .030".
5|19 G6RIN2D 254 62265461 ~ 1.00 0.35 Parts/Hr 2.86
7 === Recalibrate Every 10 Minutes and Between Each Diameter ====
519 GRIND Grind: Live Centers Required

Materials
Material
75016 EMJ 4140 HR HT 6.500 DIA 522.00
4140 HR HT 6.500 DIA
75017 EMJ CUTTING CHARGE 1.00
"""

# ---------------------------------------------------------------------------
# extract_job_number
# ---------------------------------------------------------------------------

class TestExtractJobNumber:
    def test_job_no_colon(self):
        assert extract_job_number("JOB NO: 24567") == "24567"

    def test_job_number_label(self):
        assert extract_job_number("JOB NUMBER: 12345") == "12345"

    def test_work_order(self):
        assert extract_job_number("WORK ORDER: 98765") == "98765"

    def test_wo_abbreviation(self):
        assert extract_job_number("W.O. #44321") == "44321"

    def test_job_hash(self):
        assert extract_job_number("JOB #55001") == "55001"

    def test_case_insensitive(self):
        assert extract_job_number("job no: 77777") == "77777"

    def test_no_match_returns_empty(self):
        assert extract_job_number("Nothing relevant here.") == ""

    def test_too_short_number_not_matched(self):
        # 3-digit number should not match (min 4 digits)
        assert extract_job_number("JOB NO: 123") == ""

    def test_from_traveler(self):
        assert extract_job_number(TRAVELER_TEXT) == "24567"

    def test_alpha_prefixed_job_from_drms_traveler(self):
        assert extract_job_number(DRMS_TRAVELER_TEXT) == "D26304"


# ---------------------------------------------------------------------------
# extract_part_number
# ---------------------------------------------------------------------------

class TestExtractPartNumber:
    def test_part_no_label(self):
        assert extract_part_number("PART NO: 4140-SHAFT-001") == "4140-SHAFT-001"

    def test_pn_abbreviation(self):
        assert extract_part_number("P/N: SS316-FLANGE-002") == "SS316-FLANGE-002"

    def test_part_number_label(self):
        assert extract_part_number("PART NUMBER: ABC-1234-X") == "ABC-1234-X"

    def test_part_hash(self):
        assert extract_part_number("PART #: WIDGET-99") == "WIDGET-99"

    def test_case_insensitive(self):
        assert extract_part_number("part no: xyz-001") == "xyz-001"

    def test_no_match_returns_empty(self):
        assert extract_part_number("Nothing here.") == ""

    def test_from_traveler(self):
        assert extract_part_number(TRAVELER_TEXT) == "4140-SHAFT-001"

    def test_from_router(self):
        assert extract_part_number(ROUTER_TEXT) == "SS316-FLANGE-002"

    def test_plain_part_label_from_drms_traveler(self):
        assert extract_part_number(DRMS_TRAVELER_TEXT) == "L1D30855093"


# ---------------------------------------------------------------------------
# extract_drawing_number
# ---------------------------------------------------------------------------

class TestExtractDrawingNumber:
    def test_dwg_no(self):
        assert extract_drawing_number("DWG NO: D-4140-001") == "D-4140-001"

    def test_drawing_no(self):
        assert extract_drawing_number("DRAWING NO: F-316-002") == "F-316-002"

    def test_print_no(self):
        assert extract_drawing_number("PRINT NO: P-001") == "P-001"

    def test_case_insensitive(self):
        assert extract_drawing_number("dwg no: D-1234-REV-A") == "D-1234-REV-A"

    def test_no_match_returns_empty(self):
        assert extract_drawing_number("Nothing here.") == ""

    def test_from_traveler(self):
        assert extract_drawing_number(TRAVELER_TEXT) == "D-4140-001"

    def test_plain_drawing_label_from_drms_traveler(self):
        assert extract_drawing_number(DRMS_TRAVELER_TEXT) == "L1D30855093A"


# ---------------------------------------------------------------------------
# extract_revision
# ---------------------------------------------------------------------------

class TestExtractRevision:
    def test_rev_colon(self):
        assert extract_revision("REV: B") == "B"

    def test_revision_label(self):
        assert extract_revision("REVISION: A") == "A"

    def test_case_insensitive(self):
        assert extract_revision("rev: c") == "c"

    def test_no_match_returns_empty(self):
        assert extract_revision("No revision info.") == ""

    def test_from_traveler(self):
        assert extract_revision(TRAVELER_TEXT) == "B"

    def test_from_print(self):
        assert extract_revision(PRINT_TEXT) == "A"


# ---------------------------------------------------------------------------
# extract_material
# ---------------------------------------------------------------------------

class TestExtractMaterial:
    def test_material_label(self):
        assert "4140" in extract_material("MATERIAL: 4140 ALLOY STEEL")

    def test_mat_label(self):
        assert "316" in extract_material("MAT: 316 STAINLESS STEEL")

    def test_trailing_punctuation_stripped(self):
        result = extract_material("MATERIAL: 4140 ALLOY STEEL,")
        assert not result.endswith(",")

    def test_no_match_returns_empty(self):
        assert extract_material("Nothing here.") == ""

    def test_from_traveler(self):
        assert "4140" in extract_material(TRAVELER_TEXT)

    def test_from_router(self):
        assert "316" in extract_material(ROUTER_TEXT)

    def test_normalizes_explicit_operation_material(self):
        assert extract_material(DRMS_TRAVELER_TEXT) == "4140 HR HT"

    def test_normalizes_materials_section_dia_line(self):
        text = "Materials\n75016 EMJ 4140 HR HT 6.500 DIA 522.00\n"
        raw, normalized = extract_material_details(text)
        assert raw == "75016 EMJ 4140 HR HT 6.500 DIA 522.00"
        assert normalized == "4140 HR HT"

    def test_normalizes_header_title_hint(self):
        assert extract_material('"A" Shaft, 4140 HR HT') == "4140 HR HT"

    def test_excludes_cutting_charge(self):
        assert normalize_material("75017 EMJ CUTTING CHARGE 1.00") == ""


# ---------------------------------------------------------------------------
# extract_work_centers
# ---------------------------------------------------------------------------

class TestExtractWorkCenters:
    def test_finds_lathe(self):
        assert "LATHE" in extract_work_centers("CNC LATHE TURN OD").upper()

    def test_finds_vmc(self):
        assert "VMC" in extract_work_centers("VMC MILL FEATURES").upper()

    def test_finds_multiple(self):
        result = extract_work_centers(TRAVELER_TEXT).upper()
        assert "LATHE" in result
        assert "VMC" in result
        assert "GRIND" in result or "GRINDING" in result
        assert "INSPECT" in result or "INSPECTION" in result

    def test_deduplicates(self):
        text = "LATHE\nLATHE\nLATHE"
        result = extract_work_centers(text)
        assert result.upper().count("LATHE") == 1

    def test_empty_text(self):
        assert extract_work_centers("") == ""

    def test_no_keywords(self):
        assert extract_work_centers("Nothing relevant here.") == ""


# ---------------------------------------------------------------------------
# score_confidence
# ---------------------------------------------------------------------------

class TestScoreConfidence:
    def test_high_four_fields(self):
        assert score_confidence("24567", "PART-001", "DWG-001", "A", "") == "HIGH"

    def test_high_five_fields(self):
        assert score_confidence("24567", "PART-001", "DWG-001", "A", "4140") == "HIGH"

    def test_medium_two_fields(self):
        assert score_confidence("24567", "PART-001", "", "", "") == "MEDIUM"

    def test_medium_three_fields(self):
        assert score_confidence("24567", "PART-001", "DWG-001", "", "") == "MEDIUM"

    def test_low_one_field(self):
        assert score_confidence("24567", "", "", "", "") == "LOW"

    def test_low_no_fields(self):
        assert score_confidence("", "", "", "", "") == "LOW"


# ---------------------------------------------------------------------------
# extract_routing_operations
# ---------------------------------------------------------------------------

class TestExtractRoutingOperations:
    def test_tabular_extracts_ops(self):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        assert len(ops) == 4

    def test_ops_have_required_fields(self):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        required = {
            "source_file", "job_number", "operation_sequence", "operation_number",
            "work_center", "machine", "operation_description", "operation_notes",
        }
        for op in ops:
            assert required.issubset(op.keys())

    def test_op_sequence_is_sequential(self):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        seqs = [op["operation_sequence"] for op in ops]
        assert seqs == list(range(1, len(seqs) + 1))

    def test_op_numbers_correct(self):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        assert ops[0]["operation_number"] == "10"
        assert ops[1]["operation_number"] == "20"
        assert ops[2]["operation_number"] == "30"
        assert ops[3]["operation_number"] == "40"

    def test_work_center_extracted(self):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        wcs = [op["work_center"].upper() for op in ops]
        assert any("LATHE" in w or "CNC" in w for w in wcs)

    def test_machine_keyword_extracted(self):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        machines = [op["machine"].upper() for op in ops]
        assert any("LATHE" in m for m in machines)

    def test_labeled_format(self):
        ops = extract_routing_operations(OP_LABEL_TEXT, "test.txt", "33100")
        assert len(ops) == 3
        assert ops[0]["operation_number"] == "10"
        assert ops[1]["operation_number"] == "20"

    def test_empty_text_returns_empty(self):
        assert extract_routing_operations("", "test.txt", "") == []

    def test_non_multiple_of_five_not_matched(self):
        # Line starting with 11 (not multiple of 5) should not be an operation
        text = "11  Some work   description here"
        ops = extract_routing_operations(text, "test.txt", "")
        assert ops == []

    def test_source_file_and_job_number_propagated(self):
        ops = extract_routing_operations(TRAVELER_TEXT, "some/path/file.txt", "24567")
        for op in ops:
            assert op["source_file"] == "some/path/file.txt"
            assert op["job_number"] == "24567"

    def test_drms_work_center_codes_and_types(self):
        ops = extract_routing_operations(DRMS_TRAVELER_TEXT, "traveler.pdf", "D26304")
        pairs = {(op["work_center_code"], op["work_center_type"]) for op in ops}
        assert ("417", "LATHE") in pairs
        assert ("413", "LATHE") in pairs
        assert ("655", "MILL") in pairs
        assert ("519", "GRIND") in pairs

    def test_operation_notes_capture_detail_lines(self):
        ops = extract_routing_operations(DRMS_TRAVELER_TEXT, "traveler.pdf", "D26304")
        op417 = next(op for op in ops if op["work_center"] == "417 LATHE")
        assert "Length: 55.400" in op417["operation_notes"]


# ---------------------------------------------------------------------------
# extract_file_text (text-file path only; PDF tested via build_ tests)
# ---------------------------------------------------------------------------

class TestExtractFileText:
    def test_reads_txt_file(self, tmp_path):
        f = tmp_path / "router.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        text, notes = extract_file_text(f)
        assert "24567" in text
        assert notes == ""

    def test_missing_file_returns_read_error(self, tmp_path):
        f = tmp_path / "missing.txt"
        text, notes = extract_file_text(f)
        assert text == ""
        assert notes == "read_error"

    def test_latin1_file_readable(self, tmp_path):
        f = tmp_path / "latin.txt"
        f.write_bytes(b"JOB NO: 55555\nMAT: STEEL\n")
        text, notes = extract_file_text(f)
        assert "55555" in text


# ---------------------------------------------------------------------------
# build_job_metadata_record
# ---------------------------------------------------------------------------

class TestBuildJobMetadataRecord:
    def test_returns_tuple(self, tmp_path):
        f = tmp_path / "traveler.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        result = build_job_metadata_record(f)
        assert isinstance(result, tuple) and len(result) == 2

    def test_metadata_record_fields(self, tmp_path):
        f = tmp_path / "traveler.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, ops = build_job_metadata_record(f)
        assert meta["job_number"] == "24567"
        assert meta["part_number"] == "4140-SHAFT-001"
        assert meta["drawing_number"] == "D-4140-001"
        assert meta["revision"] == "B"
        assert "4140" in meta["material"]
        assert meta["operation_count"] == 4
        assert meta["extraction_confidence"] == "HIGH"
        assert meta["source_file"] == str(f)
        assert meta["filename"] == "traveler.txt"

    def test_all_required_columns_present(self, tmp_path):
        f = tmp_path / "traveler.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        for col in _JOB_META_COLS:
            assert col in meta, f"Missing column: {col}"

    def test_sparse_document_is_low_confidence(self, tmp_path):
        f = tmp_path / "sparse.txt"
        f.write_text(SPARSE_TEXT, encoding="utf-8")
        meta, ops = build_job_metadata_record(f)
        assert meta["extraction_confidence"] == "LOW"
        assert ops == []

    def test_drms_traveler_expected_fields(self, tmp_path):
        f = tmp_path / "OP_Traveler_DRMS.txt"
        f.write_text(DRMS_TRAVELER_TEXT, encoding="utf-8")
        meta, ops = build_job_metadata_record(f)
        assert meta["job_number"] == "D26304"
        assert meta["part_number"] == "L1D30855093"
        assert meta["drawing_number"] == "L1D30855093A"
        assert meta["revision"] == "AA"
        assert meta["raw_material_text"] == '6-1/2" Diameter 4140 HR HT'
        assert meta["normalized_material"] == "4140 HR HT"
        assert meta["material"] == "4140 HR HT"
        assert {"417 LATHE", "413 LATHE", "655 MILL", "519 GRIND"}.issubset(
            set(meta["work_centers"].split(", "))
        )
        assert len(ops) >= 4


# ---------------------------------------------------------------------------
# build_shared_print_record
# ---------------------------------------------------------------------------

class TestBuildSharedPrintRecord:
    def test_returns_dict(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        assert isinstance(rec, dict)

    def test_extracts_print_fields(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        assert rec["part_number"] == "ABC-1234-X"
        assert rec["drawing_number"] == "D-1234-REV-A"
        assert rec["revision"] == "A"
        assert "17-4" in rec["material"]

    def test_all_required_columns_present(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        for col in _PRINT_INDEX_COLS:
            assert col in rec, f"Missing column: {col}"

    def test_material_detail_fields_present(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        assert "raw_material_text" in rec
        assert "normalized_material" in rec

    def test_file_size_populated(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        assert isinstance(rec["file_size_bytes"], int)
        assert rec["file_size_bytes"] > 0


# ---------------------------------------------------------------------------
# scan_job_folders
# ---------------------------------------------------------------------------

class TestScanJobFolders:
    def _make_job_dir(self, tmp_path: Path) -> Path:
        """Create a small synthetic job folder tree."""
        root = tmp_path / "JOB FOLDERS" / "2024 Orders"
        root.mkdir(parents=True)
        (root / "24567").mkdir()
        (root / "24567" / "24567_traveler.txt").write_text(TRAVELER_TEXT, encoding="utf-8")
        (root / "98765").mkdir()
        (root / "98765" / "98765_router.txt").write_text(ROUTER_TEXT, encoding="utf-8")
        (root / "readme.docx").write_bytes(b"not scanned")  # wrong extension
        return root

    def test_returns_tuple(self, tmp_path):
        root = self._make_job_dir(tmp_path)
        result = scan_job_folders(root)
        assert isinstance(result, tuple) and len(result) == 2

    def test_finds_txt_files(self, tmp_path):
        root = self._make_job_dir(tmp_path)
        meta, _ = scan_job_folders(root)
        assert len(meta) == 2

    def test_skips_wrong_extension(self, tmp_path):
        root = self._make_job_dir(tmp_path)
        meta, _ = scan_job_folders(root)
        filenames = [m["filename"] for m in meta]
        assert "readme.docx" not in filenames

    def test_missing_root_returns_empty(self, tmp_path):
        meta, ops = scan_job_folders(tmp_path / "does_not_exist")
        assert meta == []
        assert ops == []

    def test_operations_extracted(self, tmp_path):
        root = self._make_job_dir(tmp_path)
        _, ops = scan_job_folders(root)
        assert len(ops) > 0

    def test_metadata_has_all_columns(self, tmp_path):
        root = self._make_job_dir(tmp_path)
        meta, _ = scan_job_folders(root)
        for rec in meta:
            for col in _JOB_META_COLS:
                assert col in rec, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# scan_shared_prints
# ---------------------------------------------------------------------------

class TestScanSharedPrints:
    def _make_prints_dir(self, tmp_path: Path) -> Path:
        root = tmp_path / "Shared Part Prints"
        root.mkdir(parents=True)
        (root / "ABC-1234.txt").write_text(PRINT_TEXT, encoding="utf-8")
        (root / "SS316.txt").write_text(ROUTER_TEXT, encoding="utf-8")
        (root / "notes.docx").write_bytes(b"not scanned")
        return root

    def test_finds_txt_files(self, tmp_path):
        root = self._make_prints_dir(tmp_path)
        records = scan_shared_prints(root)
        assert len(records) == 2

    def test_skips_wrong_extension(self, tmp_path):
        root = self._make_prints_dir(tmp_path)
        records = scan_shared_prints(root)
        filenames = [r["filename"] for r in records]
        assert "notes.docx" not in filenames

    def test_missing_root_returns_empty(self, tmp_path):
        records = scan_shared_prints(tmp_path / "does_not_exist")
        assert records == []

    def test_records_have_all_columns(self, tmp_path):
        root = self._make_prints_dir(tmp_path)
        records = scan_shared_prints(root)
        for rec in records:
            for col in _PRINT_INDEX_COLS:
                assert col in rec, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

class TestExportJobMetadata:
    def test_creates_csv(self, tmp_path):
        rec, _ = build_job_metadata_record(
            self._write_file(tmp_path, "t.txt", TRAVELER_TEXT)
        )
        path = export_job_metadata([rec], tmp_path, "20260101_000000")
        assert path.exists()
        assert path.name == "job_metadata_20260101_000000.csv"

    def test_csv_has_correct_columns(self, tmp_path):
        rec, _ = build_job_metadata_record(
            self._write_file(tmp_path, "t.txt", TRAVELER_TEXT)
        )
        path = export_job_metadata([rec], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        for col in _JOB_META_COLS:
            assert col in df.columns, f"Column missing: {col}"

    def test_empty_records_creates_header_only(self, tmp_path):
        path = export_job_metadata([], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        assert len(df) == 0
        for col in _JOB_META_COLS:
            assert col in df.columns

    def test_does_not_overwrite_with_different_timestamp(self, tmp_path):
        rec, _ = build_job_metadata_record(
            self._write_file(tmp_path, "t.txt", TRAVELER_TEXT)
        )
        p1 = export_job_metadata([rec], tmp_path, "20260101_000001")
        p2 = export_job_metadata([rec], tmp_path, "20260101_000002")
        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    @staticmethod
    def _write_file(tmp_path, name, content):
        f = tmp_path / name
        f.write_text(content, encoding="utf-8")
        return f


class TestExportSharedPrintIndex:
    def test_creates_csv(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        path = export_shared_print_index([rec], tmp_path, "20260101_000000")
        assert path.exists()
        assert path.name == "shared_print_index_20260101_000000.csv"

    def test_csv_has_correct_columns(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        path = export_shared_print_index([rec], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        for col in _PRINT_INDEX_COLS:
            assert col in df.columns

    def test_empty_records_creates_header_only(self, tmp_path):
        path = export_shared_print_index([], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        assert len(df) == 0


class TestExportRouterOperations:
    def test_creates_csv(self, tmp_path):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        path = export_router_operations(ops, tmp_path, "20260101_000000")
        assert path.exists()
        assert path.name == "router_operations_20260101_000000.csv"

    def test_csv_has_correct_columns(self, tmp_path):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        path = export_router_operations(ops, tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        for col in _ROUTER_OPS_COLS:
            assert col in df.columns

    def test_row_count_matches(self, tmp_path):
        ops = extract_routing_operations(TRAVELER_TEXT, "test.txt", "24567")
        path = export_router_operations(ops, tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        assert len(df) == len(ops)

    def test_empty_records_creates_header_only(self, tmp_path):
        path = export_router_operations([], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Extension constants
# ---------------------------------------------------------------------------

class TestExtensionConstants:
    def test_pdf_in_job_folder_extensions(self):
        assert ".PDF" in JOB_FOLDER_EXTENSIONS

    def test_txt_in_job_folder_extensions(self):
        assert ".TXT" in JOB_FOLDER_EXTENSIONS

    def test_pdf_in_shared_print_extensions(self):
        assert ".PDF" in SHARED_PRINT_EXTENSIONS

    def test_txt_in_shared_print_extensions(self):
        assert ".TXT" in SHARED_PRINT_EXTENSIONS


# ---------------------------------------------------------------------------
# Lookback window / scan date filtering
# ---------------------------------------------------------------------------

def _set_mtime(path: Path, days_ago: float) -> None:
    """Set a file's mtime to `days_ago` days in the past."""
    t = time.time() - days_ago * 86400
    os.utime(path, (t, t))


class TestDefaultLookbackConstant:
    def test_default_is_365(self):
        assert _DEFAULT_LOOKBACK_DAYS == 365


class TestScanWindowJobFolders:
    def _make_root(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Return (root, recent_file, old_file)."""
        root = tmp_path / "jobs"
        root.mkdir()
        recent = root / "recent.txt"
        recent.write_text(TRAVELER_TEXT, encoding="utf-8")
        old = root / "old.txt"
        old.write_text(TRAVELER_TEXT, encoding="utf-8")
        _set_mtime(old, 400)  # 400 days ago — outside 365-day window
        return root, recent, old

    def test_recent_file_included_with_default_window(self, tmp_path):
        root, _, _ = self._make_root(tmp_path)
        meta, _ = scan_job_folders(root, lookback_days=365)
        filenames = [m["filename"] for m in meta]
        assert "recent.txt" in filenames

    def test_old_file_excluded_with_default_window(self, tmp_path):
        root, _, _ = self._make_root(tmp_path)
        meta, _ = scan_job_folders(root, lookback_days=365)
        filenames = [m["filename"] for m in meta]
        assert "old.txt" not in filenames

    def test_old_file_included_with_large_window(self, tmp_path):
        root, _, _ = self._make_root(tmp_path)
        meta, _ = scan_job_folders(root, lookback_days=9999)
        filenames = [m["filename"] for m in meta]
        assert "old.txt" in filenames

    def test_zero_day_window_excludes_all(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        f = root / "file.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        _set_mtime(f, 1)  # 1 day ago — outside a 0-day window
        meta, _ = scan_job_folders(root, lookback_days=0)
        assert meta == []

    def test_operations_only_from_included_files(self, tmp_path):
        root, _, _ = self._make_root(tmp_path)
        _, ops = scan_job_folders(root, lookback_days=365)
        # Operations must all reference the recent file, not the old one
        for op in ops:
            assert "old.txt" not in op["source_file"]

    def test_custom_lookback_boundary(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        just_inside = root / "inside.txt"
        just_inside.write_text(TRAVELER_TEXT, encoding="utf-8")
        _set_mtime(just_inside, 29)  # 29 days ago

        just_outside = root / "outside.txt"
        just_outside.write_text(TRAVELER_TEXT, encoding="utf-8")
        _set_mtime(just_outside, 31)  # 31 days ago

        meta, _ = scan_job_folders(root, lookback_days=30)
        filenames = [m["filename"] for m in meta]
        assert "inside.txt" in filenames
        assert "outside.txt" not in filenames


class TestScanWindowSharedPrints:
    def _make_root(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        root = tmp_path / "prints"
        root.mkdir()
        recent = root / "recent.txt"
        recent.write_text(PRINT_TEXT, encoding="utf-8")
        old = root / "old.txt"
        old.write_text(PRINT_TEXT, encoding="utf-8")
        _set_mtime(old, 400)
        return root, recent, old

    def test_recent_file_included(self, tmp_path):
        root, _, _ = self._make_root(tmp_path)
        records = scan_shared_prints(root, lookback_days=365)
        filenames = [r["filename"] for r in records]
        assert "recent.txt" in filenames

    def test_old_file_excluded(self, tmp_path):
        root, _, _ = self._make_root(tmp_path)
        records = scan_shared_prints(root, lookback_days=365)
        filenames = [r["filename"] for r in records]
        assert "old.txt" not in filenames

    def test_old_file_included_with_large_window(self, tmp_path):
        root, _, _ = self._make_root(tmp_path)
        records = scan_shared_prints(root, lookback_days=9999)
        filenames = [r["filename"] for r in records]
        assert "old.txt" in filenames

    def test_zero_day_window_excludes_all(self, tmp_path):
        root = tmp_path / "prints"
        root.mkdir()
        f = root / "file.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        _set_mtime(f, 1)
        records = scan_shared_prints(root, lookback_days=0)
        assert records == []


# ---------------------------------------------------------------------------
# modified_datetime field
# ---------------------------------------------------------------------------

class TestModifiedDatetime:
    def test_job_metadata_record_has_modified_datetime(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        assert "modified_datetime" in meta
        assert meta["modified_datetime"] != ""

    def test_shared_print_record_has_modified_datetime(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        assert "modified_datetime" in rec
        assert rec["modified_datetime"] != ""

    def test_modified_datetime_is_iso_format(self, tmp_path):
        from datetime import datetime as dt
        f = tmp_path / "t.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        parsed = dt.fromisoformat(meta["modified_datetime"])
        assert parsed.year >= 2020

    def test_modified_datetime_reflects_mtime(self, tmp_path):
        from datetime import datetime as dt
        f = tmp_path / "t.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        _set_mtime(f, 100)  # 100 days ago
        meta, _ = build_job_metadata_record(f)
        parsed = dt.fromisoformat(meta["modified_datetime"])
        # Should be roughly 100 days ago (within 1 day tolerance)
        from datetime import timedelta
        expected = dt.now() - timedelta(days=100)
        assert abs((parsed - expected).total_seconds()) < 86400

    def test_modified_datetime_in_job_metadata_csv(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        path = export_job_metadata([meta], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        assert "modified_datetime" in df.columns
        assert df["modified_datetime"].notna().all()

    def test_modified_datetime_in_shared_print_csv(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        path = export_shared_print_index([rec], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        assert "modified_datetime" in df.columns
        assert df["modified_datetime"].notna().all()


# ===========================================================================
# Phase 6A-Optimize — classification and smart parse tests
# ===========================================================================

# ---------------------------------------------------------------------------
# classify_filename
# ---------------------------------------------------------------------------

BOL_TEXT = """\
BILL OF LADING
Shipper: ACME Corp
Consignee: EMPOWER MANUFACTURING
Tracking No: 123456789
"""

CERT_TEXT = """\
CERTIFICATE OF CONFORMANCE
Material: 316 Stainless Steel
Heat No: 123456
Certificate of Compliance
"""

class TestClassifyFilename:
    # HIGH — traveler / router keywords
    def test_traveler_filename_is_high(self):
        cls, score = classify_filename("OP_Traveler_DRMS.pdf")
        assert cls == "HIGH"
        assert score == 2

    def test_router_filename_is_high(self):
        cls, _ = classify_filename("routing_sheet.pdf")
        assert cls == "HIGH"

    def test_setup_sheet_is_high(self):
        cls, _ = classify_filename("setup_sheet_655.pdf")
        assert cls == "HIGH"

    # HIGH — EM drawing numbers
    def test_em_number_is_high(self):
        cls, _ = classify_filename("EM10986.NC")
        assert cls == "HIGH"

    def test_em_number_in_pdf_is_high(self):
        cls, _ = classify_filename("EM0825 Rev B.pdf")
        assert cls == "HIGH"

    # HIGH — D-numbered drawings (5+ digits)
    def test_d_number_drawing_is_high(self):
        cls, _ = classify_filename("D19937 OP1 Nate.pdf")
        assert cls == "HIGH"

    def test_d22310fai_is_low(self):
        # FAI suffix on a D-number → LOW (inspection report, not a traveler)
        cls, _ = classify_filename("D22310FAI OP4-Erin West.pdf")
        assert cls == "LOW"

    # HIGH — L1D Boeing drawing numbers
    def test_l1d_drawing_is_high(self):
        cls, _ = classify_filename("L1D30093240A Rev AB converted.pdf")
        assert cls == "HIGH"

    # LOW — BOL
    def test_bol_filename_is_low(self):
        cls, score = classify_filename("BOL 6 26 25.pdf")
        assert cls == "LOW"
        assert score == 0

    def test_bol_lowercase_is_low(self):
        cls, _ = classify_filename("bol.pdf")
        assert cls == "LOW"

    # LOW — certificates
    def test_cert_filename_is_low(self):
        cls, _ = classify_filename("Certificate of Origin Moosic.pdf")
        assert cls == "LOW"

    def test_material_cert_is_low(self):
        cls, _ = classify_filename("D22739 MATERIAL CERT.pdf")
        assert cls == "LOW"

    # LOW — FAI
    def test_fai_filename_is_low(self):
        cls, _ = classify_filename("D19939FAI OP10 James.pdf")
        assert cls == "LOW"

    # LOW — purchase orders / acknowledgments
    def test_purchase_order_is_low(self):
        cls, _ = classify_filename("PO_POR070586.pdf")
        assert cls == "LOW"

    def test_acknowledgment_is_low(self):
        cls, _ = classify_filename("OP_OrderAcknow.pdf")
        assert cls == "LOW"

    def test_sales_acknowledgment_is_low(self):
        cls, _ = classify_filename("Sales_Acknowledgment-SO-415540.pdf")
        assert cls == "LOW"

    # LOW — invoices
    def test_invoice_is_low(self):
        cls, _ = classify_filename("Invoice_2025.pdf")
        assert cls == "LOW"

    # MEDIUM — unclassified
    def test_generic_pdf_is_medium(self):
        cls, score = classify_filename("document.pdf")
        assert cls == "MEDIUM"
        assert score == 1

    def test_numeric_filename_is_medium(self):
        # "60454 Rev 2.pdf" — number without D prefix, not enough for HIGH
        cls, _ = classify_filename("60454 Rev 2.pdf")
        assert cls == "MEDIUM"

    # LOW wins over HIGH
    def test_em_with_cert_in_name_is_low(self):
        # LOW pattern (cert) beats HIGH pattern (EM number) — admin doc wins
        cls, _ = classify_filename("EM10986 material cert.pdf")
        assert cls == "LOW"

    def test_case_insensitive(self):
        cls, _ = classify_filename("OP_TRAVELER_DRMS.PDF")
        assert cls == "HIGH"


# ---------------------------------------------------------------------------
# classify_content
# ---------------------------------------------------------------------------

TRAVELER_CONTENT = """\
EMPOWER MANUFACTURING
ROUTING COMMENTS
WC/VENDOR   OPERATION  DESCRIPTION
Job No: 24567
Part: 4140-SHAFT-001
"""

BOL_CONTENT = """\
BILL OF LADING
Shipper Address: 123 Main St
Consignee: Customer Inc
"""

CERT_CONTENT = """\
CERTIFICATE OF CONFORMANCE
This certifies that the material meets specification.
Certificate of Compliance
"""

PO_CONTENT = """\
PURCHASE ORDER
Order No: 12345
Vendor: ACME SUPPLY
"""

FAI_CONTENT = """\
FIRST ARTICLE INSPECTION REPORT
Part Number: ABC-123
Feature 1: 1.000 +/- 0.005
"""


class TestClassifyContent:
    def test_traveler_content_is_high(self):
        cls, score = classify_content(TRAVELER_CONTENT)
        assert cls == "HIGH"
        assert score == 2

    def test_routing_comments_is_high(self):
        cls, _ = classify_content("ROUTING COMMENTS\nWC/VENDOR\nOperation 10")
        assert cls == "HIGH"

    def test_work_order_number_is_high(self):
        cls, _ = classify_content("WORK ORDER: 12345\nPart: ABC")
        assert cls == "HIGH"

    def test_job_number_line_is_high(self):
        cls, _ = classify_content("Job No: 24567\nMaterial: 4140")
        assert cls == "HIGH"

    def test_bol_content_is_low(self):
        cls, score = classify_content(BOL_CONTENT)
        assert cls == "LOW"
        assert score == 0

    def test_cert_content_is_low(self):
        cls, _ = classify_content(CERT_CONTENT)
        assert cls == "LOW"

    def test_purchase_order_content_is_low(self):
        cls, _ = classify_content(PO_CONTENT)
        assert cls == "LOW"

    def test_fai_content_is_low(self):
        cls, _ = classify_content(FAI_CONTENT)
        assert cls == "LOW"

    def test_empty_text_is_medium(self):
        cls, score = classify_content("")
        assert cls == "MEDIUM"
        assert score == 1

    def test_generic_text_is_medium(self):
        cls, _ = classify_content("Some random document text without keywords.")
        assert cls == "MEDIUM"


# ---------------------------------------------------------------------------
# decide_parse_action
# ---------------------------------------------------------------------------

class TestDecideParseAction:
    def test_high_fname_extractable_is_full_parse(self):
        assert decide_parse_action("HIGH", "MEDIUM", True) == "full_parse"

    def test_high_content_extractable_is_full_parse(self):
        assert decide_parse_action("MEDIUM", "HIGH", True) == "full_parse"

    def test_both_high_is_full_parse(self):
        assert decide_parse_action("HIGH", "HIGH", True) == "full_parse"

    def test_high_fname_low_content_is_full_parse(self):
        # fname=HIGH wins — always full_parse
        assert decide_parse_action("HIGH", "LOW", True) == "full_parse"

    def test_both_medium_is_full_parse(self):
        assert decide_parse_action("MEDIUM", "MEDIUM", True) == "full_parse"

    def test_medium_low_is_shallow_index(self):
        assert decide_parse_action("MEDIUM", "LOW", True) == "shallow_index"

    def test_low_medium_is_shallow_index(self):
        assert decide_parse_action("LOW", "MEDIUM", True) == "shallow_index"

    def test_both_low_is_skipped_low_priority(self):
        assert decide_parse_action("LOW", "LOW", True) == "skipped_low_priority"

    def test_not_extractable_is_skipped_image_pdf(self):
        assert decide_parse_action("HIGH", "HIGH", False) == "skipped_image_pdf"

    def test_not_extractable_overrides_high_priority(self):
        # Unreadable PDF → always skipped_image_pdf, regardless of filename
        assert decide_parse_action("HIGH", "LOW", False) == "skipped_image_pdf"

    def test_all_actions_are_valid(self):
        combos = [
            ("HIGH", "HIGH", True), ("HIGH", "LOW", True),
            ("MEDIUM", "MEDIUM", True), ("MEDIUM", "LOW", True),
            ("LOW", "MEDIUM", True), ("LOW", "LOW", True),
            ("HIGH", "HIGH", False),
        ]
        for f, c, e in combos:
            action = decide_parse_action(f, c, e)
            assert action in PARSE_ACTIONS, f"Invalid action '{action}' for ({f},{c},{e})"


# ---------------------------------------------------------------------------
# Classification fields in record builders
# ---------------------------------------------------------------------------

class TestClassificationFieldsInRecords:
    def test_build_job_metadata_has_classification_fields(self, tmp_path):
        f = tmp_path / "traveler.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        assert "filename_classification" in meta
        assert "content_classification" in meta
        assert "parse_priority" in meta
        assert "parse_action" in meta
        assert "text_extractable" in meta
        assert "sampled_text_length" in meta

    def test_build_shared_print_has_classification_fields(self, tmp_path):
        f = tmp_path / "print.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        for field in ("filename_classification", "content_classification",
                      "parse_priority", "parse_action",
                      "text_extractable", "sampled_text_length"):
            assert field in rec, f"Missing: {field}"

    def test_traveler_filename_classified_high(self, tmp_path):
        f = tmp_path / "OP_Traveler_part.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        assert meta["filename_classification"] == "HIGH"

    def test_traveler_content_classified_high(self, tmp_path):
        f = tmp_path / "work_order.txt"
        f.write_text("WORK ORDER: 12345\nPart: ABC-001\nMATERIAL: 4140\n", encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        assert meta["content_classification"] == "HIGH"

    def test_parse_action_defaults_to_full_parse(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        # Backward-compatible call: no preloaded_text, default parse_action
        assert meta["parse_action"] == "full_parse"

    def test_preloaded_classification_respected(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(
            f,
            preloaded_text=TRAVELER_TEXT,
            filename_classification="HIGH",
            content_classification="HIGH",
            parse_action="shallow_index",
            text_extractable=True,
            sampled_text_length=100,
        )
        assert meta["filename_classification"] == "HIGH"
        assert meta["parse_action"] == "shallow_index"
        assert meta["sampled_text_length"] == 100

    def test_text_extractable_false_on_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        assert meta["text_extractable"] == False


# ---------------------------------------------------------------------------
# Scan loop — parse action filtering
# ---------------------------------------------------------------------------

class TestScanLoopParseActions:
    def test_traveler_file_full_parsed(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        (root / "OP_Traveler_job.txt").write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = scan_job_folders(root, lookback_days=365)
        assert len(meta) == 1
        assert meta[0]["parse_action"] == "full_parse"

    def test_bol_txt_file_skipped(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        (root / "BOL_shipment.txt").write_text(BOL_TEXT, encoding="utf-8")
        meta, _ = scan_job_folders(root, lookback_days=365)
        # BOL → fname=LOW, content=LOW → skipped_low_priority → no record
        assert len(meta) == 0

    def test_cert_txt_file_skipped(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        (root / "material_cert.txt").write_text(CERT_TEXT, encoding="utf-8")
        meta, _ = scan_job_folders(root, lookback_days=365)
        assert len(meta) == 0

    def test_medium_medium_file_full_parsed(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        # Generic filename, neutral content → MEDIUM+MEDIUM → full_parse
        (root / "document.txt").write_text(
            "Some manufacturing notes without specific keywords.", encoding="utf-8"
        )
        meta, _ = scan_job_folders(root, lookback_days=365)
        assert len(meta) == 1
        assert meta[0]["parse_action"] == "full_parse"

    def test_classification_fields_present_in_scan_output(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        (root / "OP_Traveler.txt").write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = scan_job_folders(root, lookback_days=365)
        assert len(meta) == 1
        for field in ("filename_classification", "content_classification",
                      "parse_priority", "parse_action"):
            assert field in meta[0], f"Missing: {field}"

    def test_em_filename_classified_high(self, tmp_path):
        root = tmp_path / "jobs"
        root.mkdir()
        (root / "EM10986.txt").write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = scan_job_folders(root, lookback_days=365)
        assert len(meta) == 1
        assert meta[0]["filename_classification"] == "HIGH"


# ---------------------------------------------------------------------------
# Unreadable / image-only PDFs
# ---------------------------------------------------------------------------

class TestUnreadablePDF:
    def test_sample_pdf_first_page_missing_file_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.pdf"
        text, extractable = sample_pdf_first_page(path)
        assert text == ""
        assert extractable == False

    def test_decide_parse_action_image_pdf(self):
        # text_extractable=False → skipped_image_pdf regardless of other signals
        assert decide_parse_action("HIGH", "HIGH", False) == "skipped_image_pdf"
        assert decide_parse_action("MEDIUM", "MEDIUM", False) == "skipped_image_pdf"
        assert decide_parse_action("LOW", "LOW", False) == "skipped_image_pdf"


# ---------------------------------------------------------------------------
# compute_scan_metrics
# ---------------------------------------------------------------------------

class TestComputeScanMetrics:
    def _records(self, actions: list[str]) -> list[dict]:
        """Build minimal mock records with the given parse_actions."""
        return [
            {
                "parse_action": a,
                "filename_classification": "HIGH" if i % 2 == 0 else "LOW",
                "content_classification": "HIGH" if i < 3 else "MEDIUM",
                "sampled_text_length": 500,
            }
            for i, a in enumerate(actions)
        ]

    def test_empty_records_returns_zero(self):
        m = compute_scan_metrics([])
        assert m["total_records"] == 0

    def test_counts_full_parse(self):
        records = self._records(["full_parse", "full_parse", "shallow_index"])
        m = compute_scan_metrics(records)
        assert m["total_full_parsed"] == 2
        assert m["total_shallow_indexed"] == 1

    def test_total_records_correct(self):
        records = self._records(["full_parse"] * 5)
        m = compute_scan_metrics(records)
        assert m["total_records"] == 5

    def test_parse_action_distribution_present(self):
        records = self._records(["full_parse", "shallow_index", "full_parse"])
        m = compute_scan_metrics(records)
        assert "parse_action_distribution" in m
        assert m["parse_action_distribution"]["full_parse"] == 2
        assert m["parse_action_distribution"]["shallow_index"] == 1

    def test_classification_distributions_present(self):
        records = self._records(["full_parse"])
        m = compute_scan_metrics(records)
        assert "filename_cls_distribution" in m
        assert "content_cls_distribution" in m

    def test_avg_sampled_text_length(self):
        records = self._records(["full_parse", "full_parse"])
        m = compute_scan_metrics(records)
        assert m["avg_sampled_text_length"] == 500


# ---------------------------------------------------------------------------
# New column schema — backward compatibility
# ---------------------------------------------------------------------------

class TestNewColumnSchema:
    """Ensure the 6 new fields are present in all column lists and all record builders."""

    NEW_FIELDS = (
        "filename_classification", "content_classification",
        "parse_priority", "parse_action",
        "text_extractable", "sampled_text_length",
    )

    def test_new_fields_in_job_meta_cols(self):
        for f in self.NEW_FIELDS:
            assert f in _JOB_META_COLS, f"Missing from _JOB_META_COLS: {f}"

    def test_new_fields_in_print_index_cols(self):
        for f in self.NEW_FIELDS:
            assert f in _PRINT_INDEX_COLS, f"Missing from _PRINT_INDEX_COLS: {f}"

    def test_existing_fields_still_in_job_meta_cols(self):
        core = ("source_file", "filename", "job_number", "part_number",
                "drawing_number", "revision", "material", "extraction_confidence")
        for f in core:
            assert f in _JOB_META_COLS

    def test_material_detail_fields_in_metadata_cols(self):
        for f in ("raw_material_text", "normalized_material"):
            assert f in _JOB_META_COLS
            assert f in _PRINT_INDEX_COLS

    def test_work_center_detail_fields_in_router_ops_cols(self):
        for f in ("work_center_code", "work_center_type"):
            assert f in _ROUTER_OPS_COLS

    def test_job_metadata_csv_has_new_columns(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text(TRAVELER_TEXT, encoding="utf-8")
        meta, _ = build_job_metadata_record(f)
        path = export_job_metadata([meta], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        for field in self.NEW_FIELDS:
            assert field in df.columns, f"Missing in CSV: {field}"

    def test_shared_print_csv_has_new_columns(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text(PRINT_TEXT, encoding="utf-8")
        rec = build_shared_print_record(f)
        path = export_shared_print_index([rec], tmp_path, "20260101_000000")
        df = pd.read_csv(path)
        for field in self.NEW_FIELDS:
            assert field in df.columns, f"Missing in CSV: {field}"
