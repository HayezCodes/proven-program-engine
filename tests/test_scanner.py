"""
test_scanner.py — Tests for the scanner module.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.scanner import (
    ALLOWED_EXTENSIONS,
    find_proven_folders,
    scan_files,
    export_manifest,
)


class TestFindProvenFolders:
    def test_finds_two_proven_folders(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        assert len(folders) == 2

    def test_case_insensitive_match(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        names_lower = {f.name.lower() for f in folders}
        assert "proven" in names_lower

    def test_nonexistent_root_returns_empty(self, tmp_path):
        folders = find_proven_folders(tmp_path / "does_not_exist")
        assert folders == []

    def test_no_proven_folder_returns_empty(self, tmp_path):
        (tmp_path / "machine1" / "RECEIVE").mkdir(parents=True)
        folders = find_proven_folders(tmp_path)
        assert folders == []

    def test_returns_paths_to_proven_directories(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        for f in folders:
            assert f.is_dir()


class TestScanFiles:
    def _cutoff(self, days: int = 730) -> datetime:
        return datetime.now() - timedelta(days=days)

    def test_returns_records_for_all_files(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        records = scan_files(proven_dir_structure, folders, self._cutoff())
        assert len(records) == 2

    def test_included_files_pass_extension_check(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        records = scan_files(proven_dir_structure, folders, self._cutoff())
        for rec in (r for r in records if r["included"]):
            assert rec["extension"].upper() in ALLOWED_EXTENSIONS

    def test_required_manifest_fields_present(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        records = scan_files(proven_dir_structure, folders, self._cutoff())
        required = {
            "program_id", "source_file", "relative_path", "machine_folder",
            "filename", "extension", "modified_datetime", "file_size_bytes",
            "included", "skip_reason",
        }
        for rec in records:
            assert required.issubset(rec.keys())

    def test_excluded_extension_marked_not_included(self, proven_dir_structure):
        bad_file = proven_dir_structure / "421, 423, 424" / "Proven" / "notes.DOCX"
        bad_file.write_text("not cnc")
        folders = find_proven_folders(proven_dir_structure)
        records = scan_files(proven_dir_structure, folders, self._cutoff())
        bad = [r for r in records if r["filename"] == "notes.DOCX"]
        assert bad and bad[0]["included"] is False
        assert "extension_excluded" in bad[0]["skip_reason"]

    def test_old_file_marked_not_included(self, proven_dir_structure, tmp_path):
        # Use a future cutoff to exclude everything
        future_cutoff = datetime.now() + timedelta(days=1)
        folders = find_proven_folders(proven_dir_structure)
        records = scan_files(proven_dir_structure, folders, future_cutoff)
        assert all(not r["included"] for r in records)
        assert all("too_old" in r["skip_reason"] for r in records)

    def test_program_ids_are_sequential(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        records = scan_files(proven_dir_structure, folders, self._cutoff())
        ids = [r["program_id"] for r in records]
        assert ids == list(range(1, len(ids) + 1))

    def test_machine_folder_name_captured(self, proven_dir_structure):
        folders = find_proven_folders(proven_dir_structure)
        records = scan_files(proven_dir_structure, folders, self._cutoff())
        machine_names = {r["machine_folder"] for r in records}
        assert "421, 423, 424" in machine_names
        assert "432, 437" in machine_names


class TestOP123Extensions:
    """OP1 / OP2 / OP3 files must be treated identically to NC/EIA files."""

    def test_op_extensions_in_allowed_set(self):
        assert ".OP1" in ALLOWED_EXTENSIONS
        assert ".OP2" in ALLOWED_EXTENSIONS
        assert ".OP3" in ALLOWED_EXTENSIONS

    def test_op1_file_included(self, proven_dir_with_op_files):
        cutoff = datetime.now() - timedelta(days=730)
        folders = find_proven_folders(proven_dir_with_op_files)
        records = scan_files(proven_dir_with_op_files, folders, cutoff)
        op1 = [r for r in records if r["filename"].upper().endswith(".OP1")]
        assert op1 and op1[0]["included"] is True

    def test_op2_file_included(self, proven_dir_with_op_files):
        cutoff = datetime.now() - timedelta(days=730)
        folders = find_proven_folders(proven_dir_with_op_files)
        records = scan_files(proven_dir_with_op_files, folders, cutoff)
        op2 = [r for r in records if r["filename"].upper().endswith(".OP2")]
        assert op2 and op2[0]["included"] is True

    def test_op3_file_included(self, proven_dir_with_op_files):
        cutoff = datetime.now() - timedelta(days=730)
        folders = find_proven_folders(proven_dir_with_op_files)
        records = scan_files(proven_dir_with_op_files, folders, cutoff)
        op3 = [r for r in records if r["filename"].upper().endswith(".OP3")]
        assert op3 and op3[0]["included"] is True

    def test_lowercase_op1_extension_included(self, tmp_path):
        """Extension matching must be case-insensitive."""
        folder = tmp_path / "machine" / "Proven"
        folder.mkdir(parents=True)
        (folder / "part.op1").write_text("%\nT0101\nG97 S900 M03\nM30\n%\n")
        cutoff = datetime.now() - timedelta(days=730)
        folders = find_proven_folders(tmp_path)
        records = scan_files(tmp_path, folders, cutoff)
        rec = [r for r in records if "op1" in r["filename"].lower()]
        assert rec and rec[0]["included"] is True

    def test_mixed_case_op2_extension_included(self, tmp_path):
        folder = tmp_path / "machine" / "Proven"
        folder.mkdir(parents=True)
        (folder / "part.Op2").write_text("%\nT0101\nG97 S900 M03\nM30\n%\n")
        cutoff = datetime.now() - timedelta(days=730)
        folders = find_proven_folders(tmp_path)
        records = scan_files(tmp_path, folders, cutoff)
        rec = [r for r in records if "op2" in r["filename"].lower()]
        assert rec and rec[0]["included"] is True


class TestExportManifest:
    def test_creates_csv_and_json(self, proven_dir_structure, tmp_path):
        folders = find_proven_folders(proven_dir_structure)
        cutoff = datetime.now() - timedelta(days=730)
        records = scan_files(proven_dir_structure, folders, cutoff)
        exports_dir = tmp_path / "exports"
        csv_path, json_path = export_manifest(records, exports_dir)
        assert csv_path.exists()
        assert json_path.exists()

    def test_csv_row_count_matches_records(self, proven_dir_structure, tmp_path):
        import pandas as pd
        folders = find_proven_folders(proven_dir_structure)
        cutoff = datetime.now() - timedelta(days=730)
        records = scan_files(proven_dir_structure, folders, cutoff)
        csv_path, _ = export_manifest(records, tmp_path / "exports")
        df = pd.read_csv(csv_path)
        assert len(df) == len(records)
