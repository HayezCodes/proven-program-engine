"""
test_safety.py — Tests for the production write protection guard.

Every write point in the codebase calls assert_safe_write() before touching
a file. These tests verify the guard correctly blocks production paths and
allows project-local and temp-dir paths.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.safety import (
    PROJECT_ROOT,
    ALLOWED_WRITE_ROOTS,
    ProductionWriteViolation,
    assert_safe_write,
)


# ---------------------------------------------------------------------------
# Core guard — allowed paths
# ---------------------------------------------------------------------------

class TestAllowedPaths:
    def test_exports_dir_allowed(self):
        assert_safe_write(PROJECT_ROOT / "exports" / "manifest_test.csv")

    def test_logs_dir_allowed(self):
        assert_safe_write(PROJECT_ROOT / "logs" / "ppe_20260101.log")

    def test_data_overrides_allowed(self):
        assert_safe_write(PROJECT_ROOT / "data" / "overrides" / "tooling_overrides.csv")

    def test_project_root_itself_allowed(self):
        assert_safe_write(PROJECT_ROOT / "any_new_file.txt")

    def test_nested_subdir_allowed(self):
        assert_safe_write(PROJECT_ROOT / "exports" / "subdir" / "file.csv")

    def test_temp_dir_allowed(self):
        tmp = Path(tempfile.gettempdir()) / "ppe_test_output.csv"
        assert_safe_write(tmp)

    def test_pytest_tmp_path_allowed(self, tmp_path):
        assert_safe_write(tmp_path / "test_output.csv")

    def test_pytest_tmp_path_nested_allowed(self, tmp_path):
        assert_safe_write(tmp_path / "sub" / "dir" / "output.csv")

    def test_returns_none_on_success(self):
        result = assert_safe_write(PROJECT_ROOT / "exports" / "x.csv")
        assert result is None


# ---------------------------------------------------------------------------
# Core guard — blocked paths
# ---------------------------------------------------------------------------

class TestBlockedPaths:
    def test_p_drive_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"P:\Manufacturing\test.csv"))

    def test_p_drive_proven_folder_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"P:\655\Proven\test_output.NC"))

    def test_g_drive_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"G:\Manufacturing\JOB FOLDERS\test.csv"))

    def test_g_drive_shared_prints_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(
                Path(r"G:\Manufacturing\Programming\CAM Files\Shared Part Prints\test.pdf")
            )

    def test_m_drive_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"M:\SomeFolder\test.csv"))

    def test_windows_system_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"C:\Windows\test.csv"))

    def test_other_user_folder_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"C:\Users\OtherUser\Desktop\test.csv"))

    def test_program_files_blocked(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"C:\Program Files\test.csv"))


# ---------------------------------------------------------------------------
# Exception type and message
# ---------------------------------------------------------------------------

class TestExceptionDetails:
    def test_is_permission_error_subclass(self):
        with pytest.raises(PermissionError):
            assert_safe_write(Path(r"P:\test.csv"))

    def test_is_production_write_violation(self):
        with pytest.raises(ProductionWriteViolation):
            assert_safe_write(Path(r"G:\test.csv"))

    def test_error_message_contains_blocked_path(self):
        bad = Path(r"P:\Manufacturing\output.csv")
        with pytest.raises(ProductionWriteViolation, match=r"P:"):
            assert_safe_write(bad)

    def test_error_message_contains_read_only_warning(self):
        with pytest.raises(ProductionWriteViolation, match="READ-ONLY"):
            assert_safe_write(Path(r"G:\any\path.csv"))

    def test_error_message_contains_project_root(self):
        with pytest.raises(ProductionWriteViolation, match="proven_program_engine"):
            assert_safe_write(Path(r"P:\any.csv"))


# ---------------------------------------------------------------------------
# PROJECT_ROOT constant
# ---------------------------------------------------------------------------

class TestProjectRoot:
    def test_project_root_is_absolute(self):
        assert PROJECT_ROOT.is_absolute()

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_project_root_contains_src(self):
        assert (PROJECT_ROOT / "src").is_dir()

    def test_allowed_write_roots_are_under_project(self):
        for root in ALLOWED_WRITE_ROOTS:
            root.relative_to(PROJECT_ROOT)  # should not raise


# ---------------------------------------------------------------------------
# Integration: export functions call assert_safe_write
# ---------------------------------------------------------------------------

class TestExportFunctionsBlocked:
    """Confirm each export function is guarded — calling with a production
    path raises ProductionWriteViolation before any file is touched."""

    def test_scanner_export_manifest_blocked(self):
        from src.scanner import export_manifest
        with pytest.raises(ProductionWriteViolation):
            export_manifest([], Path(r"P:\Manufacturing"))

    def test_exports_export_tool_summary_blocked(self):
        from src.exports import export_tool_summary
        with pytest.raises(ProductionWriteViolation):
            export_tool_summary([], Path(r"G:\Manufacturing"))

    def test_exports_export_parser_summary_blocked(self):
        from src.exports import export_parser_summary
        with pytest.raises(ProductionWriteViolation):
            export_parser_summary([], Path(r"P:\Manufacturing"))

    def test_material_matcher_blocked(self):
        from src.material_matcher import export_material_candidates
        with pytest.raises(ProductionWriteViolation):
            export_material_candidates(pd.DataFrame(), [], Path(r"G:\Manufacturing"))

    def test_reference_loader_blocked(self):
        from src.reference_loader import export_reference
        with pytest.raises(ProductionWriteViolation):
            export_reference([], Path(r"P:\Manufacturing"))

    def test_tooling_reference_loader_blocked(self):
        from src.tooling_reference_loader import export_tooling_reference
        with pytest.raises(ProductionWriteViolation):
            export_tooling_reference([], Path(r"P:\Manufacturing"))

    def test_tooling_matcher_blocked(self):
        from src.tooling_matcher import export_tooling_review
        with pytest.raises(ProductionWriteViolation):
            export_tooling_review(pd.DataFrame(), [], Path(r"P:\Manufacturing"))

    def test_job_metadata_export_job_metadata_blocked(self):
        from src.job_metadata_scanner import export_job_metadata
        with pytest.raises(ProductionWriteViolation):
            export_job_metadata([], Path(r"G:\Manufacturing"), "20260101_000000")

    def test_job_metadata_export_shared_print_blocked(self):
        from src.job_metadata_scanner import export_shared_print_index
        with pytest.raises(ProductionWriteViolation):
            export_shared_print_index([], Path(r"G:\Manufacturing"), "20260101_000000")

    def test_job_metadata_export_router_ops_blocked(self):
        from src.job_metadata_scanner import export_router_operations
        with pytest.raises(ProductionWriteViolation):
            export_router_operations([], Path(r"G:\Manufacturing"), "20260101_000000")

    def test_job_linker_export_links_blocked(self):
        from src.job_linker import export_program_job_links
        with pytest.raises(ProductionWriteViolation):
            export_program_job_links(pd.DataFrame(), Path(r"P:\Manufacturing"), "20260101_000000")

    def test_job_linker_export_backfill_blocked(self):
        from src.job_linker import export_material_backfill
        with pytest.raises(ProductionWriteViolation):
            export_material_backfill(pd.DataFrame(), Path(r"P:\Manufacturing"), "20260101_000000")

    def test_job_linker_export_context_blocked(self):
        from src.job_linker import export_router_program_context
        with pytest.raises(ProductionWriteViolation):
            export_router_program_context(pd.DataFrame(), Path(r"P:\Manufacturing"), "20260101_000000")

    def test_overrides_save_tooling_blocked(self):
        from src.dashboard.data_access.overrides import save_tooling_overrides
        with pytest.raises(ProductionWriteViolation):
            save_tooling_overrides(pd.DataFrame(), path=Path(r"P:\overrides.csv"))

    def test_tooldb_export_blocked(self):
        from src.tooldb_loader import export_tooldb_reference
        with pytest.raises(ProductionWriteViolation):
            export_tooldb_reference(pd.DataFrame(), Path(r"G:\Manufacturing"))


# ---------------------------------------------------------------------------
# Export functions work normally when given a safe path
# ---------------------------------------------------------------------------

class TestExportFunctionsAllowed:
    def test_scanner_export_manifest_safe(self, tmp_path):
        from src.scanner import export_manifest
        csv, json = export_manifest([], tmp_path)
        assert csv.exists()
        assert json.exists()

    def test_job_linker_export_links_safe(self, tmp_path):
        from src.job_linker import export_program_job_links
        from src.job_linker import _PROG_JOB_LINK_COLS
        path = export_program_job_links(
            pd.DataFrame(columns=_PROG_JOB_LINK_COLS), tmp_path, "20260101_000000"
        )
        assert path.exists()

    def test_job_metadata_export_safe(self, tmp_path):
        from src.job_metadata_scanner import export_job_metadata
        path = export_job_metadata([], tmp_path, "20260101_000000")
        assert path.exists()
