"""
test_dashboard_overrides.py — Tests for the dashboard overrides module.
"""

import pandas as pd
import pytest
from pathlib import Path

from src.dashboard.data_access.overrides import (
    load_tooling_overrides,
    save_tooling_overrides,
    upsert_tooling_override,
    apply_tooling_overrides,
    load_material_overrides,
    upsert_material_override,
    apply_material_overrides,
)


# ---------------------------------------------------------------------------
# Tooling overrides
# ---------------------------------------------------------------------------

class TestLoadToolingOverrides:
    def test_returns_empty_df_when_file_missing(self, tmp_path):
        result = load_tooling_overrides(tmp_path / "missing.csv")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_df_when_file_exists(self, tmp_path):
        f = tmp_path / "tooling_overrides.csv"
        f.write_text("machine_folder,tool_number,review_action,corrected_description,notes,override_timestamp\n"
                     "655,1,accept,1/2 end mill,,2026-01-01 12:00:00\n")
        result = load_tooling_overrides(f)
        assert len(result) == 1
        assert result.iloc[0]["machine_folder"] == "655"


class TestSaveToolingOverrides:
    def test_creates_csv(self, tmp_path):
        df = pd.DataFrame([{
            "machine_folder": "655", "tool_number": "1",
            "review_action": "accept", "corrected_description": "1/2",
            "notes": "", "override_timestamp": "2026-01-01 12:00:00",
        }])
        path = tmp_path / "tooling_overrides.csv"
        save_tooling_overrides(df, path)
        assert path.exists()

    def test_roundtrip(self, tmp_path):
        df = pd.DataFrame([{
            "machine_folder": "655", "tool_number": "1",
            "review_action": "accept", "corrected_description": "1/2",
            "notes": "test note", "override_timestamp": "2026-01-01 12:00:00",
        }])
        path = tmp_path / "tooling_overrides.csv"
        save_tooling_overrides(df, path)
        loaded = load_tooling_overrides(path)
        assert loaded.iloc[0]["notes"] == "test note"


class TestUpsertToolingOverride:
    def test_inserts_new_record(self, tmp_path):
        path = tmp_path / "tooling_overrides.csv"
        upsert_tooling_override("655", 1, "accept", "1/2", "", path)
        df = load_tooling_overrides(path)
        assert len(df) == 1
        assert df.iloc[0]["review_action"] == "accept"

    def test_updates_existing_record(self, tmp_path):
        path = tmp_path / "tooling_overrides.csv"
        upsert_tooling_override("655", 1, "accept", "1/2", "", path)
        upsert_tooling_override("655", 1, "reject", "3/8", "wrong size", path)
        df = load_tooling_overrides(path)
        assert len(df) == 1
        assert df.iloc[0]["review_action"] == "reject"
        assert df.iloc[0]["corrected_description"] == "3/8"

    def test_two_different_keys_give_two_rows(self, tmp_path):
        path = tmp_path / "tooling_overrides.csv"
        upsert_tooling_override("655", 1, "accept", "1/2", "", path)
        upsert_tooling_override("655", 2, "accept", "3/8", "", path)
        df = load_tooling_overrides(path)
        assert len(df) == 2

    def test_timestamp_is_set(self, tmp_path):
        path = tmp_path / "tooling_overrides.csv"
        upsert_tooling_override("655", 1, "accept", "1/2", "", path)
        df = load_tooling_overrides(path)
        assert df.iloc[0]["override_timestamp"] != ""

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "subdir" / "tooling_overrides.csv"
        upsert_tooling_override("655", 1, "accept", "1/2", "", path)
        assert path.exists()


# ---------------------------------------------------------------------------
# apply_tooling_overrides
# ---------------------------------------------------------------------------

_REVIEW_BASE = pd.DataFrame([
    {"machine_folder": "655", "tool_number": 1, "active_t_code": "T01",
     "review_action": "", "corrected_description": "", "notes": ""},
    {"machine_folder": "655", "tool_number": 2, "active_t_code": "T02",
     "review_action": "", "corrected_description": "", "notes": ""},
])

class TestApplyToolingOverrides:
    def test_no_overrides_adds_is_overridden_false(self, tmp_path):
        result = apply_tooling_overrides(_REVIEW_BASE, pd.DataFrame())
        assert "is_overridden" in result.columns
        assert not result["is_overridden"].any()

    def test_matching_override_applied(self, tmp_path):
        ov = pd.DataFrame([{
            "machine_folder": "655", "tool_number": "1",
            "review_action": "accept", "corrected_description": "1/2",
            "notes": "confirmed", "override_timestamp": "2026-01-01",
        }])
        result = apply_tooling_overrides(_REVIEW_BASE, ov)
        row = result[result["tool_number"] == 1].iloc[0]
        assert row["is_overridden"] is True
        assert row["review_action"] == "accept"
        assert row["corrected_description"] == "1/2"

    def test_non_matching_row_not_overridden(self, tmp_path):
        ov = pd.DataFrame([{
            "machine_folder": "655", "tool_number": "1",
            "review_action": "accept", "corrected_description": "1/2",
            "notes": "", "override_timestamp": "2026-01-01",
        }])
        result = apply_tooling_overrides(_REVIEW_BASE, ov)
        row = result[result["tool_number"] == 2].iloc[0]
        assert row["is_overridden"] is False

    def test_none_input_returns_none(self):
        result = apply_tooling_overrides(None)
        assert result is None

    def test_empty_input_returns_empty(self):
        result = apply_tooling_overrides(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Material overrides
# ---------------------------------------------------------------------------

class TestLoadMaterialOverrides:
    def test_returns_empty_when_missing(self, tmp_path):
        result = load_material_overrides(tmp_path / "missing.csv")
        assert result.empty

    def test_returns_df_when_present(self, tmp_path):
        f = tmp_path / "material_overrides.csv"
        f.write_text(
            "machine_folder,active_t_code,tool_number,review_decision,reviewer_note,override_timestamp\n"
            "655,T01,1,approved,,2026-01-01\n"
        )
        result = load_material_overrides(f)
        assert len(result) == 1


class TestUpsertMaterialOverride:
    def test_inserts_record(self, tmp_path):
        path = tmp_path / "material_overrides.csv"
        upsert_material_override("655", "T01", 1, "approved", "", path)
        df = load_material_overrides(path)
        assert len(df) == 1
        assert df.iloc[0]["review_decision"] == "approved"

    def test_updates_existing_key(self, tmp_path):
        path = tmp_path / "material_overrides.csv"
        upsert_material_override("655", "T01", 1, "approved", "", path)
        upsert_material_override("655", "T01", 1, "rejected", "wrong material", path)
        df = load_material_overrides(path)
        assert len(df) == 1
        assert df.iloc[0]["review_decision"] == "rejected"


# ---------------------------------------------------------------------------
# apply_material_overrides
# ---------------------------------------------------------------------------

class TestApplyMaterialOverrides:
    def test_adds_review_decision_pending_by_default(self):
        mc = pd.DataFrame([{
            "machine_folder": "655", "active_t_code": "T01",
            "tool_number": 1, "material_candidate_1": "4140",
        }])
        result = apply_material_overrides(mc, pd.DataFrame())
        assert "review_decision" in result.columns
        assert result.iloc[0]["review_decision"] == "pending"

    def test_applies_approved_decision(self):
        mc = pd.DataFrame([{
            "machine_folder": "655", "active_t_code": "T01",
            "tool_number": 1, "material_candidate_1": "4140",
        }])
        ov = pd.DataFrame([{
            "machine_folder": "655", "active_t_code": "T01", "tool_number": "1",
            "review_decision": "approved", "reviewer_note": "confirmed",
            "override_timestamp": "2026-01-01",
        }])
        result = apply_material_overrides(mc, ov)
        assert result.iloc[0]["review_decision"] == "approved"

    def test_none_returns_none(self):
        assert apply_material_overrides(None) is None

    def test_empty_returns_empty(self):
        result = apply_material_overrides(pd.DataFrame())
        assert result.empty
