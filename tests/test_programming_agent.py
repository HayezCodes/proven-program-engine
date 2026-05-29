"""
test_programming_agent.py — Tests for the read-only Programming Analyst Agent.
"""

import pandas as pd
import pytest

from src.agents.programming_agent import (
    OutlierReport,
    OutlierRow,
    ProgrammingAgent,
    Recommendation,
)
from src.proven_sf_lookup import SF_LOOKUP_COLS


# ---------------------------------------------------------------------------
# Lookup table factory helpers
# ---------------------------------------------------------------------------

def _lk_row(
    material: str = "4140 HR HT",
    machine_folder: str = "417,426",
    tool_type: str = "turning_rough",
    tool_description: str = "DNMG 443-PR",
    operation_intent: str = "",
    s_mode: str = "CSS",
    S_low: float = 300.0,
    S_mid: float = 450.0,
    S_high: float = 600.0,
    f_mode: str = "IPR",
    F_low: float = 0.005,
    F_mid: float = 0.010,
    F_high: float = 0.018,
    occurrence_count: int = 50,
    confidence: str = "MEDIUM",
) -> dict:
    return {
        "material": material,
        "machine_folder": machine_folder,
        "tool_type": tool_type,
        "tool_description": tool_description,
        "operation_intent": operation_intent,
        "s_mode": s_mode,
        "S_low": S_low, "S_mid": S_mid, "S_high": S_high,
        "f_mode": f_mode,
        "F_low": F_low, "F_mid": F_mid, "F_high": F_high,
        "occurrence_count": occurrence_count,
        "confidence": confidence,
    }


def _lk(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Full SF database row factory (for outlier tests)
# ---------------------------------------------------------------------------

def _db_row(
    machine_folder: str = "417,426",
    tool_name: str = "DNMG 443-PR",
    s_mode: str = "CSS",
    f_mode: str = "IPR",
    S: float = 450.0,
    F: float = 0.010,
    verified_material: str = "4140 HR HT",
    sf_record_confidence: str = "MEDIUM",
    source_file: str = "P:/417/test.OP1",
) -> dict:
    return {
        "machine_folder": machine_folder,
        "resolved_tool_name": tool_name,
        "s_mode": s_mode,
        "f_mode": f_mode,
        "S": S,
        "F": F,
        "verified_material": verified_material,
        "sf_record_confidence": sf_record_confidence,
        "source_file": source_file,
    }


def _db(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# Construction and introspection
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_empty_lookup(self):
        agent = ProgrammingAgent(pd.DataFrame())
        assert agent.list_materials() == []

    def test_from_valid_lookup(self):
        agent = ProgrammingAgent(_lk(_lk_row()))
        mats = agent.list_materials()
        assert "4140 HR HT" in mats

    def test_unknown_material_excluded_from_list(self):
        agent = ProgrammingAgent(_lk(_lk_row(material="UNKNOWN")))
        assert "UNKNOWN" not in agent.list_materials()

    def test_list_tool_types_all(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(tool_type="turning_rough"),
            _lk_row(tool_type="turning_finish"),
        ))
        types = agent.list_tool_types()
        assert "turning_rough" in types
        assert "turning_finish" in types

    def test_list_tool_types_filtered_by_material(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(material="4140 HR HT", tool_type="turning_rough"),
            _lk_row(material="316 SS",     tool_type="milling_profile"),
        ))
        types = agent.list_tool_types("4140 HR HT")
        assert types == ["turning_rough"]
        assert "milling_profile" not in types

    def test_list_machines(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(machine_folder="417,426"),
            _lk_row(machine_folder="433, 434"),
        ))
        machines = agent.list_machines()
        assert "417,426" in machines
        assert "433, 434" in machines

    def test_list_machines_filtered_by_material(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(material="4140 HR HT", machine_folder="417,426"),
            _lk_row(material="316 SS",     machine_folder="655"),
        ))
        machines = agent.list_machines("4140 HR HT")
        assert machines == ["417,426"]


# ---------------------------------------------------------------------------
# Recommendation — material only
# ---------------------------------------------------------------------------

class TestRecommendMaterialOnly:
    def test_returns_recommendation_type(self):
        agent = ProgrammingAgent(_lk(_lk_row()))
        rec = agent.recommend("4140 HR HT")
        assert isinstance(rec, Recommendation)

    def test_material_set_correctly(self):
        agent = ProgrammingAgent(_lk(_lk_row()))
        rec = agent.recommend("4140 HR HT")
        assert rec.material == "4140 HR HT"

    def test_s_low_mid_high_populated(self):
        agent = ProgrammingAgent(_lk(_lk_row(S_low=300, S_mid=450, S_high=600)))
        rec = agent.recommend("4140 HR HT")
        assert rec.S_low == 300.0
        assert rec.S_high == 600.0
        assert rec.S_mid is not None

    def test_f_low_mid_high_populated(self):
        agent = ProgrammingAgent(_lk(_lk_row(F_low=0.005, F_mid=0.010, F_high=0.018)))
        rec = agent.recommend("4140 HR HT")
        assert rec.F_low == 0.005
        assert rec.F_high == 0.018

    def test_occurrence_count_summed(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(occurrence_count=40),
            _lk_row(occurrence_count=60),
        ))
        rec = agent.recommend("4140 HR HT")
        assert rec.occurrence_count == 100

    def test_matching_groups_count(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(tool_type="turning_rough"),
            _lk_row(tool_type="turning_finish"),
        ))
        rec = agent.recommend("4140 HR HT")
        assert rec.matching_groups == 2

    def test_confidence_level(self):
        agent = ProgrammingAgent(_lk(_lk_row(confidence="MEDIUM")))
        rec = agent.recommend("4140 HR HT")
        assert rec.confidence == "MEDIUM"

    def test_high_confidence_wins_over_medium(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(confidence="HIGH"),
            _lk_row(confidence="MEDIUM"),
        ))
        rec = agent.recommend("4140 HR HT")
        assert rec.confidence == "HIGH"

    def test_tool_types_represented(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(tool_type="turning_rough"),
            _lk_row(tool_type="threading"),
        ))
        rec = agent.recommend("4140 HR HT")
        assert "turning_rough" in rec.tool_types_represented
        assert "threading" in rec.tool_types_represented

    def test_summary_contains_material(self):
        agent = ProgrammingAgent(_lk(_lk_row()))
        rec = agent.recommend("4140 HR HT")
        assert "4140 HR HT" in rec.summary

    def test_summary_contains_confidence(self):
        agent = ProgrammingAgent(_lk(_lk_row(confidence="MEDIUM")))
        rec = agent.recommend("4140 HR HT")
        assert "MEDIUM" in rec.summary

    def test_low_occurrence_warning(self):
        agent = ProgrammingAgent(_lk(_lk_row(occurrence_count=2)))
        rec = agent.recommend("4140 HR HT")
        assert any("low" in w.lower() or "limited" in w.lower() for w in rec.warnings)

    def test_nan_modes_display_unknown(self):
        agent = ProgrammingAgent(_lk(_lk_row(s_mode=float("nan"), f_mode="nan")))
        rec = agent.recommend("4140 HR HT")
        assert rec.s_mode == "UNKNOWN"
        assert rec.f_mode == "UNKNOWN"
        assert "NAN" not in rec.summary

    def test_mixed_feed_modes_warn(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(f_mode="IPR"),
            _lk_row(tool_description="1/2 ENDMILL", tool_type="milling_profile", f_mode="IPM"),
        ))
        rec = agent.recommend("4140 HR HT")
        assert any("feed" in w.lower() and "mixed" in w.lower() for w in rec.warnings)

    def test_mixed_spindle_modes_warn(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(s_mode="CSS"),
            _lk_row(tool_description="1/2 ENDMILL", tool_type="milling_profile", s_mode="RPM"),
        ))
        rec = agent.recommend("4140 HR HT")
        assert any("spindle" in w.lower() and "mixed" in w.lower() for w in rec.warnings)

    def test_material_only_many_tool_types_warns_broad_result(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(tool_type="turning_rough"),
            _lk_row(tool_description="1/2 ENDMILL", tool_type="milling_profile"),
        ))
        rec = agent.recommend("4140 HR HT")
        assert any("material-only" in w.lower() and "tool types" in w.lower() for w in rec.warnings)


# ---------------------------------------------------------------------------
# Recommendation — with machine filter
# ---------------------------------------------------------------------------

class TestRecommendWithMachine:
    def test_machine_filter_narrows_results(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(machine_folder="417,426", S_mid=450.0),
            _lk_row(machine_folder="655",     S_mid=900.0),
        ))
        rec = agent.recommend("4140 HR HT", machine="417")
        assert rec.machine == "417"
        # Only the 417,426 row → S_mid should be 450
        assert rec.S_mid == 450.0
        assert rec.matching_groups == 1

    def test_machine_partial_match(self):
        """machine='417' should match machine_folder='417,426'."""
        agent = ProgrammingAgent(_lk(_lk_row(machine_folder="417,426")))
        rec = agent.recommend("4140 HR HT", machine="417")
        assert rec.matching_groups >= 1

    def test_machine_no_results_returns_empty(self):
        agent = ProgrammingAgent(_lk(_lk_row(machine_folder="417,426")))
        rec = agent.recommend("4140 HR HT", machine="999")
        assert rec.confidence == "NONE"
        assert rec.occurrence_count == 0
        assert len(rec.warnings) > 0


# ---------------------------------------------------------------------------
# Recommendation — with tool_type filter
# ---------------------------------------------------------------------------

class TestRecommendWithToolType:
    def test_tool_type_filter_narrows_results(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(tool_type="turning_rough",  S_mid=450.0),
            _lk_row(tool_type="turning_finish", S_mid=700.0),
        ))
        rec = agent.recommend("4140 HR HT", tool_type="turning_finish")
        assert rec.tool_type == "turning_finish"
        assert rec.S_mid == 700.0
        assert rec.matching_groups == 1

    def test_tool_types_represented_single_when_filtered(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(tool_type="turning_rough"),
            _lk_row(tool_type="threading"),
        ))
        rec = agent.recommend("4140 HR HT", tool_type="turning_rough")
        assert rec.tool_types_represented == ["turning_rough"]


# ---------------------------------------------------------------------------
# Recommendation — S/F range aggregation
# ---------------------------------------------------------------------------

class TestRangeAggregation:
    def test_s_low_is_minimum_across_groups(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(S_low=200.0, S_high=500.0, occurrence_count=10),
            _lk_row(S_low=400.0, S_high=800.0, occurrence_count=10),
        ))
        rec = agent.recommend("4140 HR HT")
        assert rec.S_low == 200.0

    def test_s_high_is_maximum_across_groups(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(S_low=200.0, S_high=500.0, occurrence_count=10),
            _lk_row(S_low=400.0, S_high=800.0, occurrence_count=10),
        ))
        rec = agent.recommend("4140 HR HT")
        assert rec.S_high == 800.0

    def test_s_mid_is_weighted_average(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(S_mid=400.0, occurrence_count=100),
            _lk_row(S_mid=600.0, occurrence_count=100),
        ))
        rec = agent.recommend("4140 HR HT")
        # Weighted avg of 400 and 600 with equal weights = 500
        assert rec.S_mid == 500.0

    def test_s_mid_weighted_toward_high_occurrences(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(S_mid=400.0, occurrence_count=10),
            _lk_row(S_mid=600.0, occurrence_count=90),
        ))
        rec = agent.recommend("4140 HR HT")
        # Weighted avg: (400*10 + 600*90) / 100 = 580
        assert rec.S_mid == 580.0


# ---------------------------------------------------------------------------
# Empty / missing material
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_unknown_material_returns_none_confidence(self):
        agent = ProgrammingAgent(_lk(_lk_row()))
        rec = agent.recommend("NONEXISTENT_MATERIAL")
        assert rec.confidence == "NONE"
        assert rec.S_mid is None
        assert rec.F_mid is None

    def test_unknown_material_has_warning(self):
        agent = ProgrammingAgent(_lk(_lk_row()))
        rec = agent.recommend("NONEXISTENT_MATERIAL")
        assert len(rec.warnings) > 0

    def test_empty_lookup_returns_empty(self):
        agent = ProgrammingAgent(pd.DataFrame())
        rec = agent.recommend("4140 HR HT")
        assert rec.confidence == "NONE"

    def test_case_insensitive_material_match(self):
        """Agent should fuzzy-match 'stainless 316' → '316 STAINLESS'."""
        agent = ProgrammingAgent(_lk(_lk_row(material="316 STAINLESS")))
        rec = agent.recommend("316 stainless")
        assert rec.material == "316 STAINLESS"
        assert rec.confidence != "NONE"

    def test_partial_material_match(self):
        """Agent should match '4140' → '4140 HR HT' when it's the only match."""
        agent = ProgrammingAgent(_lk(_lk_row(material="4140 HR HT")))
        rec = agent.recommend("4140")
        # Should resolve and find data
        assert rec.occurrence_count > 0


# ---------------------------------------------------------------------------
# Machine count
# ---------------------------------------------------------------------------

class TestMachineCount:
    def test_machine_count_across_groups(self):
        agent = ProgrammingAgent(_lk(
            _lk_row(machine_folder="417,426"),
            _lk_row(machine_folder="433, 434"),
            _lk_row(machine_folder="433, 434"),  # duplicate — should count once
        ))
        rec = agent.recommend("4140 HR HT")
        assert rec.machine_count == 2


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    def test_returns_outlier_report(self):
        agent = ProgrammingAgent(pd.DataFrame(), _db(_db_row()))
        report = agent.detect_outliers(material="4140")
        assert isinstance(report, OutlierReport)

    def test_no_outliers_in_uniform_data(self):
        rows = [_db_row(S=450.0) for _ in range(5)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        assert report.outlier_count == 0

    def test_high_speed_flagged_as_outlier(self):
        # 5 rows at S=450, 1 row at S=2000 (>2× median of 450)
        rows = [_db_row(S=450.0) for _ in range(5)] + [_db_row(S=2000.0)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        assert report.outlier_count >= 1
        flag_strs = " ".join(r.outlier_flags[0] for r in report.outliers)
        assert "above" in flag_strs.lower()

    def test_low_speed_flagged_as_outlier(self):
        # 5 rows at S=450, 1 row at S=50 (<50% of 450)
        rows = [_db_row(S=450.0) for _ in range(5)] + [_db_row(S=50.0)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        assert report.outlier_count >= 1
        flag_strs = " ".join(r.outlier_flags[0] for r in report.outliers)
        assert "below" in flag_strs.lower()

    def test_high_feed_flagged_as_outlier(self):
        rows = [_db_row(F=0.010) for _ in range(5)] + [_db_row(F=0.050)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        assert report.outlier_count >= 1

    def test_low_feed_flagged_as_outlier(self):
        rows = [_db_row(F=0.010) for _ in range(5)] + [_db_row(F=0.001)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        assert report.outlier_count >= 1

    def test_fewer_than_3_records_not_flagged(self):
        """Groups with <3 records should not produce outlier flags."""
        rows = [_db_row(S=10000.0), _db_row(S=450.0)]  # only 2 rows
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        assert report.outlier_count == 0

    def test_total_records_counted(self):
        rows = [_db_row() for _ in range(8)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        assert report.total_records_checked == 8

    def test_outlier_row_has_required_fields(self):
        rows = [_db_row(S=450.0) for _ in range(5)] + [_db_row(S=3000.0)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140")
        if report.outliers:
            o = report.outliers[0]
            assert isinstance(o, OutlierRow)
            assert len(o.outlier_flags) > 0
            assert o.group_S_median is not None

    def test_custom_threshold(self):
        """3× threshold: S=1000 vs median=450 is under 3× and should not flag."""
        rows = [_db_row(S=450.0) for _ in range(5)] + [_db_row(S=1000.0)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140", s_threshold=3.0)
        # 1000 / 450 ≈ 2.2× — below 3× threshold → no S outlier
        s_outliers = [r for r in report.outliers if any("S_above" in f for f in r.outlier_flags)]
        assert len(s_outliers) == 0

    def test_empty_db_returns_empty_report(self):
        agent = ProgrammingAgent(pd.DataFrame(), pd.DataFrame())
        report = agent.detect_outliers()
        assert report.outlier_count == 0
        assert report.total_records_checked == 0

    def test_summary_string_populated(self):
        rows = [_db_row() for _ in range(4)]
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers()
        assert len(report.summary) > 0


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_recommend_does_not_mutate_lookup(self):
        lookup = _lk(_lk_row())
        original_len = len(lookup)
        original_cols = list(lookup.columns)
        agent = ProgrammingAgent(lookup)
        agent.recommend("4140 HR HT")
        assert len(lookup) == original_len
        assert list(lookup.columns) == original_cols

    def test_detect_outliers_does_not_mutate_db(self):
        db = _db(_db_row())
        original_len = len(db)
        original_cols = list(db.columns)
        agent = ProgrammingAgent(pd.DataFrame(), db)
        agent.detect_outliers()
        assert len(db) == original_len
        # tool_type may have been added to internal copy — original must be unchanged
        assert list(db.columns) == original_cols

    def test_agent_has_no_write_methods(self):
        """Agent public API must not expose any data-mutation method.

        'from_exports' is intentionally excluded: it is a read-only factory
        that loads FROM export files, not a method that writes TO them.
        """
        agent = ProgrammingAgent(pd.DataFrame())
        # Exact write-operation verbs only — not 'export' which here means 'read from'
        forbidden = [
            name for name in dir(agent)
            if any(w in name.lower() for w in ("write", "save", "delete", "modify", "update"))
            and not name.startswith("__")
        ]
        assert forbidden == [], f"Unexpected write-like methods: {forbidden}"


# ---------------------------------------------------------------------------
# Example outputs (illustrative, not assertions)
# ---------------------------------------------------------------------------

class TestExampleOutputs:
    """Realistic examples that illustrate agent output format."""

    def _agent(self) -> ProgrammingAgent:
        rows = [
            _lk_row(material="4140 HR HT", machine_folder="417,426",
                    tool_type="turning_rough",  tool_description="DNMG 443-PR",
                    s_mode="CSS", S_low=300, S_mid=450, S_high=600,
                    f_mode="IPR", F_low=0.007, F_mid=0.010, F_high=0.018,
                    occurrence_count=437, confidence="MEDIUM"),
            _lk_row(material="4140 HR HT", machine_folder="417,426",
                    tool_type="turning_finish", tool_description="VNMG 332-PF",
                    s_mode="CSS", S_low=500, S_mid=650, S_high=800,
                    f_mode="IPR", F_low=0.004, F_mid=0.006, F_high=0.009,
                    occurrence_count=182, confidence="MEDIUM"),
            _lk_row(material="316 STAINLESS", machine_folder="655",
                    tool_type="milling_profile", tool_description='3/8" FLAT ENDMILL',
                    s_mode="RPM", S_low=1800, S_mid=2400, S_high=3200,
                    f_mode="IPM", F_low=6.0, F_mid=9.0, F_high=14.0,
                    occurrence_count=94, confidence="MEDIUM"),
        ]
        return ProgrammingAgent(_lk(*rows))

    def test_material_only_recommendation_format(self):
        rec = self._agent().recommend("4140 HR HT")
        assert rec.occurrence_count == 619
        assert rec.matching_groups == 2
        assert rec.S_low == 300.0
        assert rec.S_high == 800.0
        assert "4140 HR HT" in rec.summary

    def test_material_plus_tool_type(self):
        rec = self._agent().recommend("4140 HR HT", tool_type="turning_rough")
        assert rec.S_mid == 450.0
        assert rec.F_mid == 0.010
        assert rec.matching_groups == 1

    def test_machine_filter(self):
        rec = self._agent().recommend("316 STAINLESS", machine="655")
        assert rec.s_mode == "RPM"
        assert rec.S_mid == 2400.0

    def test_recommendation_summary_is_human_readable(self, capsys):
        rec = self._agent().recommend("4140 HR HT", tool_type="turning_rough")
        print(rec.summary)
        captured = capsys.readouterr()
        assert "4140 HR HT" in captured.out
        assert "CSS" in captured.out or "IPR" in captured.out

    def test_outlier_example(self):
        rows = (
            [_db_row(S=450.0, verified_material="4140 HR HT") for _ in range(10)]
            + [_db_row(S=2500.0, verified_material="4140 HR HT")]
        )
        agent = ProgrammingAgent(pd.DataFrame(), _db(*rows))
        report = agent.detect_outliers(material="4140 HR HT")
        assert report.outlier_count >= 1
        assert "2500" in report.outliers[0].outlier_flags[0] or "outlier" in report.summary.lower()
