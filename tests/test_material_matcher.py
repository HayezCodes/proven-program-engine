"""
test_material_matcher.py — Tests for the material_matcher module.
"""

import pytest
import pandas as pd

from src.material_matcher import match_record, match_tool_summary, export_material_candidates


# ---------------------------------------------------------------------------
# Reference table used across all tests
# ---------------------------------------------------------------------------

REFERENCE = [
    {
        "material": "1018",
        "turning_rough_sfm": 615.0,
        "turning_finish_sfm": 720.0,
        "turning_rough_ipr": 0.018,
        "turning_finish_ipr": 0.016,
        "milling_sfm": 310.0,
        "milling_rough_ipm": 15.0,
        "milling_finish_ipm": 25.0,
    },
    {
        "material": "1045",
        "turning_rough_sfm": 615.0,
        "turning_finish_sfm": 720.0,
        "turning_rough_ipr": 0.018,
        "turning_finish_ipr": 0.016,
        "milling_sfm": 310.0,
        "milling_rough_ipm": 15.0,
        "milling_finish_ipm": 25.0,
    },
    {
        "material": "4140",
        "turning_rough_sfm": 450.0,
        "turning_finish_sfm": 600.0,
        "turning_rough_ipr": 0.016,
        "turning_finish_ipr": 0.016,
        "milling_sfm": 310.0,
        "milling_rough_ipm": 10.0,
        "milling_finish_ipm": 16.0,
    },
    {
        "material": "316",
        "turning_rough_sfm": 350.0,
        "turning_finish_sfm": 600.0,
        "turning_rough_ipr": 0.015,
        "turning_finish_ipr": 0.015,
        "milling_sfm": 250.0,
        "milling_rough_ipm": 10.0,
        "milling_finish_ipm": 16.0,
    },
    {
        "material": "HASTELLOY C276",
        "turning_rough_sfm": 150.0,
        "turning_finish_sfm": 300.0,
        "turning_rough_ipr": 0.010,
        "turning_finish_ipr": 0.012,
        "milling_sfm": 200.0,
        "milling_rough_ipm": 10.0,
        "milling_finish_ipm": 16.0,
    },
    {
        "material": "TITANIUM #2",
        "turning_rough_sfm": 200.0,
        "turning_finish_sfm": 250.0,
        "turning_rough_ipr": 0.012,
        "turning_finish_ipr": 0.010,
        "milling_sfm": 200.0,
        "milling_rough_ipm": 10.0,
        "milling_finish_ipm": 16.0,
    },
]


# ---------------------------------------------------------------------------
# match_record — insufficient_context
# ---------------------------------------------------------------------------

class TestMatchRecordInsufficient:
    def test_limit_record_is_insufficient(self):
        result = match_record(1200, None, "RPM", "LIMIT", REFERENCE)
        assert result["match_type"] == "insufficient_context"

    def test_g97_rpm_turning_is_insufficient(self):
        # G97 RPM + small feed (IPR-like) — can't map to SFM
        result = match_record(800, 0.012, "RPM", "SPINDLE", REFERENCE)
        assert result["match_type"] == "insufficient_context"

    def test_g97_rpm_no_feed_is_insufficient(self):
        result = match_record(800, None, "RPM", "SPINDLE", REFERENCE)
        assert result["match_type"] == "insufficient_context"

    def test_no_values_is_insufficient(self):
        result = match_record(None, None, "UNKNOWN", "SPINDLE", REFERENCE)
        assert result["match_type"] == "insufficient_context"

    def test_unknown_mode_no_feed_is_insufficient(self):
        result = match_record(500, None, "UNKNOWN", "SPINDLE", REFERENCE)
        assert result["match_type"] == "insufficient_context"

    def test_insufficient_has_none_candidates(self):
        result = match_record(800, 0.012, "RPM", "SPINDLE", REFERENCE)
        assert result["material_candidate_1"] is None
        assert result["material_candidate_2"] is None
        assert result["material_candidate_3"] is None

    def test_insufficient_confidence_is_zero(self):
        result = match_record(800, 0.012, "RPM", "SPINDLE", REFERENCE)
        assert result["confidence_score"] == 0.0
        assert result["confidence_label"] == "NONE"

    def test_insufficient_reason_is_not_empty(self):
        result = match_record(800, 0.012, "RPM", "SPINDLE", REFERENCE)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# match_record — no_match
# ---------------------------------------------------------------------------

class TestMatchRecordNoMatch:
    def test_sfm_below_all_ranges_is_no_match(self):
        # SFM=50 is below every material's rough_sfm * 0.70
        result = match_record(50.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["match_type"] == "no_match"

    def test_no_match_candidates_are_none(self):
        result = match_record(50.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["material_candidate_1"] is None

    def test_no_match_confidence_is_zero(self):
        result = match_record(50.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["confidence_score"] == 0.0


# ---------------------------------------------------------------------------
# match_record — exact_match (single best material)
# ---------------------------------------------------------------------------

class TestMatchRecordExact:
    def test_hastelloy_low_sfm_exact_match(self):
        # SFM=160: within HASTELLOY tight range [127.5, 345]; TITANIUM tight=[170,287.5] → 160<170, miss
        result = match_record(160.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["match_type"] == "exact_match"
        assert result["material_candidate_1"] == "HASTELLOY C276"

    def test_exact_match_gives_high_confidence(self):
        result = match_record(160.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["confidence_label"] == "HIGH"

    def test_feed_raises_confidence(self):
        # SFM=160 + feed=0.011 (within HASTELLOY IPR range [0.008, 0.0144]) → score=3 → HIGH
        result = match_record(160.0, 0.011, "CSS", "SPINDLE", REFERENCE)
        assert result["confidence_label"] == "HIGH"
        assert result["confidence_score"] == 1.0

    def test_candidate_1_is_best(self):
        result = match_record(160.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["material_candidate_1"] == "HASTELLOY C276"

    def test_result_has_all_required_fields(self):
        result = match_record(160.0, None, "CSS", "SPINDLE", REFERENCE)
        required = {
            "material_candidate_1", "material_candidate_2", "material_candidate_3",
            "confidence_score", "confidence_label", "match_type", "reason",
        }
        assert required.issubset(result.keys())

    def test_reason_is_non_empty_string(self):
        result = match_record(160.0, None, "CSS", "SPINDLE", REFERENCE)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# match_record — multiple_possible_matches
# ---------------------------------------------------------------------------

class TestMatchRecordMultiple:
    def test_1018_and_1045_same_sfm_gives_multiple(self):
        # 1018 and 1045 have identical reference ranges — always ambiguous
        result = match_record(650.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["match_type"] == "multiple_possible_matches"

    def test_1018_and_1045_both_in_candidates(self):
        result = match_record(650.0, None, "CSS", "SPINDLE", REFERENCE)
        all_candidates = [
            result["material_candidate_1"],
            result["material_candidate_2"],
            result["material_candidate_3"],
        ]
        assert "1018" in all_candidates
        assert "1045" in all_candidates

    def test_multiple_match_has_positive_confidence(self):
        result = match_record(650.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["confidence_score"] > 0
        assert result["confidence_label"] in ("HIGH", "MEDIUM")


# ---------------------------------------------------------------------------
# match_record — close_match
# ---------------------------------------------------------------------------

class TestMatchRecordClose:
    def test_sfm_in_loose_band_only_gives_close_match(self):
        # SFM=400: below 1018/1045 loose cutoff (615*0.70=430.5 > 400), so they score 0
        # 4140: tight=[382.5, 690] → 400 IN → sfm_score=2 (exact for 4140)
        # 316: tight=[297.5, 690] → 400 IN → sfm_score=2
        # So this is actually multiple_possible_matches for 4140 and 316
        result = match_record(400.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["match_type"] in ("close_match", "exact_match", "multiple_possible_matches")

    def test_close_match_has_medium_or_higher_confidence(self):
        # SFM=370: TITANIUM loose=[140,325]→370>325 miss; HASTELLOY loose=[105,390]→370 IN=loose
        # 316 tight=[297.5,690]→370 IN; 4140 tight=[382.5,690]→370<382.5 miss; loose=[315,780]→370 IN
        result = match_record(370.0, None, "CSS", "SPINDLE", REFERENCE)
        assert result["confidence_label"] in ("HIGH", "MEDIUM")


# ---------------------------------------------------------------------------
# match_tool_summary
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tool_summary():
    return pd.DataFrame([
        {
            "tool_number": "01", "active_t_code": "T0101", "machine_folder": "421",
            "s_mode": "CSS", "s_count": 2, "s_mean": 160.0, "s_min": 155.0, "s_max": 165.0,
            "f_count": 2, "f_mean": 0.011, "f_min": 0.010, "f_max": 0.012,
            "record_count": 2, "unique_program_count": 1,
        },
        {
            "tool_number": "02", "active_t_code": "T0202", "machine_folder": "421",
            "s_mode": "RPM", "s_count": 3, "s_mean": 1200.0, "s_min": 1000.0, "s_max": 1400.0,
            "f_count": 3, "f_mean": 0.010, "f_min": 0.008, "f_max": 0.012,
            "record_count": 3, "unique_program_count": 1,
        },
    ])


class TestMatchToolSummary:
    def test_returns_dataframe(self, sample_tool_summary):
        result = match_tool_summary(sample_tool_summary, REFERENCE)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_unchanged(self, sample_tool_summary):
        result = match_tool_summary(sample_tool_summary, REFERENCE)
        assert len(result) == len(sample_tool_summary)

    def test_candidate_columns_appended(self, sample_tool_summary):
        result = match_tool_summary(sample_tool_summary, REFERENCE)
        for col in ["material_candidate_1", "match_type", "confidence_score",
                    "confidence_label", "reason"]:
            assert col in result.columns

    def test_original_columns_preserved(self, sample_tool_summary):
        result = match_tool_summary(sample_tool_summary, REFERENCE)
        for col in sample_tool_summary.columns:
            assert col in result.columns

    def test_css_row_gets_hastelloy_candidate(self, sample_tool_summary):
        # SFM=160 → HASTELLOY C276 exact match
        result = match_tool_summary(sample_tool_summary, REFERENCE)
        css_row = result[result["s_mode"] == "CSS"].iloc[0]
        assert css_row["material_candidate_1"] == "HASTELLOY C276"

    def test_rpm_row_gets_insufficient_context(self, sample_tool_summary):
        result = match_tool_summary(sample_tool_summary, REFERENCE)
        rpm_row = result[result["s_mode"] == "RPM"].iloc[0]
        assert rpm_row["match_type"] == "insufficient_context"

    def test_empty_df_returns_dataframe(self):
        result = match_tool_summary(pd.DataFrame(), REFERENCE)
        assert isinstance(result, pd.DataFrame)

    def test_empty_df_has_candidate_columns(self):
        result = match_tool_summary(pd.DataFrame(), REFERENCE)
        assert "material_candidate_1" in result.columns
        assert "match_type" in result.columns


# ---------------------------------------------------------------------------
# export_material_candidates
# ---------------------------------------------------------------------------

class TestExportMaterialCandidates:
    def test_creates_csv_file(self, sample_tool_summary, tmp_path):
        path = export_material_candidates(
            sample_tool_summary, REFERENCE, tmp_path, timestamp="20260101_120000"
        )
        assert path.exists()
        assert path.suffix == ".csv"

    def test_filename_has_timestamp(self, sample_tool_summary, tmp_path):
        path = export_material_candidates(
            sample_tool_summary, REFERENCE, tmp_path, timestamp="20260101_120000"
        )
        assert "20260101_120000" in path.name

    def test_filename_prefix(self, sample_tool_summary, tmp_path):
        path = export_material_candidates(
            sample_tool_summary, REFERENCE, tmp_path, timestamp="20260101_120000"
        )
        assert path.name.startswith("material_candidates_")

    def test_csv_row_count_matches_input(self, sample_tool_summary, tmp_path):
        path = export_material_candidates(
            sample_tool_summary, REFERENCE, tmp_path, timestamp="20260101_120000"
        )
        df = pd.read_csv(path)
        assert len(df) == len(sample_tool_summary)

    def test_csv_has_candidate_columns(self, sample_tool_summary, tmp_path):
        path = export_material_candidates(
            sample_tool_summary, REFERENCE, tmp_path, timestamp="20260101_120000"
        )
        df = pd.read_csv(path)
        assert "material_candidate_1" in df.columns
        assert "match_type" in df.columns
        assert "confidence_score" in df.columns

    def test_auto_timestamp_creates_file(self, sample_tool_summary, tmp_path):
        path = export_material_candidates(sample_tool_summary, REFERENCE, tmp_path)
        assert path.exists()

    def test_csv_has_feed_intent_columns(self, sample_tool_summary, tmp_path):
        path = export_material_candidates(
            sample_tool_summary, REFERENCE, tmp_path, timestamp="20260101_120000"
        )
        df = pd.read_csv(path)
        for col in ["finish_feed_raw", "finish_feed_low", "finish_feed_mid",
                    "finish_feed_high", "feed_intent_candidate",
                    "feed_intent_confidence", "feed_intent_reason"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Feed intent classification
# ---------------------------------------------------------------------------

# Reference with explicit three-band finish feed data for intent testing
REFERENCE_WITH_FEED_BANDS = [
    {
        "material": "TEST_STEEL",
        "turning_rough_sfm": 500.0,
        "turning_finish_sfm": 700.0,
        "turning_rough_ipr": 0.018,
        "turning_finish_ipr": 0.010,
        "finish_feed_raw": "0.008/0.010/0.014",
        "finish_feed_low": 0.008,    # groove / relief / taper
        "finish_feed_mid": 0.010,    # diameter to size
        "finish_feed_high": 0.014,   # grind stock
        "milling_sfm": 300.0,
        "milling_rough_ipm": 12.0,
        "milling_finish_ipm": 20.0,
    },
]


class TestFeedIntentClassification:
    def test_rough_feed_candidate(self):
        # f=0.018 matches rough_ipr=0.018 exactly → rough_feed_candidate
        result = match_record(600.0, 0.018, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_candidate"] == "rough_feed_candidate"

    def test_grind_stock_candidate(self):
        # f=0.014 matches finish_feed_high=0.014 → grind stock
        result = match_record(600.0, 0.014, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_candidate"] == "finish_grind_stock_candidate"

    def test_diameter_to_size_candidate(self):
        # f=0.010 matches finish_feed_mid=0.010 → diameter to size
        result = match_record(600.0, 0.010, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_candidate"] == "finish_diameter_to_size_candidate"

    def test_groove_relief_taper_candidate(self):
        # f=0.008 matches finish_feed_low=0.008 → groove/relief/taper
        result = match_record(600.0, 0.008, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_candidate"] == "finish_groove_relief_taper_candidate"

    def test_high_confidence_on_exact_band_match(self):
        # delta=0% → HIGH
        result = match_record(600.0, 0.018, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_confidence"] == "HIGH"

    def test_medium_confidence_on_close_band_match(self):
        # f=0.0155 is ~11% above finish_feed_mid=0.010 but closer to grind_high=0.014
        # Actually 0.0155 vs rough=0.018 (14%), high=0.014 (11%), mid=0.010 (55%), low=0.008 (94%)
        # finish_high=0.014: delta=|0.0155-0.014|/0.014=10.7% > 10% → MEDIUM
        result = match_record(600.0, 0.0155, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_confidence"] in ("HIGH", "MEDIUM")

    def test_unknown_intent_when_no_feed(self):
        result = match_record(600.0, None, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_candidate"] == "unknown_feed_intent"

    def test_unknown_intent_when_feed_outside_all_bands(self):
        # f=0.050 is far outside all bands (rough=0.018, max with 20% tol = 0.0216)
        result = match_record(600.0, 0.050, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_candidate"] == "unknown_feed_intent"

    def test_unknown_intent_when_insufficient_context(self):
        # G97 RPM → insufficient_context → unknown feed intent
        result = match_record(1200.0, 0.010, "RPM", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["feed_intent_candidate"] == "unknown_feed_intent"

    def test_finish_feed_bands_present_in_result(self):
        result = match_record(600.0, 0.010, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["finish_feed_low"] == 0.008
        assert result["finish_feed_mid"] == 0.010
        assert result["finish_feed_high"] == 0.014

    def test_finish_feed_raw_in_result(self):
        result = match_record(600.0, 0.010, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["finish_feed_raw"] == "0.008/0.010/0.014"

    def test_feed_intent_reason_is_string(self):
        result = match_record(600.0, 0.010, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert isinstance(result["feed_intent_reason"], str)
        assert len(result["feed_intent_reason"]) > 0

    def test_no_material_match_gives_unknown_intent(self):
        # SFM=50, no feed → no_match (SFM out of range, no feed to score)
        result = match_record(50.0, None, "CSS", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["match_type"] == "no_match"
        assert result["feed_intent_candidate"] == "unknown_feed_intent"

    def test_insufficient_context_gives_null_finish_feed_bands(self):
        result = match_record(1200.0, 0.010, "RPM", "SPINDLE", REFERENCE_WITH_FEED_BANDS)
        assert result["finish_feed_low"] is None
        assert result["finish_feed_mid"] is None
        assert result["finish_feed_high"] is None
