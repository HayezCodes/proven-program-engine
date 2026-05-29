"""
test_tool_classifier.py — Tests for deterministic tool_type classification.
"""

import pytest
from src.tool_classifier import classify_tool_type, TOOL_TYPES


# ---------------------------------------------------------------------------
# Sanity: return values are always in the defined set
# ---------------------------------------------------------------------------

class TestReturnValueConstraint:
    def test_all_known_tools_return_valid_type(self):
        names = [
            "VNMG 332-PF", "DNMG 443-PR", "3/8\" FLAT ENDMILL",
            "16ER 12 UN", "N123F2-0267-0002-GF", "#17 Bell Center Drill",
            "#605 WOODRUFF CUTTER", ".180 DIA BORING BAR", "UNKNOWN TOOL", "",
        ]
        for name in names:
            result = classify_tool_type(name)
            assert result in TOOL_TYPES, f"{name!r} → {result!r} not in TOOL_TYPES"

    def test_empty_name_returns_unknown(self):
        assert classify_tool_type("") == "unknown"

    def test_none_coerced_to_unknown(self):
        assert classify_tool_type(None) == "unknown"


# ---------------------------------------------------------------------------
# Turning inserts
# ---------------------------------------------------------------------------

class TestTurningInserts:
    def test_vnmg_pf_is_finish(self):
        assert classify_tool_type("VNMG 332-PF") == "turning_finish"

    def test_vnmg_mf_is_finish(self):
        assert classify_tool_type("VNMG 331-MF") == "turning_finish"

    def test_vnmg_reverse_holder_is_finish(self):
        assert classify_tool_type("VNMG 331-PF REVERSE HOLDER") == "turning_finish"

    def test_dnmg_pr_is_rough(self):
        assert classify_tool_type("DNMG 443-PR") == "turning_rough"

    def test_dnmg_mrr_is_rough(self):
        assert classify_tool_type("DNMG 443-MRR") == "turning_rough"

    def test_dnmg_mmr_is_rough(self):
        assert classify_tool_type("DNMG 443 MMR 2220") == "turning_rough"

    def test_dnmg_reverse_is_rough(self):
        assert classify_tool_type("DNMG 443-PR REVERSE HOLDER") == "turning_rough"

    def test_wnmg_is_rough(self):
        assert classify_tool_type("WNMG 331") == "turning_rough"

    def test_dnmg_432fw_is_rough(self):
        assert classify_tool_type("DNMG432FW") == "turning_rough"


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

class TestThreading:
    def test_16er_un_is_threading(self):
        assert classify_tool_type("16ER 12 UN") == "threading"

    def test_16ir_id_threading(self):
        assert classify_tool_type("16IR 14 UN ID THREADING") == "threading"

    def test_16er_13_un(self):
        assert classify_tool_type("16ER 13 UN") == "threading"

    def test_rh_tap_is_threading(self):
        assert classify_tool_type("3/4-10 RH TAP") == "threading"

    def test_npt_tap_is_threading(self):
        assert classify_tool_type('1/4"-18 NPT TAP') == "threading"

    def test_unf_tap_is_threading(self):
        assert classify_tool_type("0.625 5/8-18 UNF Left-Handed Tap") == "threading"

    def test_unc_tap_is_threading(self):
        assert classify_tool_type("1/4-20 UNC RIGHT-HANDED TAP") == "threading"

    def test_right_hand_thread_is_threading(self):
        assert classify_tool_type("4.325-12 RIGHT HAND THREAD") == "threading"

    def test_metric_pitch_is_threading(self):
        assert classify_tool_type("METRIC 1.5MM PITCH") == "threading"

    def test_tlr_holder_is_threading(self):
        assert classify_tool_type("TLR-3031R") == "threading"

    def test_tlg_id_is_threading(self):
        assert classify_tool_type("TLG-3047L ID") == "threading"


# ---------------------------------------------------------------------------
# Grooving
# ---------------------------------------------------------------------------

class TestGrooving:
    def test_n123f2_is_groove(self):
        assert classify_tool_type("N123F2-0267-0002-GF") == "groove_relief"

    def test_n123t3_is_groove(self):
        assert classify_tool_type("N123T3-0080-RS") == "groove_relief"

    def test_n123h2_is_groove(self):
        assert classify_tool_type("N123H2-0475-0008-GF") == "groove_relief"


# ---------------------------------------------------------------------------
# Drilling
# ---------------------------------------------------------------------------

class TestDrilling:
    def test_bell_center_drill(self):
        assert classify_tool_type("#17 Bell Center Drill") == "drilling"

    def test_spot_drill(self):
        assert classify_tool_type("3/8 X 90 DEG. SPOT DRILL") == "drilling"

    def test_dot_09_spot_drill(self):
        assert classify_tool_type(".09 SPOT DRILL") == "drilling"

    def test_jobber_drill(self):
        assert classify_tool_type("0.578125 37/64 Jobber Drill") == "drilling"

    def test_plain_drill(self):
        assert classify_tool_type("5/16\" DRILL") == "drilling"

    def test_reamer(self):
        assert classify_tool_type("LTR.D REAMER") == "drilling"

    def test_corodrill(self):
        assert classify_tool_type("CoroDrill\\X\\AE 880 indexable insert drill") == "drilling"

    def test_880_series_corodrill(self):
        assert classify_tool_type("1.562 880 SERIES CORODRILL") == "drilling"

    def test_insert_drill(self):
        assert classify_tool_type("2.0 Insert Drill") == "drilling"

    def test_spade_drill(self):
        assert classify_tool_type("17/32 SPADE DRILL") == "drilling"

    def test_milling_drilling_when_rpm(self):
        assert classify_tool_type("#17 Bell Center Drill", s_mode="RPM") == "milling_drilling"

    def test_drilling_when_css(self):
        assert classify_tool_type("#17 Bell Center Drill", s_mode="CSS") == "drilling"


# ---------------------------------------------------------------------------
# Boring
# ---------------------------------------------------------------------------

class TestBoring:
    def test_boring_bar(self):
        assert classify_tool_type(".180 DIA BORING BAR") == "boring"

    def test_counter_bore(self):
        assert classify_tool_type('1.5" COUNTER BORE') == "boring"

    def test_dcmt_with_bar(self):
        assert classify_tool_type('DCMT 3|2.5|1-MF/.75" BAR') == "boring"

    def test_dcmt_bar_variant(self):
        assert classify_tool_type("DCMT 3 2.5 1-MF / .750 BAR") == "boring"

    def test_e12q_boring_bar(self):
        assert classify_tool_type("E12Q SDX CR3 REVERSED DCMT 3 2.5 1") == "boring"

    def test_sandvik_mb_is_boring(self):
        assert classify_tool_type("SANDVIK MB-E16-75-09R BAR ,MB-029G200-02-19R INSERT") == "boring"

    def test_cdhh_boring_holder(self):
        assert classify_tool_type("CDHH120605 / .250 BAR") == "boring"

    def test_accupro_is_boring(self):
        assert classify_tool_type("ACCUPRO 62908017 - .250 BAR") == "boring"

    def test_finish_center_hole_is_boring(self):
        assert classify_tool_type("FINISH CENTER HOLE") == "boring"


# ---------------------------------------------------------------------------
# Keyway / Woodruff
# ---------------------------------------------------------------------------

class TestKeyway:
    def test_woodruff_cutter(self):
        assert classify_tool_type("#605 WOODRUFF CUTTER") == "keyway"

    def test_woodruff_ksc(self):
        assert classify_tool_type("#610 WOODRUFF KSC") == "keyway"

    def test_405_ksc(self):
        assert classify_tool_type("#405 WOODRUFF KSC") == "keyway"

    def test_saw_cutter(self):
        assert classify_tool_type("3/32 SAW CUTTER") == "keyway"


# ---------------------------------------------------------------------------
# Milling — profile
# ---------------------------------------------------------------------------

class TestMillingProfile:
    def test_flat_endmill(self):
        assert classify_tool_type('3/8" FLAT ENDMILL') == "milling_profile"

    def test_flat_endmill_metric(self):
        assert classify_tool_type("6MM FLAT ENDMILL") == "milling_profile"

    def test_radius_endmill(self):
        assert classify_tool_type('1/2" x .060R ENDMILL') == "milling_profile"

    def test_chamfer_mill(self):
        assert classify_tool_type("0.25 Chamfer Mill") == "milling_profile"

    def test_face_mill(self):
        assert classify_tool_type("3 Face Mill / Shell Mill") == "milling_profile"

    def test_finisher_endmill(self):
        assert classify_tool_type("3MM OR 4MM FINISHER") == "milling_profile"

    def test_6mm_finisher(self):
        assert classify_tool_type("6MM FINISHER") == "milling_profile"

    def test_crb_em(self):
        assert classify_tool_type("3/32 CRB EM") == "milling_profile"

    def test_roughing_endmill(self):
        assert classify_tool_type("0.5 Flat roughing Endmill") == "milling_profile"

    def test_end_mill_variant(self):
        assert classify_tool_type("4MM END-MILL") == "milling_profile"

    def test_5flute_finish_endmill(self):
        assert classify_tool_type("0.5 5 FLUTE FINSIH ENDMILL") == "milling_profile"

    def test_stepdex_face_mill(self):
        assert classify_tool_type('3" STEPDEX FACE MILL') == "milling_profile"


# ---------------------------------------------------------------------------
# Feed-intent fallback
# ---------------------------------------------------------------------------

class TestFeedIntentFallback:
    def test_rough_intent_unrecognised_tool(self):
        assert classify_tool_type("MYSTERY INSERT", feed_intent="rough_feed_candidate") == "turning_rough"

    def test_finish_intent_unrecognised_tool(self):
        assert classify_tool_type(
            "MYSTERY INSERT", feed_intent="finish_diameter_to_size_candidate"
        ) == "turning_finish"

    def test_unknown_intent_still_unknown(self):
        assert classify_tool_type("MYSTERY INSERT", feed_intent="unknown_feed_intent") == "unknown"


# ---------------------------------------------------------------------------
# Unknown / non-cutting
# ---------------------------------------------------------------------------

class TestUnknown:
    def test_unknown_tool_label(self):
        assert classify_tool_type("UNKNOWN TOOL") == "unknown"

    def test_tool_probe(self):
        assert classify_tool_type("TOOL PROBE") == "unknown"

    def test_single_digit_name(self):
        assert classify_tool_type("1") == "unknown"
