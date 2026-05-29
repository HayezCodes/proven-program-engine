"""
tool_classifier.py — Deterministic tool_type classification.

Classifies a CNC tool into one of 12 categories using keyword rules
applied to the resolved tool name, feed intent, and spindle mode.
No AI — pure string matching in priority order.

Categories:
    turning_rough       Roughing turning inserts (CNMG, DNMG, WNMG, …)
    turning_finish      Finishing turning inserts (VNMG, DCMT finish grade, …)
    groove_relief       Grooving / relief inserts (N123, N151 series)
    threading           Threading inserts and taps (16ER, taps, NPT, …)
    drilling            Drills, spot drills, reamers, centre drills
    boring              Boring bars, ID turning bars, counterbores
    keyway              Woodruff / keyseat cutters, saw cutters
    plunge              Plunge / parting / cutoff tools
    milling_profile     End mills, face mills, chamfer mills
    milling_slot        Slotting saws, slot cutters
    milling_drilling    Drilling operation on a milling machine (RPM mode + drill)
    unknown             Probe, unknown tool, or no recognisable pattern
"""

import re

__all__ = ["classify_tool_type", "TOOL_TYPES"]

TOOL_TYPES: tuple[str, ...] = (
    "turning_rough",
    "turning_finish",
    "groove_relief",
    "threading",
    "drilling",
    "boring",
    "keyway",
    "plunge",
    "milling_profile",
    "milling_slot",
    "milling_drilling",
    "unknown",
)

# Pre-compiled regex for thread-insert codes like 16ER, 16IR, 22ER, 22IR
_THREAD_INSERT_RE = re.compile(r"\b\d{2}[EI]R\b")


def classify_tool_type(
    tool_name: str,
    feed_intent: str = "",
    s_mode: str = "",
) -> str:
    """Return the tool_type string for one tool.

    Priority: explicit unknowns → threading → grooving → plunge → keyway →
    boring → drilling → milling_slot → milling_profile → turning_finish →
    turning_rough → feed_intent fallback → unknown.
    """
    name   = str(tool_name   or "").upper().strip()
    intent = str(feed_intent or "").lower().strip()
    smode  = str(s_mode      or "").upper().strip()

    # ── 1. Non-cutting / unknown ──────────────────────────────────────────
    if not name or name in ("UNKNOWN TOOL", "UNKNOWN", "1"):
        return "unknown"
    if "PROBE" in name:
        return "unknown"

    # ── 2. Threading (check before grooving) ─────────────────────────────
    # Sandvik CoroThread holders
    if "TLR-" in name or "TLG-" in name:
        return "threading"
    # Generic thread-insert code  e.g. "16ER 12 UN", "16IR 14 UN ID THREADING"
    if _THREAD_INSERT_RE.search(name):
        return "threading"
    # Thread designation / tap keywords
    _THREAD_KW = (
        " THREAD", "THREADING",
        " UN ",                     # thread pitch " 12 UN " / " 14 UN "
        " UNF", " UNC", " NPT",
        "RIGHT HAND THREAD", "LEFT HAND THREAD",
        " TAP",                     # RH TAP, NPT TAP, etc.
        "RIGHT-HANDED TAP", "LEFT-HANDED TAP",
        "1.5MM PITCH", " PITCH",
    )
    if any(kw in name for kw in _THREAD_KW):
        return "threading"

    # ── 3. Grooving / relief ──────────────────────────────────────────────
    # Sandvik CoroGroove N123 / N151 series
    if any(kw in name for kw in ("N123", "N151", "N132", "GROOVE", "GROOVING")):
        return "groove_relief"

    # ── 4. Plunge / parting / cutoff ──────────────────────────────────────
    if any(kw in name for kw in ("PLUNGE", "PARTING", "CUTOFF", "CUT-OFF", "CUT OFF")):
        return "plunge"

    # ── 5. Keyway / Woodruff / keyseat saw ───────────────────────────────
    # KSC = keyseat cutter; SAW CUTTER = keyseat saw on a lathe
    if any(kw in name for kw in ("WOODRUFF", "KEYWAY", "KEY WAY", "KSC", "SAW CUTTER")):
        return "keyway"

    # ── 6. Boring / ID turning bars ───────────────────────────────────────
    _BORE_KW = (
        "BORING BAR", "BORING ",
        "COUNTER BORE", "COUNTERBORE",
        "SANDVIK MB-",              # Sandvik MicroBoring
        "CDHH",                     # Sandvik small boring toolholder
        "ACCUPRO",                  # Accupro boring bar brand
        "FINISH CENTER HOLE",       # ID-centering / bore finish op
    )
    if any(kw in name for kw in _BORE_KW):
        return "boring"
    # DCMT or similar insert in an explicit bar → boring
    if ("BAR" in name or "E12Q" in name) and any(
        ins in name for ins in ("DCMT", "CDHH", "SDX")
    ):
        return "boring"

    # ── 7. Drilling ───────────────────────────────────────────────────────
    _DRILL_KW = (
        "DRILL",                    # catches centre drill, bell center drill, spot drill, jobber drill, …
        " REAMER", "REAM",
        "BELL CENTER",
        "CORODRILL", "880 SERIES",  # Sandvik CoroDrill 880
        "INSERT DRILL",
        "JOBBER DRILL",
        "SPADE DRILL",
    )
    if any(kw in name for kw in _DRILL_KW):
        # Distinguish milling_drilling when spindle is in RPM mode (mill)
        if smode == "RPM":
            return "milling_drilling"
        return "drilling"

    # ── 8. Milling — slot / slitting saw ─────────────────────────────────
    if any(kw in name for kw in ("SLOTTING", "SLITTING SAW", "SLOT MILL")):
        return "milling_slot"

    # ── 9. Milling — profile / general ───────────────────────────────────
    _MILL_KW = (
        "ENDMILL", "END MILL", "END-MILL",  # all end-mill variants
        "FLAT END",                          # "FLAT ENDMILL" / "Flat Endmill"
        "BALL END", "BALL NOSE",
        "FACE MILL", "SHELL MILL",
        "CHAMFER MILL", "CHAMFER",           # chamfer mills
        "FINISHER",                          # small OD finisher endmills
        "CRB EM",                            # carbide endmill abbreviation
        "STEPDEX",                           # face/step mill
        "ROUGHING EM", "ROUGHING END",
        "FINSIH END",                        # deliberate misspelling in source data
        "FINSIH EM",
    )
    if any(kw in name for kw in _MILL_KW):
        return "milling_profile"

    # ── 10. Turning finish inserts ────────────────────────────────────────
    # V-shape (35°) inserts are finish geometry
    _FINISH_INS = ("VNMG", "VCMT", "VCGX", "TCGT", "VBMT")
    if any(ins in name for ins in _FINISH_INS):
        return "turning_finish"
    # DCMT without a bar context — check for finish grade suffix
    if "DCMT" in name:
        if any(g in name for g in ("-MF", "-PF", "-PH")):
            return "turning_finish"
        return "boring"   # DCMT in any other context defaults to boring bar insert

    # ── 11. Turning rough inserts ─────────────────────────────────────────
    _ROUGH_INS = ("DNMG", "CNMG", "WNMG", "SPMR", "SNMG")
    if any(ins in name for ins in _ROUGH_INS):
        return "turning_rough"

    # ── 12. Feed-intent fallback (when insert code is unrecognised) ───────
    if intent == "rough_feed_candidate":
        return "turning_rough"
    if intent == "finish_diameter_to_size_candidate":
        return "turning_finish"

    return "unknown"
