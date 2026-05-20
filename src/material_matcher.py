"""
material_matcher.py — Match cut records against the shop S/F reference table.

All matching is deterministic and based on numeric tolerance bands.
No inference, no ML. Material candidates are suggestions only — never written as fact.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

# SFM tolerance bands against [rough_sfm, finish_sfm]
_SFM_TIGHT = 0.15   # sfm_score = 2 when s_value within ±15% of range
_SFM_LOOSE = 0.30   # sfm_score = 1 when s_value within ±30% of range

# Feed scoring tolerance (material-matching pass)
_FEED_TOL = 0.20    # feed_score = 1 when f_value within ±20% of reference range

# Feed intent tolerance (intent classification pass)
_INTENT_TOL = 0.20  # within ±20% of a reference band → candidate; ≤10% → HIGH confidence

_MAX_SCORE = 3  # sfm_tight(2) + feed_match(1)

_CANDIDATE_COLS = [
    "material_candidate_1",
    "material_candidate_2",
    "material_candidate_3",
    "confidence_score",
    "confidence_label",
    "match_type",
    "reason",
    "finish_feed_raw",
    "finish_feed_low",
    "finish_feed_mid",
    "finish_feed_high",
    "feed_intent_candidate",
    "feed_intent_confidence",
    "feed_intent_reason",
]


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _score_one_material(
    s_mean: float | None,
    f_mean: float | None,
    s_mode: str,
    mat: dict,
) -> tuple[int, int, int]:
    """Score one material against the given S/F values.

    Returns (total_score, sfm_score, feed_score).
    sfm_score: 2=tight match, 1=loose match, 0=no match / not applicable.
    feed_score: 1=match, 0=no match / not applicable.
    """
    sfm_score = 0
    feed_score = 0

    if s_mode == "CSS" and s_mean is not None:
        rough = mat.get("turning_rough_sfm")
        finish = mat.get("turning_finish_sfm")
        if rough is not None and finish is not None:
            lo_tight = rough * (1 - _SFM_TIGHT)
            hi_tight = finish * (1 + _SFM_TIGHT)
            lo_loose = rough * (1 - _SFM_LOOSE)
            hi_loose = finish * (1 + _SFM_LOOSE)

            if lo_tight <= s_mean <= hi_tight:
                sfm_score = 2
            elif lo_loose <= s_mean <= hi_loose:
                sfm_score = 1

    if f_mean is not None:
        if f_mean < 1.0:
            # Small feed value — compare against turning IPR ranges
            r_ipr = mat.get("turning_rough_ipr")
            f_ipr = mat.get("turning_finish_ipr")
            if r_ipr is not None and f_ipr is not None:
                lo = min(r_ipr, f_ipr) * (1 - _FEED_TOL)
                hi = max(r_ipr, f_ipr) * (1 + _FEED_TOL)
                if lo <= f_mean <= hi:
                    feed_score = 1
        elif f_mean >= 2.0:
            # Large feed value — compare against milling IPM ranges
            r_ipm = mat.get("milling_rough_ipm")
            f_ipm = mat.get("milling_finish_ipm")
            if r_ipm is not None and f_ipm is not None:
                lo = min(r_ipm, f_ipm) * (1 - _FEED_TOL)
                hi = max(r_ipm, f_ipm) * (1 + _FEED_TOL)
                if lo <= f_mean <= hi:
                    feed_score = 1

    return sfm_score + feed_score, sfm_score, feed_score


# ---------------------------------------------------------------------------
# Feed intent classification
# ---------------------------------------------------------------------------

def _no_feed_intent(reason_text: str) -> dict:
    """Return an unknown-intent result with null finish feed bands."""
    return {
        "finish_feed_raw": None,
        "finish_feed_low": None,
        "finish_feed_mid": None,
        "finish_feed_high": None,
        "feed_intent_candidate": "unknown_feed_intent",
        "feed_intent_confidence": "NONE",
        "feed_intent_reason": reason_text,
    }


def _classify_feed_intent(f_mean: float | None, mat: dict) -> dict:
    """Classify feed intent against a material's reference feed bands.

    Compares f_mean to rough_ipr, finish_feed_high, finish_feed_mid, finish_feed_low using
    ±20% tolerance. The closest match within tolerance wins.
    Returns a dict with finish_feed_raw/low/mid/high and feed_intent_* fields.
    """
    finish_context = {
        "finish_feed_raw": mat.get("finish_feed_raw"),
        "finish_feed_low": mat.get("finish_feed_low"),
        "finish_feed_mid": mat.get("finish_feed_mid"),
        "finish_feed_high": mat.get("finish_feed_high"),
    }

    if f_mean is None or f_mean >= 1.0:
        return {
            **finish_context,
            "feed_intent_candidate": "unknown_feed_intent",
            "feed_intent_confidence": "NONE",
            "feed_intent_reason": "no IPR feed value for intent classification",
        }

    # Build candidate bands ordered by priority label
    bands: list[tuple[str, float]] = []
    rough = mat.get("turning_rough_ipr")
    if rough is not None:
        bands.append(("rough_feed_candidate", rough))
    f_high = mat.get("finish_feed_high")
    if f_high is not None:
        bands.append(("finish_grind_stock_candidate", f_high))
    f_mid = mat.get("finish_feed_mid")
    if f_mid is not None:
        bands.append(("finish_diameter_to_size_candidate", f_mid))
    f_low = mat.get("finish_feed_low")
    if f_low is not None:
        bands.append(("finish_groove_relief_taper_candidate", f_low))

    if not bands:
        return {
            **finish_context,
            "feed_intent_candidate": "unknown_feed_intent",
            "feed_intent_confidence": "NONE",
            "feed_intent_reason": (
                f"feed={f_mean:.4f} — no reference bands for {mat.get('material', '?')}"
            ),
        }

    # Find all bands within tolerance, pick the closest (smallest relative delta)
    in_range = []
    for intent, ref in bands:
        delta_pct = abs(f_mean - ref) / ref
        if delta_pct <= _INTENT_TOL:
            in_range.append((intent, ref, delta_pct))

    if not in_range:
        return {
            **finish_context,
            "feed_intent_candidate": "unknown_feed_intent",
            "feed_intent_confidence": "NONE",
            "feed_intent_reason": (
                f"feed={f_mean:.4f} outside all reference bands for {mat.get('material', '?')}"
            ),
        }

    best_intent, best_ref, best_delta = min(in_range, key=lambda x: x[2])
    confidence = "HIGH" if best_delta <= 0.10 else "MEDIUM"
    label = best_intent.replace("_candidate", "").replace("_", " ")
    reason = (
        f"feed={f_mean:.4f} closest to {label} "
        f"ref={best_ref:.4f} ({best_delta * 100:.1f}% delta)"
    )

    return {
        **finish_context,
        "feed_intent_candidate": best_intent,
        "feed_intent_confidence": confidence,
        "feed_intent_reason": reason,
    }


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def _insufficient(reason_text: str) -> dict:
    result = {
        "material_candidate_1": None,
        "material_candidate_2": None,
        "material_candidate_3": None,
        "confidence_score": 0.0,
        "confidence_label": "NONE",
        "match_type": "insufficient_context",
        "reason": reason_text,
    }
    result.update(_no_feed_intent("insufficient context for material matching"))
    return result


def match_record(
    s_mean: float | None,
    f_mean: float | None,
    s_mode: str,
    s_type: str,
    reference: list[dict],
) -> dict:
    """Return material candidate fields for a single S/F observation.

    s_mean: spindle speed value (SFM if CSS, RPM if G97)
    f_mean: feed value (IPR for turning, IPM for milling)
    s_mode: CSS / RPM / UNKNOWN
    s_type: SPINDLE / LIMIT / ""
    reference: list of material dicts from load_reference()
    """
    if s_type == "LIMIT":
        return _insufficient("spindle speed limit record — not a cutting condition")

    # G97 RPM turning: cannot compare RPM to SFM reference without workpiece diameter
    if s_mode == "RPM" and (f_mean is None or f_mean < 1.0):
        return _insufficient(
            "G97 RPM turning — cannot compare RPM to SFM reference without workpiece diameter"
        )

    if s_mean is None and f_mean is None:
        return _insufficient("no S or F value available")

    if s_mode == "UNKNOWN" and f_mean is None:
        return _insufficient("unknown spindle mode and no feed value")

    # Score every material
    scored = []
    for mat in reference:
        total, sfm_sc, feed_sc = _score_one_material(s_mean, f_mean, s_mode, mat)
        scored.append((mat["material"], total, sfm_sc, feed_sc))

    scored.sort(key=lambda x: -x[1])

    best_score = scored[0][1] if scored else 0

    if best_score == 0:
        vals = _format_vals(s_mean, s_mode, f_mean)
        result = {
            "material_candidate_1": None,
            "material_candidate_2": None,
            "material_candidate_3": None,
            "confidence_score": 0.0,
            "confidence_label": "NONE",
            "match_type": "no_match",
            "reason": f"no material SFM/feed range matched ({vals})",
        }
        result.update(_no_feed_intent("no material match — cannot classify feed intent"))
        return result

    top_at_best = [m for m, sc, _, _ in scored if sc == best_score]
    candidates = [m for m, _, _, _ in scored[:3]]

    if best_score >= 2 and len(top_at_best) == 1:
        match_type = "exact_match"
    elif best_score >= 2:
        match_type = "multiple_possible_matches"
    else:
        match_type = "close_match"

    confidence_score = round(best_score / _MAX_SCORE, 3)

    if confidence_score >= 0.667:
        confidence_label = "HIGH"
    elif confidence_score >= 0.333:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    top_names = top_at_best[0] if len(top_at_best) == 1 else "/".join(top_at_best[:3])
    vals = _format_vals(s_mean, s_mode, f_mean)
    reason = f"{match_type} for {top_names} ({vals})"

    result = {
        "material_candidate_1": candidates[0] if len(candidates) > 0 else None,
        "material_candidate_2": candidates[1] if len(candidates) > 1 else None,
        "material_candidate_3": candidates[2] if len(candidates) > 2 else None,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "match_type": match_type,
        "reason": reason,
    }

    # Feed intent: classify using candidate_1's finish feed bands
    cand1 = result["material_candidate_1"]
    mat1 = next((m for m in reference if m["material"] == cand1), None) if cand1 else None
    if mat1:
        result.update(_classify_feed_intent(f_mean, mat1))
    else:
        result.update(_no_feed_intent("no material candidate — cannot classify feed intent"))

    return result


def _format_vals(s_mean, s_mode, f_mean) -> str:
    parts = []
    if s_mean is not None:
        label = "SFM" if s_mode == "CSS" else f"S({s_mode})"
        parts.append(f"{label}={s_mean:.0f}")
    if f_mean is not None:
        parts.append(f"feed={f_mean:.4f}")
    return ", ".join(parts) if parts else "no values"


def match_tool_summary(
    tool_summary_df: pd.DataFrame,
    reference: list[dict],
) -> pd.DataFrame:
    """Apply material matching to each row of a tool_summary DataFrame.

    Returns a new DataFrame with the original columns plus candidate columns appended.
    """
    if tool_summary_df.empty:
        result = tool_summary_df.copy()
        for col in _CANDIDATE_COLS:
            result[col] = None
        return result

    rows = []
    for _, row in tool_summary_df.iterrows():
        s_mean = _safe_float(row.get("s_mean"))
        f_mean = _safe_float(row.get("f_mean"))
        raw_mode = row.get("s_mode")
        s_mode = "UNKNOWN" if (raw_mode is None or _safe_float(str(raw_mode)) is not None or pd.isna(raw_mode)) else str(raw_mode)
        # s_type not tracked in tool_summary; G92 LIMIT records are rare and mixed into groups
        match = match_record(s_mean, f_mean, s_mode, "SPINDLE", reference)
        rows.append(match)

    candidates_df = pd.DataFrame(rows)
    return pd.concat([tool_summary_df.reset_index(drop=True), candidates_df], axis=1)


def export_material_candidates(
    tool_summary_df: pd.DataFrame,
    reference: list[dict],
    exports_dir: Path,
    timestamp: str | None = None,
) -> Path:
    """Run material matching on tool_summary and write material_candidates_*.csv."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = match_tool_summary(tool_summary_df, reference)
    out_path = exports_dir / f"material_candidates_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    matched = df[df["match_type"].isin(
        ["exact_match", "close_match", "multiple_possible_matches"]
    )]
    logger.info(
        f"Material candidates -> {out_path}  "
        f"({len(df)} rows, {len(matched)} with candidates)"
    )
    return out_path
