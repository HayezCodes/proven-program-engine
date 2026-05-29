"""
Phase 6C — Unmatched Program Analysis

Reads existing exports only. No file scanning. No network access.
Does not modify any existing exports or linker behavior.

Exports:
  exports/unmatched_programs_TIMESTAMP.csv
  exports/link_failure_summary_TIMESTAMP.csv
  exports/link_candidate_examples_TIMESTAMP.csv
"""

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.job_linker import filename_tokens, _normalize_id
from src.safety import assert_safe_write
from src.utils import get_logger

logger = get_logger("run_analyze_unmatched")
EXPORTS_DIR = Path(__file__).parent / "exports"

_SKIP_VALUES = {"nan", "", "job", "description", "n/a", "na", "none", "ber"}


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _latest(pat: str) -> Path | None:
    files = sorted(EXPORTS_DIR.glob(pat), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load(pat: str, label: str) -> pd.DataFrame:
    p = _latest(pat)
    if p is None:
        logger.warning(f"{label}: no file matching {pat}")
        return pd.DataFrame()
    logger.info(f"  {label}: {p.name}")
    return pd.read_csv(p, low_memory=False)


def _s(d: dict, key: str) -> str:
    return str(d.get(key, "") or "")


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_job_indexes(
    job_df: pd.DataFrame,
) -> tuple[dict, dict, dict]:
    """
    Returns job_by_dwg, job_by_part, job_by_job — each maps
    normalized_key → [row_dict, ...].  Skips placeholder values.
    """
    job_by_dwg:  dict[str, list[dict]] = {}
    job_by_part: dict[str, list[dict]] = {}
    job_by_job:  dict[str, list[dict]] = {}

    for _, row in job_df.iterrows():
        r = row.to_dict()
        for field, idx in [
            ("drawing_number", job_by_dwg),
            ("part_number",    job_by_part),
            ("job_number",     job_by_job),
        ]:
            raw = str(r.get(field, "") or "").strip()
            nval = _normalize_id(raw)
            if nval and nval.lower() not in _SKIP_VALUES:
                idx.setdefault(nval, []).append(r)

    return job_by_dwg, job_by_part, job_by_job


def build_print_indexes(
    print_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Returns print_by_dwg, print_by_part."""
    by_dwg:  dict[str, list[dict]] = {}
    by_part: dict[str, list[dict]] = {}

    for _, row in print_df.iterrows():
        r = row.to_dict()
        for field, idx in [("drawing_number", by_dwg), ("part_number", by_part)]:
            raw = str(r.get(field, "") or "").strip()
            nval = _normalize_id(raw)
            if nval and nval.lower() not in _SKIP_VALUES:
                idx.setdefault(nval, []).append(r)

    return by_dwg, by_part


def build_router_word_index(
    router_df: pd.DataFrame,
) -> dict[str, list[tuple[str, str]]]:
    """
    Maps each word (length >= 4) from operation_description to
    a list of (job_number, description_snippet) pairs.
    """
    idx: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if router_df.empty:
        return idx

    for _, row in router_df.iterrows():
        desc = str(row.get("operation_description", "") or "").lower()
        jn   = str(row.get("job_number", "") or "")
        snip = desc[:80]
        for word in re.findall(r"\w+", desc):
            if len(word) >= 4:
                idx[word].append((jn, snip))

    return idx


# ---------------------------------------------------------------------------
# Candidate finder — broader than exact-match linker
# ---------------------------------------------------------------------------

def find_candidates(
    tokens: list[str],
    job_by_dwg: dict, job_by_part: dict, job_by_job: dict,
) -> tuple[str, str, str, str]:
    """
    Returns (cand_job, cand_part, cand_dwg, match_note).

    Tries: exact → token-in-key → key-in-token (substring).
    Exact hits mean the linker should have caught them; flagged in note.
    """
    # Exact (diagnostic — linker uses same normalization)
    for tok in tokens:
        for idx, field in [
            (job_by_dwg,  "drawing"),
            (job_by_part, "part"),
            (job_by_job,  "job"),
        ]:
            rows = idx.get(tok, [])
            if rows:
                r = rows[0]
                return (
                    _s(r, "job_number"), _s(r, "part_number"), _s(r, "drawing_number"),
                    f"exact_{field}='{tok}' [linker_miss?]",
                )

    # Substring: token is substring of DB key
    for tok in tokens:
        if len(tok) < 4:
            continue
        for idx, field in [
            (job_by_dwg,  "drawing"),
            (job_by_part, "part"),
            (job_by_job,  "job"),
        ]:
            for key, rows in idx.items():
                if tok in key:
                    r = rows[0]
                    return (
                        _s(r, "job_number"), _s(r, "part_number"), _s(r, "drawing_number"),
                        f"tok_in_{field}: '{tok}'⊂'{key}'",
                    )

    # Substring: DB key is substring of token
    for tok in tokens:
        if len(tok) < 4:
            continue
        for idx, field in [
            (job_by_dwg,  "drawing"),
            (job_by_part, "part"),
            (job_by_job,  "job"),
        ]:
            for key, rows in idx.items():
                if len(key) >= 4 and key in tok:
                    r = rows[0]
                    return (
                        _s(r, "job_number"), _s(r, "part_number"), _s(r, "drawing_number"),
                        f"key_in_tok: '{key}'⊂'{tok}' [{field}]",
                    )

    return "", "", "", ""


# ---------------------------------------------------------------------------
# Closest router match (informational — does not change linker behaviour)
# ---------------------------------------------------------------------------

def closest_router_match(
    tokens: list[str],
    router_word_idx: dict,
) -> str:
    """
    Returns a diagnostic string if any token (>= 4 chars) appears in a
    router op description — including ambiguous multi-job hits.
    """
    for tok in tokens:
        tok_l = tok.lower()
        if len(tok_l) < 4:
            continue
        hits = router_word_idx.get(tok_l, [])
        if hits:
            jobs = list(dict.fromkeys(jn for jn, _ in hits if jn))
            snip = hits[0][1]
            return (
                f"tok='{tok}' in {len(hits)} op(s) "
                f"jobs={jobs[:4]} "
                f"e.g.: \"{snip}\""
            )
    return ""


# ---------------------------------------------------------------------------
# Closest shared-print match (informational)
# ---------------------------------------------------------------------------

def closest_print_match(
    tokens: list[str],
    print_by_dwg: dict,
    print_by_part: dict,
) -> str:
    for tok in tokens:
        if len(tok) < 3:
            continue
        for idx, field in [(print_by_dwg, "dwg"), (print_by_part, "part")]:
            # Exact
            if tok in idx:
                r = idx[tok][0]
                fn = Path(_s(r, "source_file")).name
                return f"exact {field}='{tok}' file='{fn}'"
            # Substring
            for key in idx:
                if (len(tok) >= 4 and tok in key) or (len(key) >= 4 and key in tok):
                    r = idx[key][0]
                    fn = Path(_s(r, "source_file")).name
                    return f"near {field}: tok='{tok}' key='{key}' file='{fn}'"
    return ""


# ---------------------------------------------------------------------------
# Failure reason classifier
# ---------------------------------------------------------------------------

def classify_failure(
    filename: str,
    tokens: list[str],
    job_by_dwg: dict, job_by_part: dict, job_by_job: dict,
    print_by_dwg: dict, print_by_part: dict,
    router_word_idx: dict,
    cand_note: str,
) -> str:
    stem = Path(filename).stem

    # ── Fanuc O-number subroutine ──────────────────────────────────────────
    if re.match(r"^O\d+$", stem, re.IGNORECASE):
        return "subroutine_O_pattern"

    # ── EM-prefix (internal shop drawing number) ───────────────────────────
    if re.match(r"^EM\d", stem, re.IGNORECASE):
        if cand_note and "⊂" in cand_note:
            return "em_vs_drawing_near_match"
        return "em_drawing_not_in_job_db"

    # ── T-prefix composite (lathe machines 435/436) ────────────────────────
    if re.match(r"^T\d{7,}$", stem, re.IGNORECASE):
        return "t_prefix_composite_part_op"

    # ── Pure numeric ───────────────────────────────────────────────────────
    if re.fullmatch(r"\d+", stem):
        n = len(stem)

        if n >= 7:
            # 8-digit composite: part_number + op_seq
            if cand_note:
                return "composite_part_op_near_match"
            return "composite_part_op_no_match"

        if n <= 4:
            # 4-digit: check whether tokens appear in router (even if ambiguous)
            for tok in tokens:
                tok_l = tok.lower()
                if len(tok_l) >= 4 and router_word_idx.get(tok_l):
                    jobs = list(dict.fromkeys(
                        jn for jn, _ in router_word_idx[tok_l] if jn
                    ))
                    if len(jobs) == 1:
                        return "short_numeric_router_single_job"
                    return "short_numeric_router_ambiguous"
            # Tokens are < 5 chars — Strategy 5 minimum filters them out
            return "short_numeric_below_router_min_length"

        # 5–6 digit
        if cand_note:
            return "medium_numeric_near_match"
        return "medium_numeric_no_match"

    # ── Hyphenated / suffixed (e.g. EM0773-INDEXER, 10667-A1) ─────────────
    if re.search(r"[-_]", stem):
        base = re.split(r"[-_]", stem)[0]
        base_tokens = filename_tokens(base)
        for tok in base_tokens:
            for idx in (job_by_dwg, job_by_part, job_by_job):
                if tok in idx:
                    return "drawing_suffix_mismatch"
        if cand_note:
            return "drawing_suffix_near_match"
        return "alphanumeric_suffix_no_match"

    # ── Near match found somewhere ─────────────────────────────────────────
    if cand_note:
        if "drawing" in cand_note:
            return "drawing_suffix_mismatch"
        if "part" in cand_note:
            return "part_vs_drawing_confusion"
        if "job" in cand_note:
            return "token_normalization_needed"

    return "no_candidate_found"


# ---------------------------------------------------------------------------
# Human-readable rejection explanation
# ---------------------------------------------------------------------------

_REJECTION_TEXT: dict[str, str] = {
    "em_drawing_not_in_job_db": (
        "EM-prefix is an internal shop drawing number (e.g. EM0035). "
        "Job DB uses customer/ERP drawing numbers (e.g. 412292, 705681-038). "
        "Numeric suffix tokens (e.g. '35') are too short to be meaningful matches."
    ),
    "em_vs_drawing_near_match": (
        "EM numeric suffix appears as a substring in a DB entry, but the "
        "match is incidental (short token, different number space)."
    ),
    "composite_part_op_no_match": (
        "8-digit filename encodes part_number + operation_seq "
        "(e.g. 10007001 = part 10007 + op 001). "
        "5-digit prefix tokens (10007, 10033, …) not found in job drawing/part index."
    ),
    "composite_part_op_near_match": (
        "8-digit composite has a substring near-match, but exact normalized "
        "key differs — composite number space partially overlaps DB."
    ),
    "short_numeric_below_router_min_length": (
        "4-digit stem generates tokens < 5 chars (e.g. '0118', '118'). "
        "Strategy 5 (router_match) skips tokens shorter than 5 characters. "
        "Token never tested against router descriptions."
    ),
    "short_numeric_router_ambiguous": (
        "4-digit token appears in router op descriptions but maps to "
        "multiple job numbers — Strategy 5 requires exactly 1 unique job."
    ),
    "short_numeric_router_single_job": (
        "4-digit token maps to exactly 1 router job but tokens are < 5 chars — "
        "filtered out before Strategy 5 runs. Potential fix available."
    ),
    "subroutine_O_pattern": (
        "Fanuc O-number program (e.g. O1562). These are subroutines called by "
        "other programs, not independent job programs. Not expected to link."
    ),
    "t_prefix_composite_part_op": (
        "T-prefixed composite lathe program (e.g. T10851001). Same composite "
        "structure as 8-digit numerics — part+op encoding not in job DB."
    ),
    "drawing_suffix_mismatch": (
        "Filename has a suffix (e.g. -INDEXER, -HEX, -A1) that prevents "
        "exact token match. Stripped base may be in DB."
    ),
    "drawing_suffix_near_match": (
        "Suffix variant with a near-match in DB — suffix removes exact match "
        "but substring match found."
    ),
    "alphanumeric_suffix_no_match": (
        "Hyphenated/suffixed filename; neither full stem nor base found in DB."
    ),
    "medium_numeric_no_match": (
        "5–6 digit numeric stem not found in job_number, drawing_number, "
        "or part_number indexes."
    ),
    "medium_numeric_near_match": (
        "5–6 digit numeric with a substring near-match — exact format differs."
    ),
    "part_vs_drawing_confusion": (
        "Token matches a part_number entry but program naming uses drawing numbers, "
        "or vice versa."
    ),
    "token_normalization_needed": (
        "Token near-matches a job_number key — normalization or prefix handling "
        "difference may be blocking exact match."
    ),
    "no_candidate_found": (
        "No token from the filename (exact or substring, length >= 4) found in "
        "any job, part, drawing, shared-print, or router index."
    ),
}


def _explain(reason: str, row: dict) -> str:
    base = _REJECTION_TEXT.get(reason, f"Unclassified: {reason}")
    note = row.get("candidate_match_note", "")
    if note:
        return f"{base} | Near-match detail: {note}"
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== Phase 6C — Unmatched Program Analysis ===")
    logger.info("Loading exports...")

    links  = _load("program_job_links_20260529_074703.csv", "program_job_links")
    jobs   = _load("job_metadata_*.csv",          "job_metadata")
    prints = _load("shared_print_index_*.csv",    "shared_print_index")
    router = _load("router_operations_*.csv",     "router_operations")

    if links.empty:
        logger.error("No program_job_links found — aborting")
        return 1

    unmatched = links[links["link_method"] == "no_match"].copy()
    linked_df = links[links["link_method"] != "no_match"].copy()
    total = len(links)

    logger.info(
        f"Programs: total={total}  linked={len(linked_df)} ({len(linked_df)/total*100:.1f}%)  "
        f"unmatched={len(unmatched)} ({len(unmatched)/total*100:.1f}%)"
    )

    # Build indexes
    logger.info("Building indexes...")
    job_by_dwg, job_by_part, job_by_job = build_job_indexes(jobs)
    print_by_dwg, print_by_part = build_print_indexes(prints)
    router_word_idx = build_router_word_index(router)

    logger.info(
        f"Job index: dwg={len(job_by_dwg)}  part={len(job_by_part)}  job={len(job_by_job)}"
    )
    logger.info(
        f"Print index: dwg={len(print_by_dwg)}  part={len(print_by_part)}"
    )
    logger.info(f"Router word index: {len(router_word_idx)} unique words")

    # Analyse each unmatched program
    logger.info(f"Analysing {len(unmatched)} unmatched programs...")
    rows: list[dict] = []

    for _, prog in unmatched.iterrows():
        filename = str(prog.get("filename", ""))
        tokens   = filename_tokens(filename)

        cand_job, cand_part, cand_dwg, cand_note = find_candidates(
            tokens, job_by_dwg, job_by_part, job_by_job
        )
        r_hit = closest_router_match(tokens, router_word_idx)
        p_hit = closest_print_match(tokens, print_by_dwg, print_by_part)
        reason = classify_failure(
            filename, tokens,
            job_by_dwg, job_by_part, job_by_job,
            print_by_dwg, print_by_part,
            router_word_idx,
            cand_note,
        )

        rows.append({
            "source_file":                str(prog.get("source_file",    "")),
            "machine_folder":             str(prog.get("machine_folder", "")),
            "filename":                   filename,
            "filename_stem":              Path(filename).stem,
            "extracted_tokens":           "|".join(tokens[:14]),
            "candidate_job_number":       cand_job,
            "candidate_part_number":      cand_part,
            "candidate_drawing_number":   cand_dwg,
            "candidate_match_note":       cand_note,
            "closest_router_match":       r_hit,
            "closest_shared_print_match": p_hit,
            "failure_reason":             reason,
        })

    unmatched_out = pd.DataFrame(rows)

    # ── Export: unmatched_programs ──────────────────────────────────────────
    out1 = EXPORTS_DIR / f"unmatched_programs_{ts}.csv"
    assert_safe_write(out1)
    unmatched_out.to_csv(out1, index=False)
    logger.info(f"Unmatched programs  -> {out1.name}  ({len(unmatched_out)} rows)")

    # ── Export: link_failure_summary ───────────────────────────────────────
    summary = (
        unmatched_out["failure_reason"]
        .value_counts()
        .rename_axis("failure_reason")
        .reset_index(name="count")
    )
    summary["pct"] = (summary["count"] / len(unmatched_out) * 100).round(1)

    out2 = EXPORTS_DIR / f"link_failure_summary_{ts}.csv"
    assert_safe_write(out2)
    summary.to_csv(out2, index=False)
    logger.info(f"Failure summary     -> {out2.name}  ({len(summary)} categories)")

    # ── Export: link_candidate_examples (top 5 categories × 5 examples) ───
    top5_reasons = summary["failure_reason"].head(5).tolist()
    example_rows: list[dict] = []

    for reason in top5_reasons:
        subset = unmatched_out[unmatched_out["failure_reason"] == reason].head(5)
        for _, ex in subset.iterrows():
            ex_d = ex.to_dict()
            example_rows.append({
                "failure_reason":        reason,
                "program_filename":      ex["filename"],
                "machine_folder":        ex["machine_folder"],
                "filename_stem":         ex["filename_stem"],
                "extracted_tokens":      ex["extracted_tokens"],
                "candidate_job_number":  ex["candidate_job_number"],
                "candidate_part_number": ex["candidate_part_number"],
                "expected_drawing":      ex["candidate_drawing_number"],
                "closest_router":        ex["closest_router_match"],
                "closest_print":         ex["closest_shared_print_match"],
                "why_rejected":          _explain(reason, ex_d),
            })

    examples_out = pd.DataFrame(example_rows)
    out3 = EXPORTS_DIR / f"link_candidate_examples_{ts}.csv"
    assert_safe_write(out3)
    examples_out.to_csv(out3, index=False)
    logger.info(f"Candidate examples  -> {out3.name}  ({len(examples_out)} rows)")

    # ── Console report ─────────────────────────────────────────────────────
    w = 65
    print()
    print("=" * w)
    print("  Phase 6C — Link Failure Analysis")
    print("=" * w)
    print(f"  Total programs : {total}")
    print(f"  Linked         : {len(linked_df)} ({len(linked_df)/total*100:.1f}%)")
    print(f"  Unmatched      : {len(unmatched)} ({len(unmatched)/total*100:.1f}%)")
    print()
    print(f"  {'#':<4} {'failure_reason':<44} {'count':>6} {'pct':>6}")
    print(f"  {'-'*4} {'-'*44} {'-'*6} {'-'*6}")
    for i, row in summary.iterrows():
        print(f"  {i+1:<4} {row['failure_reason']:<44} {row['count']:>6} {row['pct']:>5.1f}%")
    print("=" * w)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
