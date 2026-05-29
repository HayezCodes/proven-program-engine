"""
Phase 6D — Composite Part+Op Program Name Analysis

Analyzes pure-numeric composite filenames (part_number + op_seq) on lathe machines
421-437 to determine how many of the 433 programs can safely link.

Does not modify any existing exports or linker behavior.
No scanning of P:, G:, or M:.

Uses existing exports:
  unmatched_programs_*.csv
  program_job_links_*.csv      (for machine_folder)
  router_operations_*.csv
  job_metadata_*.csv

Output:
  exports/composite_program_analysis_*.csv
"""

import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.safety import assert_safe_write
from src.utils import get_logger

logger = get_logger("run_analyze_composite")
EXPORTS_DIR = Path(__file__).parent / "exports"

# Machines that use composite part+op naming
COMPOSITE_MACHINE_FOLDERS = {
    "421, 423, 424",
    "430, 431",
    "432, 437",
    "433, 434",
}


# ---------------------------------------------------------------------------
# Helpers
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


def _machine_ids(machine_folder: str) -> list[str]:
    """Extract individual machine IDs from a folder name like '421, 423, 424'."""
    return [m.strip() for m in str(machine_folder).split(",") if m.strip().isdigit()]


def _parse_part_op(stem: str) -> tuple[str, str]:
    """
    Split composite stem into (part_token, op_token).

    Confirmed pattern from router op descriptions:
      8-digit:  10007001  → part=10007  op=001
      7-digit:  1790001   → part=17900  op=01
      6-digit:  432001    → part=4320   op=01

    Rule: last 2 digits = op for 6-7 digit; last 3 digits = op for 8+ digit.
    """
    n = len(stem)
    if n >= 8:
        return stem[:5], stem[5:]    # 5 + 3
    if n == 7:
        return stem[:5], stem[5:]    # 5 + 2
    if n == 6:
        return stem[:4], stem[4:]    # 4 + 2
    return stem, ""


# ---------------------------------------------------------------------------
# Router index: stem → [(job_number, work_center, machine, op_desc)]
# ---------------------------------------------------------------------------

def build_router_stem_index(
    router_df: pd.DataFrame,
    stems: list[str],
) -> dict[str, list[dict]]:
    """
    For each program stem, collect every router op row that mentions it.
    Returns stem → [row_dict, ...].
    Scans router_df once, O(stems × rows).
    """
    # Pre-lowercase descriptions for fast search
    descs_lower = router_df["operation_description"].str.lower().fillna("").tolist()
    rows_list = router_df.to_dict("records")

    # Build set of lower-cased stems for fast membership check
    stem_set_lower = {s.lower(): s for s in stems}

    index: dict[str, list[dict]] = defaultdict(list)

    for i, desc_l in enumerate(descs_lower):
        for stem_l, stem_orig in stem_set_lower.items():
            if stem_l in desc_l:
                index[stem_orig].append(rows_list[i])

    return dict(index)


def build_router_part_index(
    router_df: pd.DataFrame,
    part_tokens: set[str],
) -> dict[str, list[dict]]:
    """part_token → [row_dicts] where op_description contains the token."""
    descs_lower = router_df["operation_description"].str.lower().fillna("").tolist()
    rows_list   = router_df.to_dict("records")
    pt_lower    = {p.lower(): p for p in part_tokens if len(p) >= 4}

    index: dict[str, list[dict]] = defaultdict(list)
    for i, desc_l in enumerate(descs_lower):
        for pt_l, pt_orig in pt_lower.items():
            if pt_l in desc_l:
                index[pt_orig].append(rows_list[i])
    return dict(index)


# ---------------------------------------------------------------------------
# Work-center machine context check
# ---------------------------------------------------------------------------

def _work_center_machine_matches(
    work_center: str,
    machine_ids: list[str],
) -> list[str]:
    """Return which machine_ids from the folder appear in a work_center string."""
    wc = str(work_center or "").strip()
    return [mid for mid in machine_ids if mid in wc]


def _analyze_one(
    stem: str,
    machine_folder: str,
    stem_hits: list[dict],
    part_hits: list[dict],
) -> dict:
    """
    Returns analysis row for a single composite program.
    """
    mids = _machine_ids(machine_folder)
    part_token, op_token = _parse_part_op(stem)

    # ── Full-stem router analysis ─────────────────────────────────────────
    full_jobs_raw: list[str] = []
    # job → set of work_centers found in ops that reference this program
    job_wcs: dict[str, set[str]] = defaultdict(set)

    for row in stem_hits:
        jn = str(row.get("job_number", "") or "").strip()
        wc = str(row.get("work_center", "") or "").strip()
        if jn and jn.lower() != "nan":
            full_jobs_raw.append(jn)
            if wc:
                job_wcs[jn].add(wc)

    full_jobs  = sorted(set(full_jobs_raw))
    full_wcs   = sorted({wc for wcs in job_wcs.values() for wc in wcs})

    # ── Part-token router analysis (secondary) ────────────────────────────
    part_jobs_raw: list[str] = []
    for row in part_hits:
        jn = str(row.get("job_number", "") or "").strip()
        if jn and jn.lower() != "nan":
            part_jobs_raw.append(jn)
    part_jobs = sorted(set(part_jobs_raw))

    # ── Machine-context check ─────────────────────────────────────────────
    # For each candidate job, which specific sub-machine does the router op cite?
    # job → set of specific machine IDs matched in its work_center(s)
    job_specific_machines: dict[str, set[str]] = {}
    for jn in full_jobs:
        matched = set()
        for wc in job_wcs.get(jn, set()):
            matched.update(_work_center_machine_matches(wc, mids))
        job_specific_machines[jn] = matched

    # Collect jobs where machine context matches the program folder machines
    machine_context_jobs = [jn for jn, ms in job_specific_machines.items() if ms]

    # Check whether work centers across candidate jobs VARY (disambiguation potential)
    # Two jobs referencing the program from different specific machines → date/recency could help
    all_specific_machines = set()
    for ms in job_specific_machines.values():
        all_specific_machines.update(ms)
    wc_varies_across_jobs = len(all_specific_machines) > 1

    # ── Ambiguity reason ─────────────────────────────────────────────────
    n_full = len(full_jobs)

    if n_full == 0:
        ambiguity_reason        = "no_router_reference"
        safe_to_link_candidate  = False
        candidate_job           = ""
    elif n_full == 1:
        ambiguity_reason        = "router_single_job"
        safe_to_link_candidate  = True
        candidate_job           = full_jobs[0]
    else:
        # Multiple jobs reference this program number — check if machine context resolves it
        mc_unique = sorted(set(machine_context_jobs))
        if len(mc_unique) == 1:
            # All work-center evidence points to exactly one job
            ambiguity_reason       = "machine_context_resolved"
            safe_to_link_candidate = True
            candidate_job          = mc_unique[0]
        elif len(mc_unique) == 0:
            # Router ops reference the program but with no recognisable machine ID in work_center
            ambiguity_reason       = "router_multi_job_no_machine_wc_match"
            safe_to_link_candidate = False
            candidate_job          = ""
        else:
            # Machine context present but still points to multiple jobs
            if wc_varies_across_jobs:
                # Different specific sub-machines across jobs — potentially resolvable with dates
                ambiguity_reason   = "router_multi_job_wc_varies_date_needed"
            else:
                # All jobs cite same specific machine — true ambiguity (repeat orders)
                ambiguity_reason   = "router_multi_job_same_specific_wc"
            safe_to_link_candidate = False
            candidate_job          = ""

    return {
        "filename":                         f"{stem}.EIA",   # reconstructed; actual ext may differ
        "machine_folder":                   machine_folder,
        "filename_stem":                    stem,
        "part_token":                       part_token,
        "op_token":                         op_token,
        "router_jobs_full_token":           "|".join(full_jobs),
        "router_jobs_part_token":           "|".join(part_jobs[:8]),
        "router_work_centers":              "|".join(full_wcs),
        "machine_matched_jobs":             "|".join(mc_unique if n_full > 1 else (full_jobs if n_full == 1 else [])),
        "unique_job_count":                 n_full,
        "wc_varies_across_jobs":            wc_varies_across_jobs,
        "ambiguity_reason":                 ambiguity_reason,
        "safe_to_link_candidate":           safe_to_link_candidate,
        "candidate_job_number":             candidate_job,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== Composite Program Analysis ===")
    logger.info("Loading exports...")

    unmatched = _load("unmatched_programs_*.csv",       "unmatched_programs")
    links     = _load("program_job_links_20260529_084354.csv", "program_job_links")
    router    = _load("router_operations_*.csv",        "router_operations")
    _         = _load("job_metadata_*.csv",             "job_metadata")  # loaded for reference

    if unmatched.empty or router.empty:
        logger.error("Required inputs missing — aborting")
        return 1

    # ── Identify composite programs ───────────────────────────────────────
    # Include programs classified as composite in Phase 6C unmatched analysis,
    # plus any composite-pattern programs that were LINKED (shouldn't be many
    # but be complete).
    comp_reasons = {"composite_part_op_no_match", "composite_part_op_near_match"}
    comp_unmatched = unmatched[unmatched["failure_reason"].isin(comp_reasons)].copy()

    # Also pick up composite-pattern programs from the full links that might
    # have linked via router in Phase 6D (to understand what DID link)
    comp_linked = links[
        (links["machine_folder"].isin(COMPOSITE_MACHINE_FOLDERS)) &
        (links["link_method"] != "no_match") &
        (links["filename"].str.extract(r"^(\d+)\.", expand=False).str.len() >= 6)
    ].copy() if not links.empty else pd.DataFrame()

    logger.info(f"Composite unmatched: {len(comp_unmatched)}")
    logger.info(f"Composite linked (Phase 6D):   {len(comp_linked)}")

    # Build full analysis set from unmatched (the primary subject)
    # Use actual filename, machine_folder, and stem
    comp_unmatched["stem"] = comp_unmatched["filename"].apply(lambda f: Path(f).stem)
    stems_to_analyse = comp_unmatched[["stem", "machine_folder", "filename", "source_file"]].copy()
    stems_to_analyse = stems_to_analyse.drop_duplicates(subset=["stem", "machine_folder"])

    # ── Build router indexes ──────────────────────────────────────────────
    all_stems   = stems_to_analyse["stem"].tolist()
    part_tokens = {_parse_part_op(s)[0] for s in all_stems}

    logger.info(f"Building router index for {len(all_stems)} stems ...")
    stem_index = build_router_stem_index(router, all_stems)
    logger.info(f"  stems with any router hit: {len(stem_index)}")

    logger.info(f"Building router index for {len(part_tokens)} part tokens ...")
    part_index = build_router_part_index(router, part_tokens)
    logger.info(f"  part tokens with any router hit: {len(part_index)}")

    # ── Analyse each program ──────────────────────────────────────────────
    logger.info(f"Analysing {len(stems_to_analyse)} composite programs ...")
    rows: list[dict] = []

    for _, prog in stems_to_analyse.iterrows():
        stem           = prog["stem"]
        machine_folder = prog["machine_folder"]
        part_tok, _    = _parse_part_op(stem)

        result = _analyze_one(
            stem           = stem,
            machine_folder = machine_folder,
            stem_hits      = stem_index.get(stem, []),
            part_hits      = part_index.get(part_tok, []),
        )
        # Restore actual filename and source_file
        result["filename"]    = prog["filename"]
        result["source_file"] = prog["source_file"]
        rows.append(result)

    out_df = pd.DataFrame(rows).sort_values(
        ["machine_folder", "ambiguity_reason", "filename_stem"]
    )

    # ── Export ────────────────────────────────────────────────────────────
    out_path = EXPORTS_DIR / f"composite_program_analysis_{ts}.csv"
    assert_safe_write(out_path)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Composite analysis -> {out_path.name}  ({len(out_df)} rows)")

    # ── Summary report ────────────────────────────────────────────────────
    total      = len(out_df)
    safe_count = out_df["safe_to_link_candidate"].sum()
    w = 70

    print()
    print("=" * w)
    print("  Composite Program Analysis")
    print("=" * w)
    print(f"  Total composite programs analysed : {total}")
    print(f"  Safe to link (candidate exists)   : {safe_count}  ({safe_count/total*100:.1f}%)")
    print(f"  Not safe to link                  : {total-safe_count}  ({(total-safe_count)/total*100:.1f}%)")
    print()

    reason_counts = out_df["ambiguity_reason"].value_counts()
    print(f"  {'Ambiguity reason':<46} {'count':>6} {'pct':>6}")
    print(f"  {'-'*46} {'-'*6} {'-'*6}")
    for reason, cnt in reason_counts.items():
        print(f"  {reason:<46} {cnt:>6} {cnt/total*100:>5.1f}%")
    print()

    print("  By machine folder:")
    print(f"  {'machine_folder':<22} {'total':>6} {'safe':>6} {'no_router':>10} {'multi_job':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*10} {'-'*10}")
    for mf, grp in out_df.groupby("machine_folder"):
        safe  = grp["safe_to_link_candidate"].sum()
        no_r  = (grp["ambiguity_reason"] == "no_router_reference").sum()
        multi = (grp["unique_job_count"] > 1).sum()
        print(f"  {mf:<22} {len(grp):>6} {safe:>6} {no_r:>10} {multi:>10}")
    print()

    print("  unique_job_count distribution (programs with router hit):")
    has_router = out_df[out_df["unique_job_count"] > 0]
    jc_dist = has_router["unique_job_count"].value_counts().sort_index()
    for jc, cnt in jc_dist.items():
        print(f"    {jc:>2} job(s) : {cnt:>3} programs")
    print()

    wc_varies = out_df["wc_varies_across_jobs"].sum()
    print(f"  Programs where work-center varies across candidate jobs : {wc_varies}")
    print(f"  (These could be disambiguated with file-modification date)")
    print()

    print("=" * w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
