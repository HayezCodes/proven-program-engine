"""
job_linker.py — Deterministic linker: proven programs ↔ job/router/print metadata.

Connects CNC programs to job, router, and shared-print metadata to enable
source-verified material backfill into the proven S/F database.

All matching is deterministic — no AI, agents, embeddings, or LLMs.
Priority order: exact_drawing_number > exact_job_number > exact_part_number >
shared_print_bridge > router_match > machine_context_assist (disambiguation).

READ-ONLY against P:\\ and G:\\ — never modifies source files.
Exports are timestamped — never overwrites existing CSVs.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column schemas
# ---------------------------------------------------------------------------

_PROG_JOB_LINK_COLS = [
    "source_file", "filename", "machine_folder", "program_id",
    "matched_job_number", "matched_part_number", "matched_drawing_number",
    "matched_revision", "matched_shared_print_file", "matched_router_file",
    "link_confidence", "link_method", "link_reason",
    "material", "material_source", "material_confidence",
    "material_conflict", "needs_review",
    "candidate_job_count", "candidate_materials", "material_consensus_status",
]

_MATERIAL_BACKFILL_COLS = [
    "source_file", "machine_folder", "tool_number", "resolved_tool_description",
    "S", "s_mode", "s_type", "F", "f_mode",
    "verified_material", "material_source", "material_confidence",
    "matched_job_number", "matched_part_number", "matched_drawing_number",
    "link_confidence", "link_method", "link_reason", "needs_review",
    "candidate_job_count", "candidate_materials", "material_consensus_status",
]

_ROUTER_CONTEXT_COLS = [
    "matched_job_number", "matched_part_number", "matched_drawing_number",
    "operation_number", "work_center", "work_center_code", "work_center_type",
    "machine_hint", "operation_description",
    "program_reference", "source_router_file", "source_file", "machine_folder",
    "context_match_confidence", "context_match_reason",
]

# ---------------------------------------------------------------------------
# Auto-detection of latest exports
# ---------------------------------------------------------------------------

def _latest_export(pattern: str, exports_dir: Path) -> Path | None:
    """Return the most recently modified file matching pattern, or None."""
    try:
        matches = sorted(
            exports_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return None
    return matches[-1] if matches else None


def detect_latest_exports(exports_dir: Path) -> dict[str, Path | None]:
    """Return paths to the latest of each expected Phase 6 export file type."""
    return {
        "manifest":     _latest_export("manifest_*.csv",           exports_dir),
        "cuts":         _latest_export("cuts_*.csv",               exports_dir),
        "tool_summary": _latest_export("tool_summary_*.csv",       exports_dir),
        "job_metadata": _latest_export("job_metadata_*.csv",       exports_dir),
        "shared_print": _latest_export("shared_print_index_*.csv", exports_dir),
        "router_ops":   _latest_export("router_operations_*.csv",  exports_dir),
    }


# ---------------------------------------------------------------------------
# Token extraction and normalization
# ---------------------------------------------------------------------------

def _normalize_id(s: str) -> str:
    """Normalize an identifier for matching.

    Pure-numeric strings: strip leading zeros ('0262' → '262').
    Others: lowercase and strip whitespace.
    """
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return str(int(s)) if s else ""
    return s.lower()


def filename_tokens(filename: str) -> list[str]:
    """Extract candidate match tokens from a program filename (stem only).

    Handles the two main production patterns:
      - EM-prefixed drawing numbers: EM10986.NC → ['EM10986', 'em10986', '10986']
      - Pure-numeric job numbers:    0262.TAP   → ['0262', '262']
      - Hyphenated:                  EM0170-HEX → includes 'EM0170', '0170', '170'

    Returns a deduplicated, insertion-ordered list.
    """
    stem = Path(filename).stem
    seen: dict[str, None] = {}

    def _add(t: str) -> None:
        if t and t not in seen:
            seen[t] = None

    _add(stem)
    _add(_normalize_id(stem))  # lowercase / strip numeric leading zeros

    # Strip leading letter prefix (e.g. 'EM10986' → '10986')
    m = re.match(r"^([A-Za-z]+)(\d.*)", stem)
    if m:
        _add(m.group(2))
        _add(_normalize_id(m.group(2)))

    # All numeric substrings (handles hyphenated stems like EM0170-HEX)
    for num in re.findall(r"\d+", stem):
        _add(num)
        _add(_normalize_id(num))
        # For 7+ digit numbers try shorter prefix slices (part number embedded in stem)
        for prefix_len in range(5, min(len(num), 8)):
            _add(num[:prefix_len])
            _add(_normalize_id(num[:prefix_len]))

    return list(seen)


def _extract_machine_id(machine_folder: str) -> str:
    """Extract the leading 3-4 digit machine ID from a folder name.

    '655 HAAS VMC' → '655',  '417,426' → '417',  '421, 423, 424' → '421'
    """
    m = re.match(r"(\d{3,4})", str(machine_folder).strip())
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def _add_to_index(idx: dict[str, list[dict]], raw_key: str, row: dict) -> None:
    norm = _normalize_id(raw_key)
    if norm:
        idx.setdefault(norm, []).append(row)


def build_indexes(
    job_df: pd.DataFrame,
    print_df: pd.DataFrame,
) -> tuple[dict, dict, dict, dict, dict]:
    """Build O(1) lookup indexes from job metadata and shared print DataFrames.

    Returns:
      job_by_job   — norm(job_number)     → [job_row, ...]
      job_by_part  — norm(part_number)    → [job_row, ...]
      job_by_dwg   — norm(drawing_number) → [job_row, ...]
      print_by_part — norm(part_number)   → [print_row, ...]
      print_by_dwg  — norm(drawing_number)→ [print_row, ...]
    """
    job_by_job:  dict[str, list[dict]] = {}
    job_by_part: dict[str, list[dict]] = {}
    job_by_dwg:  dict[str, list[dict]] = {}

    for _, row in job_df.iterrows():
        r = row.to_dict()
        for field, idx in [
            ("job_number",     job_by_job),
            ("part_number",    job_by_part),
            ("drawing_number", job_by_dwg),
        ]:
            val = str(r.get(field, "") or "").strip()
            if val:
                _add_to_index(idx, val, r)

    print_by_part: dict[str, list[dict]] = {}
    print_by_dwg:  dict[str, list[dict]] = {}

    for _, row in print_df.iterrows():
        r = row.to_dict()
        for field, idx in [
            ("part_number",    print_by_part),
            ("drawing_number", print_by_dwg),
        ]:
            val = str(r.get(field, "") or "").strip()
            if val:
                _add_to_index(idx, val, r)

    return job_by_job, job_by_part, job_by_dwg, print_by_part, print_by_dwg


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

def _mat_from_job(job_row: dict) -> tuple[str, str, str]:
    mat = str(
        job_row.get("normalized_material", "")
        or job_row.get("material", "")
        or ""
    ).strip()
    return (mat, "ROUTER", "HIGH") if mat else ("UNKNOWN", "UNKNOWN", "NONE")


def _mat_from_print(print_row: dict) -> tuple[str, str, str]:
    mat = str(
        print_row.get("normalized_material", "")
        or print_row.get("material", "")
        or ""
    ).strip()
    return (mat, "SHARED_PRINT", "MEDIUM") if mat else ("UNKNOWN", "UNKNOWN", "NONE")


def resolve_material(
    job_rows: list[dict],
    print_rows: list[dict],
) -> tuple[str, str, str, bool]:
    """Resolve material from job and print candidates.

    Priority: ROUTER > SHARED_PRINT > UNKNOWN.
    Returns (material, source, confidence, conflict).
    Conflict is True when multiple distinct non-empty materials are found.
    """
    candidates: list[tuple[str, str, str]] = []

    for jr in job_rows:
        mat, src, conf = _mat_from_job(jr)
        if mat != "UNKNOWN":
            candidates.append((mat, src, conf))

    if not candidates:
        for pr in print_rows:
            mat, src, conf = _mat_from_print(pr)
            if mat != "UNKNOWN":
                candidates.append((mat, src, conf))

    if not candidates:
        return "UNKNOWN", "UNKNOWN", "NONE", False

    # Also scan print materials for conflict detection even when job has material
    all_mats: list[tuple[str, str, str]] = list(candidates)
    if job_rows:
        for pr in print_rows:
            mat, src, conf = _mat_from_print(pr)
            if mat != "UNKNOWN":
                all_mats.append((mat, src, conf))

    unique = list(dict.fromkeys(m for m, _, _ in all_mats))
    conflict = len(unique) > 1

    mat, src, conf = candidates[0]
    return mat, src, conf, conflict


# ---------------------------------------------------------------------------
# Machine context helpers
# ---------------------------------------------------------------------------

def _machine_in_ops(machine_id: str, ops: list[dict]) -> bool:
    if not machine_id:
        return False
    for op in ops:
        wc   = str(op.get("work_center", "") or "")
        mach = str(op.get("machine", "") or "")
        if machine_id in wc or machine_id in mach:
            return True
    return False


# ---------------------------------------------------------------------------
# Link record constructors
# ---------------------------------------------------------------------------

def _base_link(program_row: dict) -> dict:
    return {
        "source_file":               str(program_row.get("source_file", "")),
        "filename":                  str(program_row.get("filename", "")),
        "machine_folder":            str(program_row.get("machine_folder", "")),
        "program_id":                program_row.get("program_id", ""),
        "matched_job_number":        "",
        "matched_part_number":       "",
        "matched_drawing_number":    "",
        "matched_revision":          "",
        "matched_shared_print_file": "",
        "matched_router_file":       "",
        "link_confidence":           "NONE",
        "link_method":               "no_match",
        "link_reason":               "",
        "material":                  "UNKNOWN",
        "material_source":           "UNKNOWN",
        "material_confidence":       "NONE",
        "material_conflict":         False,
        "needs_review":              False,
        "candidate_job_count":       0,
        "candidate_materials":       "",
        "material_consensus_status": "not_applicable",
    }


def _fill_from_job(link: dict, job_row: dict) -> None:
    link["matched_job_number"]     = str(job_row.get("job_number",     "") or "")
    link["matched_part_number"]    = str(job_row.get("part_number",    "") or "")
    link["matched_drawing_number"] = str(job_row.get("drawing_number", "") or "")
    link["matched_revision"]       = str(job_row.get("revision",       "") or "")
    link["matched_router_file"]    = str(job_row.get("source_file",    "") or "")


def _fill_from_print(link: dict, print_row: dict) -> None:
    link["matched_shared_print_file"] = str(print_row.get("source_file",    "") or "")
    if not link["matched_part_number"]:
        link["matched_part_number"]    = str(print_row.get("part_number",    "") or "")
    if not link["matched_drawing_number"]:
        link["matched_drawing_number"] = str(print_row.get("drawing_number", "") or "")
    if not link["matched_revision"]:
        link["matched_revision"]       = str(print_row.get("revision",       "") or "")


# ---------------------------------------------------------------------------
# Core matching engine
# ---------------------------------------------------------------------------

def _match_program(
    program_row: dict,
    indexes: tuple[dict, dict, dict, dict, dict],
    router_df: pd.DataFrame,
    router_by_job: dict[str, list[dict]],
    router_by_source: dict[str, list[dict]],
) -> dict:
    """Deterministic program→job linking for one program row.

    Tries strategies in priority order and returns on first successful match.
    """
    job_by_job, job_by_part, job_by_dwg, print_by_part, print_by_dwg = indexes
    filename      = str(program_row.get("filename", ""))
    machine_folder = str(program_row.get("machine_folder", "") or "")
    machine_id    = _extract_machine_id(machine_folder)
    tokens        = filename_tokens(filename)

    def _prints_for_job(job_row: dict) -> list[dict]:
        """Look up shared prints matching a job's drawing_number or part_number."""
        prints: list[dict] = []
        dwg = _normalize_id(str(job_row.get("drawing_number", "") or ""))
        pn  = _normalize_id(str(job_row.get("part_number",    "") or ""))
        for idx, key in [(print_by_dwg, dwg), (print_by_part, pn)]:
            for h in idx.get(key, []):
                if h not in prints:
                    prints.append(h)
        return prints

    def _apply_job_hits(
        hits: list[dict],
        method: str,
        tok: str,
        method_label: str,
    ) -> dict:
        """Handle a list of job hits with disambiguation and ambiguity logic."""
        link = _base_link(program_row)
        if len(hits) == 1:
            _fill_from_job(link, hits[0])
            # Check for conflicting shared-print materials on the same drawing/part
            matched_prints = _prints_for_job(hits[0])
            if matched_prints:
                _fill_from_print(link, matched_prints[0])
            mat, src, conf, conflict = resolve_material(hits, matched_prints)
            link.update({
                "link_method":         method,
                "link_confidence":     "HIGH",
                "link_reason":         f"filename token '{tok}' matched {method_label}",
                "material":            mat,
                "material_source":     src,
                "material_confidence": conf,
                "material_conflict":   conflict,
                "needs_review":        conflict,
            })
        else:
            # Try machine context disambiguation via the job's own source_file
            ops_match = [
                h for h in hits
                if _machine_in_ops(
                    machine_id,
                    router_by_source.get(str(h.get("source_file", "") or ""), []),
                )
            ]
            if len(ops_match) == 1:
                _fill_from_job(link, ops_match[0])
                matched_prints = _prints_for_job(ops_match[0])
                mat, src, conf, conflict = resolve_material(ops_match, matched_prints)
                link.update({
                    "link_method":         "machine_context_assist",
                    "link_confidence":     "HIGH",
                    "link_reason":         (
                        f"filename token '{tok}' matched {method_label} "
                        f"({len(hits)} hits); disambiguated by machine_id '{machine_id}'"
                    ),
                    "material":            mat,
                    "material_source":     src,
                    "material_confidence": conf,
                    "material_conflict":   conflict,
                    "needs_review":        conflict,
                })
            else:
                _fill_from_job(link, hits[0])
                link.update({
                    "link_method":         "ambiguous_match",
                    "link_confidence":     "MEDIUM",
                    "link_reason":         (
                        f"filename token '{tok}' matched {len(hits)} {method_label}(s) "
                        f"— review required"
                    ),
                    "material":            "UNKNOWN",
                    "material_source":     "UNKNOWN",
                    "material_confidence": "NONE",
                    "material_conflict":   True,
                    "needs_review":        True,
                })
        return link

    # ------------------------------------------------------------------
    # Strategy 1: exact drawing number via job metadata
    # ------------------------------------------------------------------
    for tok in tokens:
        hits = job_by_dwg.get(tok, [])
        if hits:
            return _apply_job_hits(hits, "exact_drawing_number", tok, "drawing_number")

    # ------------------------------------------------------------------
    # Strategy 2: exact job number
    # ------------------------------------------------------------------
    for tok in tokens:
        hits = job_by_job.get(tok, [])
        if hits:
            return _apply_job_hits(hits, "exact_job_number", tok, "job_number")

    # ------------------------------------------------------------------
    # Strategy 3: exact part number via job metadata
    # ------------------------------------------------------------------
    for tok in tokens:
        hits = job_by_part.get(tok, [])
        if hits:
            return _apply_job_hits(hits, "exact_part_number", tok, "part_number")

    # ------------------------------------------------------------------
    # Strategy 4: shared print bridge → job
    # ------------------------------------------------------------------
    for tok in tokens:
        print_hits: list[dict] = []
        for idx in (print_by_dwg, print_by_part):
            for h in idx.get(tok, []):
                if h not in print_hits:
                    print_hits.append(h)
        if not print_hits:
            continue

        link = _base_link(program_row)
        _fill_from_print(link, print_hits[0])

        # Bridge from print to job metadata
        job_via_print: list[dict] = []
        for ph in print_hits:
            dwg = _normalize_id(str(ph.get("drawing_number", "") or ""))
            pn  = _normalize_id(str(ph.get("part_number",    "") or ""))
            for idx, key in [(job_by_dwg, dwg), (job_by_part, pn)]:
                for h in idx.get(key, []):
                    if h not in job_via_print:
                        job_via_print.append(h)
                if job_via_print:
                    break
            if job_via_print:
                break

        if job_via_print:
            _fill_from_job(link, job_via_print[0])

        mat, src, conf, conflict = resolve_material(job_via_print, print_hits)
        link.update({
            "link_method":         "shared_print_bridge",
            "link_confidence":     "MEDIUM",
            "link_reason":         (
                f"filename token '{tok}' matched shared print; "
                f"{'bridged to job' if job_via_print else 'no job bridge found'}"
            ),
            "material":            mat,
            "material_source":     src,
            "material_confidence": conf,
            "material_conflict":   conflict,
            "needs_review":        conflict or len(print_hits) > 1,
        })
        return link

    # ------------------------------------------------------------------
    # Strategy 5: router description match (LOW confidence)
    # ------------------------------------------------------------------
    if not router_df.empty and "operation_description" in router_df.columns:
        for tok in tokens:
            # Allow 5+ char tokens (original behaviour) and exactly-4-digit
            # pure-numeric tokens (e.g. machine 417/426 programs like 0582.OP1).
            # Shorter tokens and 4-char non-numeric tokens are too ambiguous.
            if len(tok) < 4:
                continue
            if len(tok) == 4 and not tok.isdigit():
                continue
            matched = router_df[
                router_df["operation_description"].str.contains(
                    re.escape(tok), case=False, na=False
                )
            ]
            if matched.empty:
                continue
            unique_jobs = matched["job_number"].dropna().unique()
            if len(unique_jobs) != 1:
                continue
            jn = _normalize_id(str(unique_jobs[0]))
            job_rows = job_by_job.get(jn, [])
            if not job_rows:
                continue
            link = _base_link(program_row)
            _fill_from_job(link, job_rows[0])
            mat, src, conf, conflict = resolve_material(job_rows, [])
            link.update({
                "link_method":         "router_match",
                "link_confidence":     "LOW",
                "link_reason":         (
                    f"filename token '{tok}' found in router op description "
                    f"for job {unique_jobs[0]}"
                ),
                "material":            mat,
                "material_source":     src,
                "material_confidence": conf,
                "material_conflict":   conflict,
                "needs_review":        True,
            })
            return link

    # ------------------------------------------------------------------
    # Strategy 6: composite material consensus (MEDIUM confidence)
    # Fires when multiple router jobs reference this program and enough
    # candidate materials agree (>= 2 jobs must confirm the same material).
    # Verifies material only — does NOT assign a specific job number.
    # ------------------------------------------------------------------
    if not router_df.empty and "operation_description" in router_df.columns:
        for tok in tokens:
            if len(tok) < 4:
                continue
            if len(tok) == 4 and not tok.isdigit():
                continue
            matched = router_df[
                router_df["operation_description"].str.contains(
                    re.escape(tok), case=False, na=False
                )
            ]
            if matched.empty:
                continue
            unique_jobs_raw = [
                str(j) for j in matched["job_number"].dropna().unique()
                if str(j).strip().lower() not in ("nan", "")
            ]
            if len(unique_jobs_raw) < 2:
                continue  # Strategy 5 already handles the 0- and 1-job cases

            # Collect the first known material per candidate job (one per job,
            # so count_agree reflects distinct jobs, not router-file duplicates).
            job_mat: dict[str, str] = {}
            for jn_raw in unique_jobs_raw:
                jn_norm = _normalize_id(jn_raw)
                for jr in job_by_job.get(jn_norm, []):
                    mat = str(
                        jr.get("normalized_material", "")
                        or jr.get("material", "")
                        or ""
                    ).strip()
                    if mat and mat.upper() not in ("UNKNOWN", "NAN", ""):
                        job_mat[jn_raw] = mat
                        break  # one material per job

            known_mats   = list(job_mat.values())
            unique_known = list(dict.fromkeys(known_mats))

            if len(unique_known) == 0:
                consensus_status = "insufficient_material_data"
                consensus_mat    = "UNKNOWN"
                mat_source       = "UNKNOWN"
                mat_conf         = "NONE"
                conflict         = False
                needs_rev        = False
            elif len(unique_known) == 1 and known_mats.count(unique_known[0]) >= 2:
                # >= 2 distinct jobs confirm the same material, no conflicts
                consensus_status = "consensus_material"
                consensus_mat    = unique_known[0]
                mat_source       = "ROUTER_CONSENSUS"
                mat_conf         = "MEDIUM"
                conflict         = False
                needs_rev        = False
            elif len(unique_known) == 1:
                # Only one job has a known material; rest are unknown — insufficient
                consensus_status = "insufficient_material_data"
                consensus_mat    = "UNKNOWN"
                mat_source       = "UNKNOWN"
                mat_conf         = "NONE"
                conflict         = False
                needs_rev        = False
            else:
                # Multiple distinct known materials across candidate jobs — conflict
                consensus_status = "conflicting_materials"
                consensus_mat    = "UNKNOWN"
                mat_source       = "UNKNOWN"
                mat_conf         = "NONE"
                conflict         = True
                needs_rev        = True

            link = _base_link(program_row)
            link.update({
                "link_method":               "composite_material_consensus",
                "link_confidence":           "MEDIUM",
                "link_reason":               (
                    f"filename token '{tok}' found in "
                    f"{len(unique_jobs_raw)} router jobs; "
                    f"material consensus: {consensus_status}"
                ),
                "matched_job_number":        "MULTIPLE",
                "material":                  consensus_mat,
                "material_source":           mat_source,
                "material_confidence":       mat_conf,
                "material_conflict":         conflict,
                "needs_review":              needs_rev,
                "candidate_job_count":       len(unique_jobs_raw),
                "candidate_materials":       "|".join(unique_known),
                "material_consensus_status": consensus_status,
            })
            return link

    # ------------------------------------------------------------------
    # No match
    # ------------------------------------------------------------------
    link = _base_link(program_row)
    link["link_reason"] = (
        "no filename tokens matched any job_number, part_number, drawing_number, "
        "shared print, or router description"
    )
    return link


# ---------------------------------------------------------------------------
# Program–job links builder
# ---------------------------------------------------------------------------

def build_program_job_links(
    manifest_df: pd.DataFrame,
    job_df:      pd.DataFrame,
    print_df:    pd.DataFrame,
    router_df:   pd.DataFrame,
) -> pd.DataFrame:
    """Link every included program in manifest to job/router/print metadata.

    Returns a DataFrame with one row per included program.
    """
    # Filter to included (Proven-scanner-approved) programs only
    total_in_manifest = len(manifest_df)
    if "included" in manifest_df.columns:
        programs = manifest_df[manifest_df["included"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )]
        excluded = total_in_manifest - len(programs)
        logger.info(
            f"Manifest filter (included==True): "
            f"{len(programs)} included / {excluded} excluded / {total_in_manifest} total"
        )
        logger.info("Confirmed: linking against Proven-scanner included==True records only")
    else:
        programs = manifest_df
        logger.warning(
            "Manifest has no 'included' column — processing all rows. "
            "Verify this is a Proven-scanner manifest."
        )

    if programs.empty:
        return pd.DataFrame(columns=_PROG_JOB_LINK_COLS)

    indexes = build_indexes(
        job_df  if not job_df.empty   else pd.DataFrame(),
        print_df if not print_df.empty else pd.DataFrame(),
    )

    # Pre-build router indexes
    # router_by_job: norm(job_number) → ops  (for router_match strategy)
    # router_by_source: job source_file → ops (for machine context disambiguation)
    router_by_job:    dict[str, list[dict]] = {}
    router_by_source: dict[str, list[dict]] = {}
    if not router_df.empty:
        for _, row in router_df.iterrows():
            r = row.to_dict()
            jn = _normalize_id(str(r.get("job_number", "") or ""))
            if jn:
                router_by_job.setdefault(jn, []).append(r)
            sf = str(r.get("source_file", "") or "")
            if sf:
                router_by_source.setdefault(sf, []).append(r)

    links: list[dict] = []
    n_matched = 0
    for _, row in programs.iterrows():
        result = _match_program(
            row.to_dict(), indexes, router_df, router_by_job, router_by_source
        )
        if result["link_method"] != "no_match":
            n_matched += 1
        links.append(result)

    logger.info(
        f"Program–job linking: {len(links)} programs | "
        f"{n_matched} linked | {len(links) - n_matched} unmatched"
    )

    df = pd.DataFrame(links, columns=_PROG_JOB_LINK_COLS)
    return df


# ---------------------------------------------------------------------------
# Material backfill builder
# ---------------------------------------------------------------------------

def build_material_backfill(
    cuts_df:  pd.DataFrame,
    links_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join per-cut-record S/F data with program-level verified material.

    Returns one row per cut record (left join — unmatched programs get UNKNOWN material).
    """
    if cuts_df.empty:
        return pd.DataFrame(columns=_MATERIAL_BACKFILL_COLS)

    # Rename cuts columns to backfill names
    cuts_sel = cuts_df[[
        "source_file", "machine_folder", "tool_number", "tool_description",
        "s_value", "s_mode", "s_type", "f_value", "f_mode",
    ]].copy()
    cuts_sel.rename(columns={
        "tool_description": "resolved_tool_description",
        "s_value":          "S",
        "f_value":          "F",
    }, inplace=True)

    if links_df.empty:
        for col in _MATERIAL_BACKFILL_COLS:
            if col not in cuts_sel.columns:
                cuts_sel[col] = "UNKNOWN" if "material" in col or "source" in col or "confidence" in col else (
                    "NONE" if col.endswith("_confidence") or col == "link_confidence" else (
                        False if col in ("needs_review",) else ""
                    )
                )
        # Correct defaults for new consensus columns (generic heuristic above mis-fills these)
        cuts_sel["candidate_job_count"]       = 0
        cuts_sel["candidate_materials"]       = ""
        cuts_sel["material_consensus_status"] = "not_applicable"
        return cuts_sel[_MATERIAL_BACKFILL_COLS]

    link_sel = links_df[[
        "source_file",
        "matched_job_number", "matched_part_number", "matched_drawing_number",
        "link_confidence", "link_method", "link_reason",
        "material", "material_source", "material_confidence",
        "needs_review",
        "candidate_job_count", "candidate_materials", "material_consensus_status",
    ]].copy()
    link_sel.rename(columns={
        "material":            "verified_material",
    }, inplace=True)

    merged = cuts_sel.merge(link_sel, on="source_file", how="left")

    # Fill NaN for unmatched programs
    str_fill = {
        "verified_material":         "UNKNOWN",
        "material_source":           "UNKNOWN",
        "material_confidence":       "NONE",
        "link_confidence":           "NONE",
        "link_method":               "no_match",
        "link_reason":               "",
        "matched_job_number":        "",
        "matched_part_number":       "",
        "matched_drawing_number":    "",
        "candidate_materials":       "",
        "material_consensus_status": "not_applicable",
    }
    for col, val in str_fill.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(val)

    merged["needs_review"] = (merged["needs_review"] == True)  # NaN → False
    if "candidate_job_count" in merged.columns:
        merged["candidate_job_count"] = (
            pd.to_numeric(merged["candidate_job_count"], errors="coerce").fillna(0).astype(int)
        )

    return merged[_MATERIAL_BACKFILL_COLS]


# ---------------------------------------------------------------------------
# Router program context builder
# ---------------------------------------------------------------------------

def build_router_program_context(
    links_df:  pd.DataFrame,
    router_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-reference linked programs with their router operations.

    Returns one row per (program, router_operation) pair where a job match exists.
    """
    if links_df.empty or router_df.empty:
        return pd.DataFrame(columns=_ROUTER_CONTEXT_COLS)

    linked = links_df[
        links_df["matched_job_number"].astype(str).str.strip().ne("")
    ]
    if linked.empty:
        return pd.DataFrame(columns=_ROUTER_CONTEXT_COLS)

    rows: list[dict] = []
    for _, link in linked.iterrows():
        jn = str(link["matched_job_number"])
        job_ops = router_df[router_df["job_number"].astype(str) == jn]
        if job_ops.empty:
            continue
        for _, op in job_ops.iterrows():
            rows.append({
                "matched_job_number":     jn,
                "matched_part_number":    str(link.get("matched_part_number",    "") or ""),
                "matched_drawing_number": str(link.get("matched_drawing_number", "") or ""),
                "operation_number":       str(op.get("operation_number",    "") or ""),
                "work_center":            str(op.get("work_center",         "") or ""),
                "work_center_code":       str(op.get("work_center_code",    "") or ""),
                "work_center_type":       str(op.get("work_center_type",    "") or ""),
                "machine_hint":           str(op.get("machine",             "") or ""),
                "operation_description":  str(op.get("operation_description","") or ""),
                "program_reference":      str(link.get("filename",          "") or ""),
                "source_router_file":     str(op.get("source_file",         "") or ""),
                "source_file":            str(link.get("source_file",       "") or ""),
                "machine_folder":         str(link.get("machine_folder",    "") or ""),
                "context_match_confidence": str(link.get("link_confidence", "") or ""),
                "context_match_reason":   str(link.get("link_reason",       "") or ""),
            })

    if not rows:
        return pd.DataFrame(columns=_ROUTER_CONTEXT_COLS)
    return pd.DataFrame(rows, columns=_ROUTER_CONTEXT_COLS)


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def export_program_job_links(
    df: pd.DataFrame, exports_dir: Path, timestamp: str
) -> Path:
    out = exports_dir / f"program_job_links_{timestamp}.csv"
    assert_safe_write(out)
    (pd.DataFrame(columns=_PROG_JOB_LINK_COLS) if df.empty else df).to_csv(out, index=False)
    logger.info(f"Program–job links  -> {out}  ({len(df)} row(s))")
    return out


def export_material_backfill(
    df: pd.DataFrame, exports_dir: Path, timestamp: str
) -> Path:
    out = exports_dir / f"material_backfill_{timestamp}.csv"
    assert_safe_write(out)
    (pd.DataFrame(columns=_MATERIAL_BACKFILL_COLS) if df.empty else df).to_csv(out, index=False)
    logger.info(f"Material backfill  -> {out}  ({len(df)} row(s))")
    return out


def export_router_program_context(
    df: pd.DataFrame, exports_dir: Path, timestamp: str
) -> Path:
    out = exports_dir / f"router_program_context_{timestamp}.csv"
    assert_safe_write(out)
    (pd.DataFrame(columns=_ROUTER_CONTEXT_COLS) if df.empty else df).to_csv(out, index=False)
    logger.info(f"Router context     -> {out}  ({len(df)} row(s))")
    return out


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_job_link(
    exports_dir:      Path | None = None,
    manifest_path:    Path | None = None,
    cuts_path:        Path | None = None,
    job_metadata_path: Path | None = None,
    shared_print_path: Path | None = None,
    router_ops_path:   Path | None = None,
) -> tuple[Path, Path, Path]:
    """Phase 6B pipeline: link programs to job metadata, backfill material.

    Auto-detects latest exports when paths are not supplied.
    Returns (program_job_links_path, material_backfill_path, router_context_path).
    Never writes to P:\\ or G:\\. Never overwrites existing exports.
    """
    if exports_dir is None:
        exports_dir = Path(__file__).parent.parent / "exports"
    assert_safe_write(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect latest files when not explicitly provided
    detected = detect_latest_exports(exports_dir)

    def _resolve(explicit: Path | None, key: str, label: str) -> pd.DataFrame:
        path = explicit or detected.get(key)
        if path is None or not path.exists():
            logger.warning(f"{label} not found — proceeding with empty DataFrame")
            return pd.DataFrame()
        logger.info(f"Loading {label}: {path.name}")
        try:
            return pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            logger.warning(f"{label} is empty — proceeding with empty DataFrame")
            return pd.DataFrame()

    manifest_df  = _resolve(manifest_path,     "manifest",     "manifest")
    cuts_df      = _resolve(cuts_path,         "cuts",         "cuts")
    job_df       = _resolve(job_metadata_path, "job_metadata", "job_metadata")
    print_df     = _resolve(shared_print_path, "shared_print", "shared_print_index")
    router_df    = _resolve(router_ops_path,   "router_ops",   "router_operations")

    # Resolve actual source filenames for audit trail
    _manifest_file = (manifest_path or detected.get("manifest") or Path("(none)")).name
    _cuts_file     = (cuts_path     or detected.get("cuts")     or Path("(none)")).name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== Phase 6B — Program–Job Linker ===")
    logger.info(f"Exports dir    : {exports_dir}")
    logger.info(f"Source manifest: {_manifest_file}")
    logger.info(f"Source cuts    : {_cuts_file}")

    links_df   = build_program_job_links(manifest_df, job_df, print_df, router_df)
    backfill_df = build_material_backfill(cuts_df, links_df)
    context_df = build_router_program_context(links_df, router_df)

    links_path   = export_program_job_links(links_df,    exports_dir, timestamp)
    backfill_path = export_material_backfill(backfill_df, exports_dir, timestamp)
    context_path  = export_router_program_context(context_df, exports_dir, timestamp)

    # Coverage summary
    if not links_df.empty:
        total   = len(links_df)
        matched = (links_df["link_method"] != "no_match").sum()
        high    = (links_df["link_confidence"] == "HIGH").sum()
        review  = links_df["needs_review"].sum()
        logger.info(
            f"Coverage: {matched}/{total} programs linked "
            f"({high} HIGH confidence, {review} need review)"
        )

    logger.info("=== Phase 6B complete ===")
    return links_path, backfill_path, context_path
