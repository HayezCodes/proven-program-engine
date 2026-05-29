"""
proven_sf_database.py — Final proven speeds & feeds database export.

Joins data from all prior phases into one clean, queryable table:
  cuts_*.csv              — proven S/F values parsed from CNC programs
  program_job_links_*.csv — verified material from job/router metadata
  tooldb_reference_*.csv  — tool identity (TOOLDB; S/F NOT used)
  tooling_review_*.csv    — program + reference tool descriptions
  material_candidates_*.csv — inferred material candidates from S/F matching
  router_program_context_*.csv — router operation context per program

Material priority (highest → lowest):
  1. verified_material from router/job metadata (ROUTER / SHARED_PRINT)
  2. material_candidate_1 from shop S/F reference matching (INFERRED)
  3. UNKNOWN

TOOLDB S/F (feed_rate_ipm / spindle_rpm) are NEVER used as proven values.
G92 spindle-limit records are included but always flagged for review.

READ-ONLY against P:\\, G:\\, M:\\. Exports only to local exports/ folder.
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger
from .dashboard.data_access.tool_identity import resolve_tool_identity_df
from .proven_sf_lookup import build_sf_lookup, export_sf_lookup

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output column schemas
# ---------------------------------------------------------------------------

SF_DB_COLS = [
    "machine_folder",
    "machine_family",
    "source_file",
    "filename",
    "program_id",
    "tool_number",
    "resolved_tool_name",
    "resolved_tool_description",
    "tool_identity_source",
    "tool_needs_review",
    "S",
    "s_mode",
    "s_type",
    "F",
    "f_mode",
    "feed_intent_candidate",
    "verified_material",
    "material_source",
    "material_confidence",
    "material_candidate_1",
    "material_candidate_confidence_1",
    "matched_job_number",
    "matched_part_number",
    "matched_drawing_number",
    "linked_router_file",
    "router_work_center",
    "router_operation_description",
    "link_confidence",
    "link_method",
    "extraction_confidence",
    "sf_record_confidence",
    "needs_review",
    "review_reason",
    "raw_line",
    "prev_line",
    "next_line",
]

SF_SUMMARY_COLS = [
    "machine_folder",
    "final_material",
    "resolved_tool_name",
    "resolved_tool_description",
    "tool_number",
    "s_mode",
    "f_mode",
    "feed_intent_candidate",
    "occurrence_count",
    "program_count",
    "job_count",
    "S_min",
    "S_avg",
    "S_max",
    "F_min",
    "F_avg",
    "F_max",
    "confidence_mix",
    "needs_review_count",
]

SF_PROGRAMMER_COLS = [
    "material",
    "machine_folder",
    "machine_family",
    "tool_number",
    "resolved_tool_name",
    "resolved_tool_description",
    "tool_identity_source",
    "S_min",
    "S_avg",
    "S_max",
    "s_mode",
    "F_min",
    "F_avg",
    "F_max",
    "f_mode",
    "feed_intent_candidate",
    "occurrence_count",
    "program_count",
    "confidence_mix",
    "needs_review_count",
]

_MACHINE_ID_RE = re.compile(r"(\d{3,4})")

# ---------------------------------------------------------------------------
# Auto-detection of latest exports
# ---------------------------------------------------------------------------

def detect_latest_exports(exports_dir: Path) -> dict[str, Path | None]:
    """Return paths to the latest of each required export file type."""
    def _latest(pattern: str) -> Path | None:
        try:
            files = sorted(exports_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
            return files[-1] if files else None
        except OSError:
            return None

    return {
        "cuts":           _latest("cuts_*.csv"),
        "links":          _latest("program_job_links_*.csv"),
        "tooldb":         _latest("tooldb_reference_*.csv"),
        "tooling_review": _latest("tooling_review_*.csv"),
        "mat_candidates": _latest("material_candidates_*.csv"),
        "router_context": _latest("router_program_context_*.csv"),
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path | None, label: str) -> pd.DataFrame:
    """Load a CSV to DataFrame; return empty DataFrame on missing/empty file."""
    if path is None or not path.exists():
        logger.warning(f"{label}: not found — using empty DataFrame")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Loaded {label}: {path.name}  ({len(df)} rows)")
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"{label}: empty file — using empty DataFrame")
        return pd.DataFrame()
    except Exception as exc:
        logger.warning(f"{label}: failed to load ({exc}) — using empty DataFrame")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_tn(val) -> str:
    """Normalise tool number to plain integer string ('1.0' → '1', NaN → '')."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _safe_str(val, default: str = "") -> str:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none") else default


def _machine_family(folder: str) -> str:
    """Extract leading 3-4 digit machine ID from a folder name."""
    m = _MACHINE_ID_RE.search(str(folder).strip())
    return m.group(1) if m else str(folder).strip()


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def score_sf_record_confidence(row) -> str:
    """Assign HIGH / MEDIUM / LOW to one S/F database record.

    HIGH:  verified material from router/job  +  resolved tool  +
           MEDIUM/HIGH parser confidence  +  not a spindle-limit record  +
           HIGH/MEDIUM link confidence.
    MEDIUM: some material context (verified or candidate)  +  resolved tool  +
            MEDIUM/HIGH parser confidence.
    LOW:   anything else.
    """
    verified   = _safe_str(row.get("verified_material")) not in ("", "UNKNOWN")
    mat_src    = _safe_str(row.get("material_source"))
    good_mat   = verified and mat_src in ("ROUTER", "SHARED_PRINT")

    tool_src   = _safe_str(row.get("tool_identity_source"))
    tool_known = tool_src not in ("", "UNKNOWN")

    parser_conf = _safe_str(row.get("extraction_confidence"))
    good_parser = parser_conf in ("MEDIUM", "HIGH")

    s_type      = _safe_str(row.get("s_type"))
    not_limit   = s_type != "LIMIT"

    link_conf   = _safe_str(row.get("link_confidence"))
    good_link   = link_conf in ("HIGH", "MEDIUM")

    if good_mat and tool_known and good_parser and not_limit and good_link:
        return "HIGH"

    has_any_mat = verified or bool(_safe_str(row.get("material_candidate_1")))
    if has_any_mat and tool_known and good_parser:
        return "MEDIUM"

    return "LOW"


def compute_needs_review(row) -> tuple[bool, str]:
    """Return (needs_review, comma-separated reason string) for one record."""
    reasons: list[str] = []

    if _safe_str(row.get("verified_material")) in ("", "UNKNOWN"):
        reasons.append("no_verified_material")

    mat_conf = _safe_str(row.get("material_confidence"))
    if mat_conf in ("LOW", "NONE", ""):
        reasons.append("low_material_confidence")

    tool_src = _safe_str(row.get("tool_identity_source"))
    if tool_src in ("", "UNKNOWN"):
        reasons.append("unknown_tool_identity")

    tnr = row.get("tool_needs_review")
    if tnr is True or str(tnr).lower() == "true":
        reasons.append("tool_needs_review")

    if _safe_str(row.get("extraction_confidence")) == "LOW":
        reasons.append("low_extraction_confidence")

    if _safe_str(row.get("s_type")) == "LIMIT":
        reasons.append("spindle_limit_record")

    lc = _safe_str(row.get("link_confidence"))
    if lc in ("LOW", "NONE", ""):
        reasons.append("low_link_confidence")

    conflict = row.get("material_conflict")
    if conflict is True or str(conflict).lower() == "true":
        reasons.append("material_conflict")

    return bool(reasons), ", ".join(reasons)


# ---------------------------------------------------------------------------
# Main database builder
# ---------------------------------------------------------------------------

def build_sf_database(
    cuts_df:      pd.DataFrame,
    links_df:     pd.DataFrame,
    tooldb_df:    pd.DataFrame,
    tooling_df:   pd.DataFrame,
    mat_cands_df: pd.DataFrame,
    router_df:    pd.DataFrame,
) -> pd.DataFrame:
    """Build the proven S/F database from all input DataFrames.

    Returns a DataFrame with SF_DB_COLS columns.  All joins are LEFT JOINs
    so every cut record appears regardless of available metadata.
    """
    if cuts_df.empty:
        logger.warning("cuts_df is empty — returning empty SF database")
        return pd.DataFrame(columns=SF_DB_COLS)

    df = cuts_df.copy()

    # Rename S/F value columns to clean output names
    if "s_value" in df.columns:
        df.rename(columns={"s_value": "S"}, inplace=True)
    if "f_value" in df.columns:
        df.rename(columns={"f_value": "F"}, inplace=True)

    # Rename tool_description → program_description for tool identity resolver
    if "tool_description" in df.columns:
        df.rename(columns={"tool_description": "program_description"}, inplace=True)

    # Add machine_family
    df["machine_family"] = df["machine_folder"].apply(_machine_family)

    # Normalise tool_number for reliable joins
    df["_tn"] = df["tool_number"].apply(_norm_tn)

    # -----------------------------------------------------------------------
    # JOIN 1: program_job_links → verified material + link metadata
    # One link row per program (source_file), applied to all its cut records.
    # -----------------------------------------------------------------------
    if not links_df.empty:
        link_keep = [
            "source_file", "matched_job_number", "matched_part_number",
            "matched_drawing_number", "matched_router_file",
            "link_confidence", "link_method", "link_reason",
            "material", "material_source", "material_confidence",
            "material_conflict",
        ]
        link_avail = [c for c in link_keep if c in links_df.columns]
        lsub = links_df[link_avail].drop_duplicates("source_file").rename(
            columns={"material": "verified_material",
                     "matched_router_file": "linked_router_file"}
        )
        df = df.merge(lsub, on="source_file", how="left")
    else:
        for col in ("verified_material", "material_source", "material_confidence",
                    "linked_router_file", "matched_job_number", "matched_part_number",
                    "matched_drawing_number", "link_confidence", "link_method",
                    "link_reason", "material_conflict"):
            df[col] = ""

    # Fill NaN strings for verified_material
    df["verified_material"] = df["verified_material"].fillna("UNKNOWN").replace("", "UNKNOWN")

    # -----------------------------------------------------------------------
    # JOIN 2: material_candidates → inferred material candidates
    # Grouped by (machine_folder, active_t_code, tool_number, s_mode).
    # -----------------------------------------------------------------------
    if not mat_cands_df.empty:
        mc_key = ["machine_folder", "active_t_code", "tool_number", "s_mode"]
        mc_avail = [c for c in mc_key if c in mat_cands_df.columns]
        mc_keep_extra = [
            "material_candidate_1",
            "confidence_label",
            "feed_intent_candidate",
            "feed_intent_confidence",
        ]
        mc_keep = mc_avail + [c for c in mc_keep_extra if c in mat_cands_df.columns]
        mcsub = mat_cands_df[mc_keep].drop_duplicates(mc_avail).rename(
            columns={"confidence_label": "material_candidate_confidence_1"}
        )
        mcsub["_tn_mc"] = mcsub["tool_number"].apply(_norm_tn) if "tool_number" in mcsub.columns else ""
        # Normalise tool_number in mcsub for joining
        if "tool_number" in mcsub.columns:
            mcsub = mcsub.drop(columns=["tool_number"])
        mcsub = mcsub.rename(columns={"_tn_mc": "_tn"})

        mc_join_keys = [k for k in ["machine_folder", "active_t_code", "_tn", "s_mode"]
                        if k in mcsub.columns and k in df.columns]
        if mc_join_keys:
            df = df.merge(mcsub, on=mc_join_keys, how="left")

    # Ensure material_candidate_1 exists
    for col in ("material_candidate_1", "material_candidate_confidence_1",
                "feed_intent_candidate", "feed_intent_confidence"):
        if col not in df.columns:
            df[col] = ""

    # -----------------------------------------------------------------------
    # JOIN 3: tooling_review → reference / corrected descriptions
    # Keyed on (machine_folder, tool_number).
    # -----------------------------------------------------------------------
    if not tooling_df.empty:
        tr_keep = ["machine_folder", "tool_number", "reference_description",
                   "corrected_description", "notes"]
        tr_avail = [c for c in tr_keep if c in tooling_df.columns]
        trsub = tooling_df[tr_avail].copy()
        trsub["_tn"] = trsub["tool_number"].apply(_norm_tn)
        trsub = trsub.drop(columns=["tool_number"])
        df = df.merge(
            trsub.drop_duplicates(["machine_folder", "_tn"]),
            on=["machine_folder", "_tn"],
            how="left",
        )

    # Ensure these exist for resolve_tool_identity_df
    for col in ("reference_description", "corrected_description", "notes"):
        if col not in df.columns:
            df[col] = ""

    # -----------------------------------------------------------------------
    # JOIN 4: router_program_context → work center + operation description
    # Take first operation per source_file.
    # -----------------------------------------------------------------------
    if not router_df.empty and "source_file" in router_df.columns:
        r_keep = ["source_file", "work_center", "operation_description"]
        r_avail = [c for c in r_keep if c in router_df.columns]
        rfirst = (
            router_df[r_avail]
            .drop_duplicates("source_file")
            .rename(columns={
                "work_center":          "router_work_center",
                "operation_description":"router_operation_description",
            })
        )
        df = df.merge(rfirst, on="source_file", how="left")

    for col in ("router_work_center", "router_operation_description"):
        if col not in df.columns:
            df[col] = ""

    # -----------------------------------------------------------------------
    # TOOL IDENTITY RESOLUTION
    # resolve_tool_identity_df adds: resolved_tool_name, resolved_tool_description,
    # resolved_tool_source, resolved_tool_confidence, resolved_tool_needs_review
    # -----------------------------------------------------------------------
    df = resolve_tool_identity_df(df, tooldb_ref=tooldb_df if not tooldb_df.empty else None)

    # Rename to output column names
    rename_map = {}
    if "resolved_tool_source" in df.columns:
        rename_map["resolved_tool_source"] = "tool_identity_source"
    if "resolved_tool_needs_review" in df.columns:
        rename_map["resolved_tool_needs_review"] = "tool_needs_review"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Ensure resolved tool columns exist
    for col in ("resolved_tool_name", "resolved_tool_description",
                "tool_identity_source", "tool_needs_review"):
        if col not in df.columns:
            df[col] = "UNKNOWN" if "source" in col or "name" in col else (
                True if "review" in col else ""
            )

    # -----------------------------------------------------------------------
    # CONFIDENCE SCORING AND REVIEW FLAGS
    # -----------------------------------------------------------------------
    df["sf_record_confidence"] = df.apply(score_sf_record_confidence, axis=1)

    review_results = df.apply(compute_needs_review, axis=1)
    df["needs_review"]  = [r[0] for r in review_results]
    df["review_reason"] = [r[1] for r in review_results]

    # -----------------------------------------------------------------------
    # DROP internal helper columns; select output columns
    # -----------------------------------------------------------------------
    df.drop(columns=[c for c in ("_tn",) if c in df.columns], inplace=True)

    # Ensure all required output columns exist (fill missing with empty)
    for col in SF_DB_COLS:
        if col not in df.columns:
            df[col] = ""

    return df[SF_DB_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_sf_summary(sf_db_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the SF database by tool / material / mode.

    final_material priority: verified_material → material_candidate_1 → UNKNOWN.
    """
    if sf_db_df.empty:
        return pd.DataFrame(columns=SF_SUMMARY_COLS)

    df = sf_db_df.copy()

    # Build final_material
    def _final(row) -> str:
        vm = _safe_str(row.get("verified_material"))
        if vm and vm != "UNKNOWN":
            return vm
        mc = _safe_str(row.get("material_candidate_1"))
        return mc if mc else "UNKNOWN"

    df["final_material"] = df.apply(_final, axis=1)

    # Only rows with at least one S or F value
    has_sf = df["S"].notna() | df["F"].notna()
    df = df[has_sf]

    if df.empty:
        return pd.DataFrame(columns=SF_SUMMARY_COLS)

    group_cols = [
        "machine_folder", "final_material", "resolved_tool_name",
        "resolved_tool_description", "tool_number", "s_mode", "f_mode",
        "feed_intent_candidate",
    ]
    # Fill NaN in group keys with empty string to avoid groupby drop
    for gc in group_cols:
        if gc in df.columns:
            df[gc] = df[gc].fillna("").astype(str)

    rows: list[dict] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        s_data = pd.to_numeric(grp["S"], errors="coerce").dropna()
        f_data = pd.to_numeric(grp["F"], errors="coerce").dropna()

        prog_count = grp["source_file"].nunique()
        job_nums   = grp["matched_job_number"].dropna().astype(str)
        job_count  = job_nums[job_nums.str.strip().ne("")].nunique()

        conf_dist  = grp["sf_record_confidence"].value_counts()
        conf_mix   = " | ".join(f"{k}:{v}" for k, v in conf_dist.items())

        row: dict = dict(zip(group_cols, keys))
        row.update({
            "occurrence_count":  len(grp),
            "program_count":     prog_count,
            "job_count":         job_count,
            "S_min":  round(s_data.min(),  2) if len(s_data) else None,
            "S_avg":  round(s_data.mean(), 2) if len(s_data) else None,
            "S_max":  round(s_data.max(),  2) if len(s_data) else None,
            "F_min":  round(f_data.min(),  6) if len(f_data) else None,
            "F_avg":  round(f_data.mean(), 6) if len(f_data) else None,
            "F_max":  round(f_data.max(),  6) if len(f_data) else None,
            "confidence_mix":    conf_mix,
            "needs_review_count": int((grp["needs_review"] == True).sum()),
        })
        rows.append(row)

    result = pd.DataFrame(rows, columns=SF_SUMMARY_COLS)
    result.sort_values("occurrence_count", ascending=False, inplace=True)
    return result.reset_index(drop=True)


def _final_material(row) -> str:
    """Return programmer-facing material using verified material first."""
    vm = _safe_str(row.get("verified_material"))
    if vm and vm != "UNKNOWN":
        return vm
    mc = _safe_str(row.get("material_candidate_1"))
    return mc if mc else "UNKNOWN"


def _confidence_mix(series: pd.Series) -> str:
    conf_dist = series.value_counts()
    return " | ".join(f"{k}:{v}" for k, v in conf_dist.items())


def build_programmer_view(sf_db_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the full S/F database into the clean programmer-facing view."""
    if sf_db_df.empty:
        return pd.DataFrame(columns=SF_PROGRAMMER_COLS)

    df = sf_db_df.copy()
    df["material"] = df.apply(_final_material, axis=1)

    has_sf = df["S"].notna() | df["F"].notna()
    df = df[has_sf]
    if df.empty:
        return pd.DataFrame(columns=SF_PROGRAMMER_COLS)

    group_cols = [
        "material",
        "machine_folder",
        "machine_family",
        "tool_number",
        "resolved_tool_name",
        "resolved_tool_description",
        "s_mode",
        "f_mode",
        "feed_intent_candidate",
    ]
    for gc in group_cols:
        if gc not in df.columns:
            df[gc] = ""
        df[gc] = df[gc].fillna("").astype(str)

    rows: list[dict] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        s_data = pd.to_numeric(grp["S"], errors="coerce").dropna()
        f_data = pd.to_numeric(grp["F"], errors="coerce").dropna()
        tool_sources = (
            grp.get("tool_identity_source", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
        )
        tool_sources = tool_sources[tool_sources.ne("")]

        row: dict = dict(zip(group_cols, keys))
        row.update({
            "tool_identity_source": " | ".join(sorted(tool_sources.unique())),
            "S_min": round(s_data.min(), 2) if len(s_data) else None,
            "S_avg": round(s_data.mean(), 2) if len(s_data) else None,
            "S_max": round(s_data.max(), 2) if len(s_data) else None,
            "F_min": round(f_data.min(), 6) if len(f_data) else None,
            "F_avg": round(f_data.mean(), 6) if len(f_data) else None,
            "F_max": round(f_data.max(), 6) if len(f_data) else None,
            "occurrence_count": len(grp),
            "program_count": grp["source_file"].nunique() if "source_file" in grp.columns else 0,
            "confidence_mix": _confidence_mix(grp["sf_record_confidence"])
                              if "sf_record_confidence" in grp.columns else "",
            "needs_review_count": int((grp["needs_review"] == True).sum())
                                  if "needs_review" in grp.columns else 0,
        })
        rows.append(row)

    result = pd.DataFrame(rows, columns=SF_PROGRAMMER_COLS)
    result.sort_values("occurrence_count", ascending=False, inplace=True)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def export_sf_database(
    df: pd.DataFrame, exports_dir: Path, timestamp: str
) -> Path:
    out = exports_dir / f"proven_sf_database_{timestamp}.csv"
    assert_safe_write(out)
    (pd.DataFrame(columns=SF_DB_COLS) if df.empty else df).to_csv(out, index=False)
    logger.info(f"SF database -> {out}  ({len(df)} row(s))")
    return out


def export_sf_summary(
    df: pd.DataFrame, exports_dir: Path, timestamp: str
) -> Path:
    out = exports_dir / f"proven_sf_summary_{timestamp}.csv"
    assert_safe_write(out)
    (pd.DataFrame(columns=SF_SUMMARY_COLS) if df.empty else df).to_csv(out, index=False)
    logger.info(f"SF summary  -> {out}  ({len(df)} row(s))")
    return out


def export_programmer_view(
    df: pd.DataFrame, exports_dir: Path, timestamp: str
) -> Path:
    out = exports_dir / f"proven_sf_programmer_view_{timestamp}.csv"
    assert_safe_write(out)
    (pd.DataFrame(columns=SF_PROGRAMMER_COLS) if df.empty else df).to_csv(out, index=False)
    logger.info(f"Programmer S/F view -> {out}  ({len(df)} row(s))")
    return out


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_build_sf_database(
    exports_dir:       Path | None = None,
    cuts_path:         Path | None = None,
    links_path:        Path | None = None,
    tooldb_path:       Path | None = None,
    tooling_path:      Path | None = None,
    mat_cands_path:    Path | None = None,
    router_ctx_path:   Path | None = None,
) -> tuple[Path, Path, Path]:
    """Build and export the proven S/F database from latest phase outputs.

    Auto-detects latest export files when explicit paths are not supplied.
    Returns (sf_database_path, sf_summary_path, programmer_view_path).
    Never writes to production folders. Never overwrites existing exports.
    """
    if exports_dir is None:
        exports_dir = Path(__file__).parent.parent / "exports"
    assert_safe_write(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    detected = detect_latest_exports(exports_dir)

    def _r(explicit: Path | None, key: str, label: str) -> pd.DataFrame:
        return _load_csv(explicit or detected.get(key), label)

    cuts_df      = _r(cuts_path,       "cuts",         "cuts")
    links_df     = _r(links_path,      "links",        "program_job_links")
    tooldb_df    = _r(tooldb_path,     "tooldb",       "tooldb_reference")
    tooling_df   = _r(tooling_path,    "tooling_review", "tooling_review")
    mat_cands_df = _r(mat_cands_path,  "mat_candidates", "material_candidates")
    router_df    = _r(router_ctx_path, "router_context", "router_program_context")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== Proven S/F Database Builder ===")
    logger.info(f"Exports dir : {exports_dir}")

    sf_db   = build_sf_database(cuts_df, links_df, tooldb_df, tooling_df,
                                 mat_cands_df, router_df)
    sf_summ = build_sf_summary(sf_db)
    programmer_view = build_programmer_view(sf_db)
    sf_lookup = build_sf_lookup(sf_db)

    db_path    = export_sf_database(sf_db,   exports_dir, timestamp)
    summ_path  = export_sf_summary(sf_summ,  exports_dir, timestamp)
    prog_path  = export_programmer_view(programmer_view, exports_dir, timestamp)
    lookup_path = export_sf_lookup(sf_lookup, exports_dir, timestamp)

    # Coverage log
    if not sf_db.empty:
        total       = len(sf_db)
        high_conf   = (sf_db["sf_record_confidence"] == "HIGH").sum()
        med_conf    = (sf_db["sf_record_confidence"] == "MEDIUM").sum()
        verified    = (sf_db["verified_material"] != "UNKNOWN").sum()
        needs_rev   = sf_db["needs_review"].sum()
        logger.info(
            f"Records: {total}  |  "
            f"HIGH conf: {high_conf}  MEDIUM: {med_conf}  |  "
            f"Verified material: {verified} ({100*verified//total}%)  |  "
            f"Needs review: {needs_rev}"
        )

    logger.info(f"Lookup groups  : {len(sf_lookup)}")
    logger.info("=== SF database build complete ===")
    return db_path, summ_path, prog_path, lookup_path
