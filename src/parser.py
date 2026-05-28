"""
parser.py — CNC program parser for Proven Program Engine.

Extracts T codes (tool changes), S values (spindle speed), and F values
(feedrate) from proven CNC programs.

Rules:
  - One output record per line that contains at least one S or F value.
  - Records inherit the most-recently-seen T code, speed mode (G96/G97),
    and feed mode (G94/G95).
  - Comments (parentheses and semicolons) are stripped before extraction.
  - G4/G04 dwell lines: F value is dwell time — never extracted as feedrate.
  - G92 S lines: spindle speed limit — extracted but tagged s_type=LIMIT.
  - Block-skip lines (starting with '/') are extracted and flagged.
  - No values are inferred or fabricated.

READ-ONLY — this module never modifies source files.
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger, read_file_lines

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# T code: T followed by 1–4 digits, not preceded/followed by another letter/digit.
# Handles T0101, T01, T1, T0101M06 (stops before M).
_T_CODE_RE = re.compile(r"(?<![A-Za-z])T(\d{1,4})(?!\d)", re.IGNORECASE)

# S value: S followed by digits (optionally decimal). Avoids matching mid-word.
_S_VALUE_RE = re.compile(r"(?<![A-Za-z])S(\d+(?:\.\d+)?)", re.IGNORECASE)

# F value: F followed by a number (integer or decimal, leading zero optional).
# Matches F0.012, F.012, F12, F1.5, F114.
_F_VALUE_RE = re.compile(r"(?<![A-Za-z])F(\d+\.?\d*|\d*\.\d+)", re.IGNORECASE)

# Speed mode
_G96_RE = re.compile(r"\bG96\b", re.IGNORECASE)   # CSS — constant surface speed
_G97_RE = re.compile(r"\bG97\b", re.IGNORECASE)   # RPM — direct spindle RPM

# Feed mode
_G94_RE = re.compile(r"\bG94\b", re.IGNORECASE)   # feed per minute (IPM / mm/min)
_G95_RE = re.compile(r"\bG95\b", re.IGNORECASE)   # feed per revolution (IPR / mm/rev)

# Spindle speed limit command — S here is a clamp, not a cutting speed
_G92_RE = re.compile(r"\bG92\b", re.IGNORECASE)

# Dwell command — F on these lines is a time value, not a feedrate
_G4_RE = re.compile(r"\bG0?4\b", re.IGNORECASE)

# Block skip: optional block marker at start of line
_BLOCK_SKIP_RE = re.compile(r"^/")

# Comment patterns
_PAREN_COMMENT_RE = re.compile(r"\([^)]*\)")
_SEMI_COMMENT_RE = re.compile(r";.*$")

_CONTEXT_WINDOW = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_comments(line: str) -> str:
    """Remove parenthetical ( ) and semicolon comments from a CNC line."""
    line = _PAREN_COMMENT_RE.sub("", line)
    line = _SEMI_COMMENT_RE.sub("", line)
    return line.strip()


def _extract_comment(line: str) -> str:
    """Return the text of the first parenthetical comment in a line, or ''."""
    m = _PAREN_COMMENT_RE.search(line)
    return m.group(0)[1:-1].strip() if m else ""


def _extract_tool_number(t_code: str) -> str:
    """Derive a normalized tool number from a raw T code.

    T0101 → '01'   (4-digit Fanuc: first two digits are the tool slot)
    T0202 → '02'
    T01   → '01'
    T1    → '1'    (single digit: returned as-is)
    """
    m = re.search(r"\d+", t_code)
    if not m:
        return ""
    d = m.group(0)
    return d[:2] if len(d) == 4 else d


def _get_context(lines: list[str], center: int, window: int = _CONTEXT_WINDOW) -> list[str]:
    """Return annotated context lines surrounding a center index.

    Each line is prefixed with '>>>' (target) or '   ' (context) and a
    1-based line number so the source location can be recovered exactly.
    """
    start = max(0, center - window)
    end = min(len(lines), center + window + 1)
    result = []
    for i in range(start, end):
        marker = ">>>" if i == center else "   "
        result.append(f"{marker} L{i + 1:05d}: {lines[i].rstrip()}")
    return result


def _resolve_machine_folder(file_path: Path) -> str:
    """Walk the path to find the machine folder (the parent of the Proven folder)."""
    parts = file_path.parts
    for i, part in enumerate(parts):
        if re.match(r"^proven$", part, re.IGNORECASE) and i > 0:
            return parts[i - 1]
    return file_path.parent.parent.name


def _find_tool_description(lines: list[str], t_code_idx: int, same_line_comment: str) -> str:
    """Resolve the best available tool description for a T code at t_code_idx.

    Priority:
      1. Comment on the T code line itself.
      2. Nearest preceding pure-comment line (up to 4 lines back).
    Stops looking back if it hits a non-comment, non-empty code line.
    """
    if same_line_comment:
        return same_line_comment

    for look_back in range(1, 5):
        back_idx = t_code_idx - look_back
        if back_idx < 0:
            break
        back_raw = lines[back_idx].strip()
        if not back_raw:
            continue
        back_code = _strip_comments(back_raw)
        back_comment = _extract_comment(back_raw)
        if not back_code and back_comment:
            return back_comment
        # Hit a line with actual code — stop looking back
        if back_code:
            break

    return ""


def _score_confidence(
    active_t_code: str,
    s_val: float | None,
    f_val: float | None,
    has_spindle_mode_on_line: bool,
) -> str:
    """Return HIGH / MEDIUM / LOW extraction confidence.

    HIGH  — has active tool AND (explicit G96/G97 with S on this line,
              OR both S and F present on the same line)
    MEDIUM — has active tool AND at least one of S or F
    LOW   — no active tool at time of extraction
    """
    if not active_t_code:
        return "LOW"
    if s_val is not None and has_spindle_mode_on_line:
        return "HIGH"
    if s_val is not None and f_val is not None:
        return "HIGH"
    return "MEDIUM"


def _annotate_duplicates(records: list[dict]) -> None:
    """Add sf_combo_count and is_duplicate fields to each record in-place.

    A combo is (tool_number, rounded_s_value, rounded_f_value).
    is_duplicate = True for the second and later occurrences of a combo.
    sf_combo_count = total occurrences of that combo across the full batch.
    """
    def _combo_key(r: dict) -> tuple:
        t = r.get("tool_number", "")
        s = round(r["s_value"], 4) if r.get("s_value") is not None else None
        f = round(r["f_value"], 6) if r.get("f_value") is not None else None
        return (t, s, f)

    counts: Counter = Counter(_combo_key(r) for r in records)
    seen: set = set()

    for r in records:
        key = _combo_key(r)
        r["sf_combo_count"] = counts[key]
        r["is_duplicate"] = key in seen
        seen.add(key)


# ---------------------------------------------------------------------------
# Single-file parser
# ---------------------------------------------------------------------------

def parse_file(file_path: Path, program_id: int | None = None) -> list[dict]:
    """Parse one CNC file and return a list of speed/feed extraction records.

    Each record represents one line containing at least one S or F value.
    Parser state (active tool, speed mode, feed mode) persists across lines.
    """
    lines = read_file_lines(file_path)
    if lines is None:
        logger.warning(f"Skipping unreadable file: {file_path}")
        return []

    machine_folder = _resolve_machine_folder(file_path)
    source_file = str(file_path)
    filename = file_path.name

    # Mutable parser state
    active_t_code: str = ""
    tool_number: str = ""
    tool_description: str = ""
    current_s_mode: str = "UNKNOWN"
    current_f_mode: str = "UNKNOWN"
    lines_since_t: int = -1   # -1 = no T code seen yet in this file

    records: list[dict] = []
    record_id = 0

    logger.debug(f"Parsing: {filename}  ({len(lines)} lines)")

    for i, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            if lines_since_t >= 0:
                lines_since_t += 1
            continue

        block_skip = bool(_BLOCK_SKIP_RE.match(stripped))

        # Strip the optional-block marker before further parsing
        parse_line = stripped.lstrip("/").strip() if block_skip else stripped

        code_part = _strip_comments(parse_line)
        comment_part = _extract_comment(parse_line)

        # -- Tool change --
        t_match = _T_CODE_RE.search(code_part)
        if t_match:
            active_t_code = t_match.group(0).upper()
            tool_number = _extract_tool_number(active_t_code)
            tool_description = _find_tool_description(lines, i, comment_part)
            lines_since_t = 0
            logger.debug(f"  L{i+1:05d} Tool → {active_t_code}  desc='{tool_description}'")
        elif lines_since_t >= 0:
            lines_since_t += 1

        # -- Speed mode --
        if _G96_RE.search(code_part):
            current_s_mode = "CSS"
        elif _G97_RE.search(code_part):
            current_s_mode = "RPM"

        # -- Feed mode --
        if _G94_RE.search(code_part):
            current_f_mode = "IPM"
        elif _G95_RE.search(code_part):
            current_f_mode = "IPR"

        # -- Dwell: skip F extraction entirely on G4/G04 lines --
        is_dwell = bool(_G4_RE.search(code_part))

        # -- Spindle limit: G92 S sets a clamp, not a cutting speed --
        is_spindle_limit = bool(_G92_RE.search(code_part))

        # -- Extract values --
        s_matches = _S_VALUE_RE.findall(code_part)
        f_matches = [] if is_dwell else _F_VALUE_RE.findall(code_part)

        if not s_matches and not f_matches:
            continue

        s_val = float(s_matches[0]) if s_matches else None
        f_val = float(f_matches[0]) if f_matches else None

        # Determine per-record metadata
        s_type = ""
        if s_val is not None:
            s_type = "LIMIT" if is_spindle_limit else "SPINDLE"

        s_mode_for_record = current_s_mode if s_val is not None else ""
        f_mode_for_record = current_f_mode if f_val is not None else ""

        has_spindle_mode_on_line = bool(
            _G96_RE.search(code_part) or _G97_RE.search(code_part)
        )
        confidence = _score_confidence(active_t_code, s_val, f_val, has_spindle_mode_on_line)

        prev_line = lines[i - 1].rstrip() if i > 0 else ""
        next_line = lines[i + 1].rstrip() if i < len(lines) - 1 else ""

        context = _get_context(lines, i)
        record_id += 1

        records.append({
            "record_id": record_id,
            "program_id": program_id,
            "source_file": source_file,
            "machine_folder": machine_folder,
            "filename": filename,
            "line_number": i + 1,
            "active_t_code": active_t_code,
            "tool_number": tool_number,
            "tool_description": tool_description,
            "s_value": s_val,
            "s_mode": s_mode_for_record,
            "s_type": s_type,
            "f_value": f_val,
            "f_mode": f_mode_for_record,
            "block_skip": block_skip,
            "lines_since_t_code": lines_since_t,
            "extraction_confidence": confidence,
            "raw_line": stripped,
            "prev_line": prev_line,
            "next_line": next_line,
            "context_json": json.dumps(context),
        })

    logger.debug(f"  → {len(records)} records from {filename}")
    return records


# ---------------------------------------------------------------------------
# Batch parser (reads manifest CSV)
# ---------------------------------------------------------------------------

def parse_from_manifest(
    manifest_csv: Path,
    exports_dir: Path,
) -> dict[str, Path]:
    """Parse all 'included=True' files from a manifest CSV.

    Writes three outputs to exports_dir:
      cuts_*.csv        — one record per extracted S/F line
      parser_summary_*.json — aggregate metrics
      tool_summary_*.csv    — grouped stats per tool/machine/mode

    Returns a dict with keys 'cuts', 'summary', 'tool_summary'.
    """
    from .exports import export_parser_summary, export_tool_summary

    df_manifest = pd.read_csv(manifest_csv)
    included = df_manifest[df_manifest["included"] == True]

    logger.info(f"Manifest: {len(df_manifest)} total files, {len(included)} included")

    all_records: list[dict] = []
    parsed_count = 0
    error_count = 0

    for _, row in included.iterrows():
        file_path = Path(row["source_file"])
        if not file_path.exists():
            logger.warning(f"File not found (may have moved or been deleted): {file_path}")
            error_count += 1
            continue

        pid = int(row["program_id"]) if pd.notna(row["program_id"]) else None
        recs = parse_file(file_path, program_id=pid)
        all_records.extend(recs)
        parsed_count += 1

    logger.info(
        f"Parsing complete: {parsed_count} files parsed | "
        f"{error_count} not found | {len(all_records)} records extracted"
    )

    # Annotate duplicates across the full batch
    _annotate_duplicates(all_records)

    assert_safe_write(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- cuts CSV ---
    cuts_path = exports_dir / f"cuts_{timestamp}.csv"
    df_out = pd.DataFrame(all_records)
    assert_safe_write(cuts_path)
    df_out.to_csv(cuts_path, index=False)
    logger.info(f"Cuts CSV       → {cuts_path}  ({len(df_out)} rows)")

    # --- parser summary JSON ---
    summary_path = export_parser_summary(all_records, exports_dir, timestamp)

    # --- tool summary CSV ---
    tool_summary_path = export_tool_summary(all_records, exports_dir, timestamp)

    return {
        "cuts": cuts_path,
        "summary": summary_path,
        "tool_summary": tool_summary_path,
    }
