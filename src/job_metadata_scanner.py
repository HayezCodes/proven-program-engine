"""
job_metadata_scanner.py — Read-only scanner for job folders and shared part prints.

Phase 6A-Optimize: intelligent document classification and prioritization
before full extraction.  Pipeline per file:
  1. classify_filename  → fast, free (string ops only)
  2. sample_pdf_first_page → one page only, pre-classification
  3. classify_content   → deterministic keyword matching
  4. decide_parse_action → full_parse / shallow_index / skipped_*
  5. extract (or skip)

Scans:
  G:\\Manufacturing\\JOB FOLDERS\\2024 Orders
  G:\\Manufacturing\\Programming\\CAM Files\\Shared Part Prints

Exports timestamped CSVs to exports/. Never writes to source folders.
"""

import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger, read_file_lines

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Source roots (READ-ONLY)
# ---------------------------------------------------------------------------

JOB_FOLDERS_ROOT   = Path("G:/Manufacturing/JOB FOLDERS/2024 Orders")
SHARED_PRINTS_ROOT = Path("G:/Manufacturing/Programming/CAM Files/Shared Part Prints")

# ---------------------------------------------------------------------------
# Scannable extensions
# ---------------------------------------------------------------------------

_DEFAULT_LOOKBACK_DAYS = 365

JOB_FOLDER_EXTENSIONS:    frozenset[str] = frozenset({".PDF", ".TXT"})
SHARED_PRINT_EXTENSIONS:  frozenset[str] = frozenset({".PDF", ".TXT"})

# ---------------------------------------------------------------------------
# Extraction regex patterns (deterministic — no inference)
# ---------------------------------------------------------------------------

_JOB_RE = re.compile(
    r"(?:JOB\s*(?:NO|NUMBER|NUM|#)\.?\s*[:\-]?\s*"
    r"|WORK\s+ORDER\s*[#:\-]?\s*"
    r"|W\.?O\.?\s*[#:\-]?\s*"
    r")\s*([A-Z]?[0-9]{4,7}(?:\-[0-9]{1,6})?)",
    re.IGNORECASE,
)

_PART_RE = re.compile(
    r"(?:PART\s*(?:NO|NUMBER|NUM|#)\.?\s*[:\-]?\s*"
    r"|P\.?/?N\.?\s*[:\-]?\s*"
    r")\s*([A-Z0-9][A-Z0-9\-_/\.]{2,29})",
    re.IGNORECASE,
)

_DWG_RE = re.compile(
    r"(?:DWG\.?\s*(?:NO|NUMBER|NUM|#)\.?\s*[:\-]?\s*"
    r"|DRAWING\s*(?:NO|NUMBER|NUM|#)\.?\s*[:\-]?\s*"
    r"|PRINT\s*(?:NO|NUMBER|NUM|#)\.?\s*[:\-]?\s*"
    r")\s*([A-Z0-9][A-Z0-9\-_/\.]{1,29})",
    re.IGNORECASE,
)

_REV_RE = re.compile(
    r"(?:REV(?:ISION)?\.?\s*[:\-]?\s*)([A-Z0-9]{1,3})\b",
    re.IGNORECASE,
)

_MAT_RE = re.compile(
    r"(?:MAT(?:ERIAL|L)?\.?\s*[:\-]?\s*)([^\n\r\|;]{3,60})",
    re.IGNORECASE,
)

_EXPLICIT_MATERIAL_RE = re.compile(
    r"(?im)^\s*Material\s*:\s*([^\n\r]{3,120})$"
)

_MATERIAL_TOKEN_PATTERNS = [
    re.compile(
        r"\b(?:10[124][058]|11[0-9]{2}|12L14|41[034]0|43[34]0|46[24]0|86[24]0|93[124]0)\b"
        r"(?:\s+(?:HR|HT|H\.T\.|CD|CF|CRS|HRS|ANN(?:EALED)?|NORM(?:ALIZED)?|"
        r"PREHARD|ALLOY|STEEL))*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:17-4|15-5)\s*(?:PH)?(?:\s+(?:SS|SST|STAINLESS|STEEL))*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:303|304L?|316L?|321|347|410|416|420|440C)\b"
        r"(?:\s+(?:SS|SST|STAINLESS|STEEL))*",
        re.IGNORECASE,
    ),
    re.compile(r"\bS31803\b(?:\s+DUPLEX)?", re.IGNORECASE),
    re.compile(r"\bHASTELLOY\s*C-?276\b", re.IGNORECASE),
    re.compile(r"\bMONEL\s*K-?500\b", re.IGNORECASE),
    re.compile(r"\bINCONEL\s*\d{3}\b", re.IGNORECASE),
    re.compile(r"\b(?:6061|7075|2024)\b(?:\s+ALUMINUM|\s+ALUM)?", re.IGNORECASE),
]

_MATERIAL_NOISE_RE = re.compile(
    r"\b(?:CUTTING\s+CHARGE|PO\s+#?|PURCHASE\s+ORDER|QUOTE#?|VENDOR|QTY|"
    r"RECEIVED|CERTIFICATION|SIEMENS\s+SPECIFICATION)\b",
    re.IGNORECASE,
)

_OP_ROW_RE   = re.compile(r"^\s*(\d{2,3})\s{2,}(.+)$")
_TRAVELER_OP_RE = re.compile(r"^\s*(\d)\|(\d{2})\s+(\S+)\s+.*$")
_WC_LINE_RE = re.compile(r"^\s*(\d{3})\s+([A-Z][A-Z /-]{2,20})\b(.*)$")
_OP_LABEL_RE = re.compile(
    r"(?:OP(?:ERATION)?\s+)(\d+)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)

_MACHINE_RE = re.compile(
    r"\b(LATHE|MILL|VMC|HMC|TURNING|MILLING|GRIND(?:ING)?|DRILL(?:ING)?"
    r"|BORE|BORING|HONE|BROACH|HEAT\s+TREAT|INSPECT(?:ION)?|DEBURR"
    r"|WASH|BLAST|ANODIZE|PLATE|WELD|PRESS|SAW|BAND\s+SAW"
    r"|HAAS|MAZAK|OKUMA|HURCO|MATSUURA|DOOSAN|CHEVALIER)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Document classification — deterministic keyword matching
# ---------------------------------------------------------------------------

# Filename patterns for HIGH-priority documents (likely travelers / drawings)
_FNAME_HIGH_RE = re.compile(
    r"(?:traveler|router(?:ing)?|setup[\s_\-]sheet|setup[\s_\-]card|empower"
    r"|work[\s_\-]order|operation[\s_\-]sheet|routing[\s_\-]sheet|job[\s_\-]traveler"
    r"|EM\d{3,}"                       # EM drawing numbers: EM10986
    r"|L1D\d{6,}"                      # Boeing-style: L1D30093240A
    r"|(?<![a-zA-Z0-9])D\d{5,}(?!\d))", # D-numbered drawings: D19937
    re.IGNORECASE,
)

# Filename patterns for LOW-priority documents (administrative / shipping / certs).
# Uses [^a-zA-Z] boundaries instead of \b so underscore-separated words match.
# LOW wins over HIGH when both patterns fire.
_FNAME_LOW_RE = re.compile(
    r"(?i)(?:"
    r"bol(?:[^a-zA-Z]|$)"           # BOL, BOL_123 (not "bolts")
    r"|bill[\s_\-]of[\s_\-]lading"
    r"|certif"                       # cert, certif*, certification, certificate
    r"|(?<![a-zA-Z])fai(?:[^a-zA-Z]|$)"  # FAI not embedded in a word
    r"|first[\s_\-]article"
    r"|inspection[\s_\-]report"
    r"|packing[\s_\-](?:slip|list)"
    r"|purchase[\s_\-]order"
    r"|(?:^|[^a-zA-Z0-9])po[_\s]"   # PO_ or standalone "PO " prefix
    r"|invoice(?:[^a-zA-Z]|$)"
    r"|receiving(?:[^a-zA-Z]|$)"
    r"|acknowledgm|acknow"           # acknowledgment / acknowledgement / Acknow abbreviation
    r"|quotation|confirmation"
    r"|shipper(?:[^a-zA-Z]|$)"
    r"|airway[\s_\-]bill"
    r"|delivery[\s_\-](?:order|note)"
    r"|material[\s_\-]cert"
    r"|key[\s_\-]char"
    r"|ballooned"
    r"|conformance|compliance"
    r"|certificate[\s_\-]of"
    r"|quality[\s_\-]notif"
    r")"
)

# Content patterns for HIGH-priority documents (from first-page text sample)
_CONTENT_HIGH_RE = re.compile(
    r"(?:ROUTING\s+COMMENTS|WC\s*/\s*VENDOR|WC\s+VENDOR|WORK\s+CENTER\s+ROUTING"
    r"|EMPOWER\s+(?:MANUFACTURING|MFG)|EMPOWER\s+MACHINE"
    r"|ROUTING\s+SHEET|OPERATION\s+ROUTING|TRAVELER"
    r"|OPERATION\s+\d+\s*[-–]"         # "Operation 10 -" tabular header
    r"|JOB\s*(?:NO|NUMBER|#)\s*:?\s*\d"
    r"|WORK\s+ORDER\s*:?\s*\d)",
    re.IGNORECASE,
)

# Content patterns for LOW-priority documents (administrative documents)
_CONTENT_LOW_RE = re.compile(
    r"(?:BILL\s+OF\s+LADING|CERTIFICATE\s+OF\s+ORIGIN"
    r"|MATERIAL\s+CERTIFICATION|FIRST\s+ARTICLE\s+INSPECTION"
    r"|FIRST\s+ARTICLE\s+REPORT|INSPECTION\s+REPORT"
    r"|PURCHASE\s+ORDER|PACKING\s+(?:LIST|SLIP)"
    r"|INVOICE\s*(?:NO|NUMBER|#|:)"
    r"|CERTIFICATE\s+OF\s+(?:CONFORMANCE|COMPLIANCE)"
    r"|AIR\s+WAYBILL|AIRWAY\s+BILL|DELIVERY\s+(?:ORDER|NOTE)"
    r"|QUALITY\s+NOTIFICATION|CERTIFICATE\s+OF\s+ANALYSIS)",
    re.IGNORECASE,
)

# Valid parse_action values
PARSE_ACTIONS = frozenset({
    "full_parse",
    "shallow_index",
    "skipped_low_priority",
    "skipped_image_pdf",
    "skipped_unreadable",
})


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

def classify_filename(filename: str) -> tuple[str, int]:
    """Classify a filename as HIGH / MEDIUM / LOW document priority.

    HIGH (score 2): likely traveler, router, drawing, or job document.
    LOW  (score 0): likely BOL, cert, PO, inspection report, or shipping doc.
    MEDIUM (score 1): unknown — classify further by content.

    LOW beats HIGH: a D-numbered material cert is still LOW priority.
    Returns (classification, score).
    """
    has_high = bool(_FNAME_HIGH_RE.search(filename))
    has_low  = bool(_FNAME_LOW_RE.search(filename))
    if has_low:               # LOW always wins — avoid parsing admin docs
        return "LOW", 0
    if has_high:
        return "HIGH", 2
    return "MEDIUM", 1


def sample_pdf_first_page(path: Path) -> tuple[str, bool]:
    """Extract text from the FIRST PAGE only — fast pre-classification sample.

    Returns (text, is_text_extractable).
    Never raises; returns ("", False) on any failure.
    """
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                return "", False
            normal = pdf.pages[0].extract_text() or ""
            flow = pdf.pages[0].extract_text(use_text_flow=True) or ""
            text = normal if flow in normal else f"{normal}\n{flow}"
        stripped = text.strip()
        return stripped, bool(stripped)
    except Exception as exc:
        logger.debug(f"PDF first-page sample failed for {path.name}: {exc}")
        return "", False


def classify_content(text: str) -> tuple[str, int]:
    """Classify document content as HIGH / MEDIUM / LOW priority.

    Applied to first-page text sample only.  Empty text returns MEDIUM
    so that filename classification is the deciding factor.

    LOW beats HIGH: a shipping doc addressed to EMPOWER is still LOW.
    Returns (classification, score).
    """
    if not text:
        return "MEDIUM", 1
    has_low  = bool(_CONTENT_LOW_RE.search(text))
    has_high = bool(_CONTENT_HIGH_RE.search(text))
    if has_low:               # LOW wins — e.g., BOL mentioning EMPOWER as consignee
        return "LOW", 0
    if has_high:
        return "HIGH", 2
    return "MEDIUM", 1


def decide_parse_action(
    fname_cls: str,
    content_cls: str,
    text_extractable: bool,
) -> str:
    """Choose the parse action for a document based on classification scores.

    Returns one of:
      full_parse           — extract all fields (use full document text)
      shallow_index        — extract from first-page sample only
      skipped_low_priority — both filename and content classify as low value
      skipped_image_pdf    — PDF with no extractable text (scanned image)
    """
    if not text_extractable:
        return "skipped_image_pdf"
    if fname_cls == "HIGH" or content_cls == "HIGH":
        return "full_parse"
    if fname_cls == "MEDIUM" and content_cls == "MEDIUM":
        return "full_parse"
    if fname_cls == "MEDIUM" or content_cls == "MEDIUM":
        return "shallow_index"
    return "skipped_low_priority"


def _combined_priority(fname_cls: str, content_cls: str) -> str:
    """Map filename + content classifications to a single priority label."""
    s = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    total = s.get(fname_cls, 1) + s.get(content_cls, 1)
    if total >= 3:
        return "HIGH"
    if total >= 2:
        return "MEDIUM"
    return "LOW"


def compute_scan_metrics(records: list[dict]) -> dict:
    """Compute coverage and classification metrics from a completed scan.

    Accepts the metadata records list returned by scan_job_folders or
    scan_shared_prints.  Returns a plain dict suitable for logging.
    """
    if not records:
        return {"total_records": 0}

    parse_dist = Counter(r.get("parse_action", "unknown") for r in records)
    fname_dist = Counter(r.get("filename_classification", "?") for r in records)
    content_dist = Counter(r.get("content_classification", "?") for r in records)
    text_lengths = [r.get("sampled_text_length", 0) for r in records]

    return {
        "total_records":        len(records),
        "total_full_parsed":    parse_dist.get("full_parse", 0),
        "total_shallow_indexed": parse_dist.get("shallow_index", 0),
        "parse_action_distribution": dict(parse_dist),
        "filename_cls_distribution": dict(fname_dist),
        "content_cls_distribution":  dict(content_dist),
        "avg_sampled_text_length":
            sum(text_lengths) // max(len(text_lengths), 1),
    }


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: Path) -> tuple[str, str]:
    """Extract full text from a PDF (all pages) using pdfplumber.

    Returns (text, notes).
    """
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages_text = []
            for page in pdf.pages:
                normal = page.extract_text() or ""
                flow = page.extract_text(use_text_flow=True) or ""
                pages_text.append(normal if flow in normal else f"{normal}\n{flow}")
        text = "\n".join(pages_text).strip()
        if not text:
            return "", "unreadable_pdf"
        return text, ""
    except Exception as exc:
        logger.warning(f"PDF extraction failed for {path}: {exc}")
        return "", f"pdf_error:{type(exc).__name__}"


def extract_file_text(path: Path) -> tuple[str, str]:
    """Extract text from a file (PDF or text). Returns (text, notes)."""
    ext = path.suffix.upper()
    if ext == ".PDF":
        return extract_pdf_text(path)
    lines = read_file_lines(path)
    if lines is None:
        return "", "read_error"
    return "\n".join(lines), ""


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _first_match(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def extract_job_number(text: str) -> str:
    direct = re.search(r"(?im)^\s*Job\s*:\s*([A-Z]?[0-9]{4,7}(?:-[0-9]{1,6})?)\b", text)
    if direct:
        return direct.group(1).strip()
    return _first_match(_JOB_RE, text)


def extract_part_number(text: str) -> str:
    direct = re.search(r"(?im)^\s*Part\s*:\s*([A-Z0-9][A-Z0-9\-_/\.]{2,29})\b", text)
    if direct:
        return direct.group(1).strip()
    return _first_match(_PART_RE, text)


def extract_drawing_number(text: str) -> str:
    direct = re.search(r"(?im)\bDrawing\s*:\s*([A-Z0-9][A-Z0-9\-_/\.]{1,29})\b", text)
    if direct:
        return direct.group(1).strip()
    return _first_match(_DWG_RE, text)


def extract_revision(text: str) -> str:
    return _first_match(_REV_RE, text)


def _clean_material_raw(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip(" \t.,;:-")


def normalize_material(raw: str) -> str:
    """Normalize raw traveler material text to the alloy/condition only."""
    candidate = _clean_material_raw(raw)
    if not candidate or _MATERIAL_NOISE_RE.search(candidate):
        return ""

    candidate = re.sub(r"(?i)\bMaterial\s*:\s*", "", candidate)
    candidate = re.sub(r"(?i)\b(?:Diameter|Dia\.?)\b", " DIA ", candidate)
    candidate = re.sub(r"(?i)\b(?:Length|Long)\b.*$", "", candidate)
    candidate = re.sub(r"(?i)=+\s*Cut\s+to\b.*$", "", candidate)

    for pattern in _MATERIAL_TOKEN_PATTERNS:
        m = pattern.search(candidate)
        if m:
            material = _clean_material_raw(m.group(0)).upper()
            material = material.replace("H.T.", "HT")
            material = re.sub(r"\s+", " ", material)
            return material

    return ""


def _candidate_material_lines(text: str) -> tuple[str, str]:
    # 1. Explicit operation block line beginning with Material:
    for m in _EXPLICIT_MATERIAL_RE.finditer(text):
        raw = _clean_material_raw(m.group(1))
        normalized = normalize_material(raw)
        if normalized:
            return raw, normalized

    lines = [_clean_material_raw(line) for line in text.splitlines()]

    # 2. Materials section description lines.
    in_materials = False
    for line in lines:
        if re.fullmatch(r"(?i)(?:Materials?|Buys Comments)", line):
            in_materials = True
            continue
        if in_materials:
            if re.match(r"(?i)^Part:\s+", line):
                in_materials = False
                continue
            normalized = normalize_material(line)
            if normalized:
                return line, normalized

    # 3. Header/title material hints.
    for line in lines[:80]:
        if re.search(r"(?i)\b(?:shaft|plate|bar|tube|round|blank|casting|forging)\b", line):
            normalized = normalize_material(line)
            if normalized:
                return line, normalized

    # 4. Existing fallback logic.
    raw = _first_match(_MAT_RE, text)
    normalized = normalize_material(raw)
    if normalized:
        return _clean_material_raw(raw), normalized

    return "", ""


def extract_material_details(text: str) -> tuple[str, str]:
    """Return (raw_material_text, normalized_material)."""
    return _candidate_material_lines(text)


def extract_material(text: str) -> str:
    return extract_material_details(text)[1]


def _machine_keyword(text: str) -> str:
    m = _MACHINE_RE.search(text)
    return m.group(1).strip().upper() if m else ""


def _work_center_code(text: str) -> str:
    m = re.search(r"\b(\d{3})\b", str(text))
    return m.group(1) if m else ""


def _work_center_type(text: str) -> str:
    machine = _machine_keyword(text)
    aliases = {
        "TURNING": "LATHE",
        "MILLING": "MILL",
        "VMC": "MILL",
        "HMC": "MILL",
        "GRINDING": "GRIND",
        "INSPECTION": "INSPECT",
        "DRILLING": "DRILL",
        "BORING": "BORE",
        "BAND SAW": "SAW",
    }
    return aliases.get(machine, machine)


def _type_from_compact_work_center(token: str) -> str:
    compact = re.sub(r"[^A-Z]", "", str(token).upper())
    if "LATH" in compact:
        return "LATHE"
    if "MILL" in compact:
        return "MILL"
    if "GRIND" in compact or "GRIN" in compact:
        return "GRIND"
    if "INSP" in compact:
        return "INSPECT"
    if "LASER" in compact or "LASE" in compact:
        return "LASER"
    if "SHIP" in compact:
        return "SHIP"
    if "MATRCV" in compact:
        return "MATRCV"
    if "TOOL" in compact:
        return "TOOL"
    if "PROG" in compact:
        return "PROG"
    if "MODEL" in compact or "MOD" in compact:
        return "MODEL"
    return ""


def _format_work_center(code: str, typ: str) -> str:
    code = str(code or "").strip()
    typ = str(typ or "").strip().upper()
    if code and typ:
        return f"{code} {typ}"
    return code or typ


def extract_work_centers(text: str) -> str:
    seen: list[str] = []
    seen_upper: set[str] = set()
    wc_re = re.compile(r"\b(\d{3})\s+([A-Z][A-Z /-]{2,20})\b")
    for m in wc_re.finditer(text):
        code = m.group(1)
        typ = _work_center_type(m.group(2))
        if not typ:
            continue
        kw = f"{code} {typ}"
        ku = kw.upper()
        if ku not in seen_upper:
            seen.append(kw)
            seen_upper.add(ku)
    for m in _MACHINE_RE.finditer(text):
        kw = m.group(1).strip().upper()
        ku = kw.upper()
        if kw and ku not in seen_upper:
            seen.append(kw)
            seen_upper.add(ku)
    return ", ".join(seen)


def score_confidence(
    job_number: str,
    part_number: str,
    drawing_number: str,
    revision: str,
    material: str,
) -> str:
    count = sum(bool(v) for v in [job_number, part_number, drawing_number, revision, material])
    if count >= 4:
        return "HIGH"
    if count >= 2:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Router operation extraction
# ---------------------------------------------------------------------------

def extract_routing_operations(
    text: str,
    source_file: str,
    job_number: str,
) -> list[dict]:
    """Extract manufacturing routing operations from document text."""
    ops: list[dict] = []
    seq = 0
    pending_wc: dict | None = None
    current_op: dict | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        traveler_match = _TRAVELER_OP_RE.match(stripped)
        if traveler_match:
            wc_code = traveler_match.group(1) + traveler_match.group(2)
            wc_type = _type_from_compact_work_center(traveler_match.group(3))
            pending_wc = {
                "work_center_code": wc_code,
                "work_center_type": wc_type,
                "work_center": _format_work_center(wc_code, wc_type),
            }
            current_op = None
            continue

        if pending_wc:
            op_desc = re.match(r"^\s*(\d{1,3})\s+(.+)$", stripped)
            if op_desc:
                op_num_str = op_desc.group(1)
                description = op_desc.group(2).strip()
                seq += 1
                current_op = {
                    "source_file": source_file,
                    "job_number": job_number,
                    "operation_sequence": seq,
                    "operation_number": op_num_str,
                    "work_center": pending_wc["work_center"],
                    "work_center_code": pending_wc["work_center_code"],
                    "work_center_type": pending_wc["work_center_type"],
                    "machine": pending_wc["work_center_type"] or _machine_keyword(description),
                    "operation_description": description,
                    "operation_notes": "",
                }
                ops.append(current_op)
                pending_wc = None
                continue

        wc_line = _WC_LINE_RE.match(stripped)
        if wc_line and current_op is not None:
            wc_code = wc_line.group(1)
            wc_type = _work_center_type(wc_line.group(2))
            if wc_code == current_op.get("work_center_code") or not current_op.get("work_center_code"):
                current_op["work_center_code"] = wc_code
                current_op["work_center_type"] = wc_type
                current_op["work_center"] = _format_work_center(wc_code, wc_type)
                current_op["machine"] = wc_type
            extra = _clean_material_raw(wc_line.group(3))
            if extra:
                current_op["operation_notes"] = (
                    f"{current_op['operation_notes']} {extra}".strip()
                )
            continue

        if (
            current_op is not None
            and not _OP_ROW_RE.match(stripped)
            and not _OP_LABEL_RE.match(stripped)
        ):
            if not re.match(r"^(?:Part|Rev):\s+", stripped, flags=re.IGNORECASE):
                current_op["operation_notes"] = (
                    f"{current_op['operation_notes']} {stripped}".strip()
                )
            continue

        m = _OP_ROW_RE.match(stripped)
        if m:
            op_num_str = m.group(1)
            try:
                op_num_int = int(op_num_str)
            except ValueError:
                continue
            if op_num_int % 5 != 0 or op_num_int < 5 or op_num_int > 990:
                continue
            rest = m.group(2).strip()
            parts = re.split(r"\s{2,}", rest, maxsplit=1)
            if len(parts) == 2:
                work_center, description = parts[0].strip(), parts[1].strip()
            else:
                work_center, description = "", rest
            seq += 1
            ops.append({
                "source_file": source_file,
                "job_number": job_number,
                "operation_sequence": seq,
                "operation_number": op_num_str,
                "work_center": work_center,
                "work_center_code": _work_center_code(work_center),
                "work_center_type": _work_center_type(work_center + " " + description),
                "machine": _machine_keyword(work_center + " " + description),
                "operation_description": description,
                "operation_notes": "",
            })
            current_op = ops[-1]
            continue

        m = _OP_LABEL_RE.match(stripped)
        if m:
            op_num_str = m.group(1)
            description = m.group(2).strip()
            seq += 1
            ops.append({
                "source_file": source_file,
                "job_number": job_number,
                "operation_sequence": seq,
                "operation_number": op_num_str,
                "work_center": "",
                "work_center_code": "",
                "work_center_type": _work_center_type(description),
                "machine": _machine_keyword(description),
                "operation_description": description,
                "operation_notes": "",
            })
            current_op = ops[-1]

    return ops


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def build_job_metadata_record(
    path: Path,
    preloaded_text: str | None = None,
    preloaded_notes: str = "",
    filename_classification: str = "MEDIUM",
    content_classification: str = "MEDIUM",
    parse_action: str = "full_parse",
    text_extractable: bool = True,
    sampled_text_length: int = 0,
) -> tuple[dict, list[dict]]:
    """Extract job metadata and routing operations from one file.

    If preloaded_text is provided (by the scan loop after first-page sampling),
    that text is used instead of re-reading the file.  All classification
    keyword-arguments default to MEDIUM / full_parse so callers that do not
    supply them (e.g. direct test calls) get the same behaviour as before
    Phase 6A-Optimize.

    Returns (metadata_record, operations_list).
    """
    try:
        modified_datetime = datetime.fromtimestamp(
            path.stat().st_mtime
        ).isoformat(timespec="seconds")
    except OSError:
        modified_datetime = ""

    if preloaded_text is None:
        # Backward-compatible path: read file and self-classify
        text, notes = extract_file_text(path)
        filename_classification, _ = classify_filename(path.name)
        content_classification, _ = classify_content(text)
        text_extractable = bool(text)
        sampled_text_length = len(text)
    else:
        text  = preloaded_text
        notes = preloaded_notes

    job_number     = extract_job_number(text)
    part_number    = extract_part_number(text)
    drawing_number = extract_drawing_number(text)
    revision       = extract_revision(text)
    raw_material_text, normalized_material = extract_material_details(text)
    material       = normalized_material
    work_centers   = extract_work_centers(text)
    ops            = extract_routing_operations(text, str(path), job_number)

    record = {
        "source_file":            str(path),
        "filename":               path.name,
        "modified_datetime":      modified_datetime,
        "job_number":             job_number,
        "part_number":            part_number,
        "drawing_number":         drawing_number,
        "revision":               revision,
        "material":               material,
        "raw_material_text":      raw_material_text,
        "normalized_material":    normalized_material,
        "work_centers":           work_centers,
        "operation_count":        len(ops),
        "extraction_confidence":  score_confidence(
            job_number, part_number, drawing_number, revision, material
        ),
        "extraction_notes":       notes,
        # Classification fields (Phase 6A-Optimize)
        "filename_classification": filename_classification,
        "content_classification":  content_classification,
        "parse_priority":         _combined_priority(
            filename_classification, content_classification
        ),
        "parse_action":           parse_action,
        "text_extractable":       text_extractable,
        "sampled_text_length":    sampled_text_length,
    }
    return record, ops


def build_shared_print_record(
    path: Path,
    preloaded_text: str | None = None,
    preloaded_notes: str = "",
    filename_classification: str = "MEDIUM",
    content_classification: str = "MEDIUM",
    parse_action: str = "full_parse",
    text_extractable: bool = True,
    sampled_text_length: int = 0,
) -> dict:
    """Extract metadata from a shared part print."""
    try:
        stat = path.stat()
        file_size: int | None = stat.st_size
        modified_datetime = datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(timespec="seconds")
    except OSError:
        file_size = None
        modified_datetime = ""

    if preloaded_text is None:
        text, notes = extract_file_text(path)
        filename_classification, _ = classify_filename(path.name)
        content_classification, _ = classify_content(text)
        text_extractable = bool(text)
        sampled_text_length = len(text)
    else:
        text  = preloaded_text
        notes = preloaded_notes

    part_number    = extract_part_number(text)
    drawing_number = extract_drawing_number(text)
    revision       = extract_revision(text)
    raw_material_text, normalized_material = extract_material_details(text)
    material       = normalized_material

    count = sum(bool(v) for v in [part_number, drawing_number, revision, material])
    if count >= 3:
        confidence = "HIGH"
    elif count >= 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "source_file":      str(path),
        "filename":         path.name,
        "modified_datetime": modified_datetime,
        "part_number":      part_number,
        "drawing_number":   drawing_number,
        "revision":         revision,
        "material":         material,
        "raw_material_text": raw_material_text,
        "normalized_material": normalized_material,
        "file_size_bytes":  file_size if file_size is not None else "",
        "extraction_confidence":  confidence,
        "extraction_notes":       notes,
        # Classification fields
        "filename_classification": filename_classification,
        "content_classification":  content_classification,
        "parse_priority":         _combined_priority(
            filename_classification, content_classification
        ),
        "parse_action":    parse_action,
        "text_extractable": text_extractable,
        "sampled_text_length": sampled_text_length,
    }


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def _scan_one_file(
    path: Path,
    is_pdf: bool,
) -> tuple[str, str, str, bool, int]:
    """Sample content for classification; return (fname_cls, content_cls, sample_text, extractable, txt_len).

    For PDFs: reads first page only.
    For TXT: reads the full file (typically small).
    """
    fname_cls, _ = classify_filename(path.name)

    if is_pdf:
        sample_text, extractable = sample_pdf_first_page(path)
    else:
        lines = read_file_lines(path) or []
        sample_text = "\n".join(lines)
        extractable = bool(sample_text.strip())

    content_cls, _ = classify_content(sample_text)
    return fname_cls, content_cls, sample_text, extractable, len(sample_text)


def scan_job_folders(
    root: Path,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> tuple[list[dict], list[dict]]:
    """Recursively scan root for traveler/router documents modified within lookback_days.

    Uses intelligent pre-classification to skip unrelated documents (BOLs, certs,
    POs, etc.) before attempting full metadata extraction.

    Returns (metadata_records, operation_records).
    READ-ONLY — never modifies source folders.
    """
    metadata:   list[dict] = []
    operations: list[dict] = []

    if not root.exists():
        logger.warning(f"Job folders root not accessible: {root}")
        return metadata, operations

    cutoff = datetime.now() - timedelta(days=lookback_days)
    logger.info(f"Job folders: cutoff={cutoff.date()} ({lookback_days}d) root={root}")

    all_files  = sorted(root.rglob("*"))
    candidates = [
        f for f in all_files
        if f.is_file() and f.suffix.upper() in JOB_FOLDER_EXTENSIONS
    ]

    # Per-scan counters (logged at end; not returned to preserve existing API)
    counts: dict[str, int] = {
        "date_skipped": 0,
        "full_parse": 0,
        "shallow_index": 0,
        "skipped_low_priority": 0,
        "skipped_image_pdf": 0,
        "skipped_unreadable": 0,
        "errors": 0,
    }
    parse_times_ms: list[float] = []

    for path in candidates:
        is_pdf = path.suffix.upper() == ".PDF"

        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            mtime = datetime.now()

        if mtime < cutoff:
            counts["date_skipped"] += 1
            logger.debug(f"  skip (too old {mtime.date()}): {path.name}")
            continue

        t0 = time.perf_counter()
        try:
            fname_cls, content_cls, sample_text, extractable, txt_len = \
                _scan_one_file(path, is_pdf)

            parse_action = decide_parse_action(fname_cls, content_cls, extractable)

            if parse_action in ("skipped_low_priority", "skipped_image_pdf",
                                "skipped_unreadable"):
                counts[parse_action] += 1
                logger.debug(
                    f"  {parse_action}: fname={fname_cls} "
                    f"content={content_cls} {path.name}"
                )
                continue

            if parse_action == "full_parse" and is_pdf:
                # For PDFs: read ALL pages now (sample was first page only)
                full_text, full_notes = extract_pdf_text(path)
                text_for_record = full_text
                notes_for_record = full_notes
            else:
                # TXT (full file already in sample) or shallow_index (first page only)
                text_for_record = sample_text
                notes_for_record = ""

            meta, ops = build_job_metadata_record(
                path,
                preloaded_text=text_for_record,
                preloaded_notes=notes_for_record,
                filename_classification=fname_cls,
                content_classification=content_cls,
                parse_action=parse_action,
                text_extractable=extractable,
                sampled_text_length=txt_len,
            )
            metadata.append(meta)
            operations.extend(ops)
            counts[parse_action] += 1

            logger.debug(
                f"  [{parse_action}] fname={fname_cls} content={content_cls} "
                f"job={meta['job_number'] or '—'} "
                f"part={meta['part_number'] or '—'} "
                f"ops={meta['operation_count']} conf={meta['extraction_confidence']} "
                f"{path.name}"
            )

        except Exception as exc:
            counts["errors"] += 1
            logger.warning(f"Failed processing {path}: {exc}")

        parse_times_ms.append((time.perf_counter() - t0) * 1000)

    avg_ms = (sum(parse_times_ms) / len(parse_times_ms)) if parse_times_ms else 0.0

    logger.info(
        f"Job folders scan: {len(metadata)} record(s) "
        f"[full={counts['full_parse']} shallow={counts['shallow_index']}] "
        f"| skipped low={counts['skipped_low_priority']} "
        f"img={counts['skipped_image_pdf']} "
        f"date={counts['date_skipped']} "
        f"| {len(operations)} op(s) "
        f"| avg {avg_ms:.1f} ms/file"
    )
    return metadata, operations


def scan_shared_prints(
    root: Path,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> list[dict]:
    """Recursively scan root for shared part prints modified within lookback_days.

    Returns list of print index records.
    READ-ONLY — never modifies source folders.
    """
    records: list[dict] = []

    if not root.exists():
        logger.warning(f"Shared prints root not accessible: {root}")
        return records

    cutoff = datetime.now() - timedelta(days=lookback_days)
    logger.info(f"Shared prints: cutoff={cutoff.date()} ({lookback_days}d) root={root}")

    all_files  = sorted(root.rglob("*"))
    candidates = [
        f for f in all_files
        if f.is_file() and f.suffix.upper() in SHARED_PRINT_EXTENSIONS
    ]

    counts: dict[str, int] = {
        "date_skipped": 0,
        "full_parse": 0,
        "shallow_index": 0,
        "skipped_low_priority": 0,
        "skipped_image_pdf": 0,
        "errors": 0,
    }
    parse_times_ms: list[float] = []

    for path in candidates:
        is_pdf = path.suffix.upper() == ".PDF"

        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            mtime = datetime.now()

        if mtime < cutoff:
            counts["date_skipped"] += 1
            logger.debug(f"  skip (too old {mtime.date()}): {path.name}")
            continue

        t0 = time.perf_counter()
        try:
            fname_cls, content_cls, sample_text, extractable, txt_len = \
                _scan_one_file(path, is_pdf)

            parse_action = decide_parse_action(fname_cls, content_cls, extractable)

            if parse_action in ("skipped_low_priority", "skipped_image_pdf",
                                "skipped_unreadable"):
                counts[parse_action] += 1
                logger.debug(f"  {parse_action}: {path.name}")
                continue

            if parse_action == "full_parse" and is_pdf:
                full_text, full_notes = extract_pdf_text(path)
                text_for_record = full_text
                notes_for_record = full_notes
            else:
                text_for_record = sample_text
                notes_for_record = ""

            rec = build_shared_print_record(
                path,
                preloaded_text=text_for_record,
                preloaded_notes=notes_for_record,
                filename_classification=fname_cls,
                content_classification=content_cls,
                parse_action=parse_action,
                text_extractable=extractable,
                sampled_text_length=txt_len,
            )
            records.append(rec)
            counts[parse_action] += 1

            logger.debug(
                f"  [{parse_action}] pn={rec['part_number'] or '—'} "
                f"dwg={rec['drawing_number'] or '—'} "
                f"conf={rec['extraction_confidence']} {path.name}"
            )

        except Exception as exc:
            counts["errors"] += 1
            logger.warning(f"Failed processing {path}: {exc}")

        parse_times_ms.append((time.perf_counter() - t0) * 1000)

    avg_ms = (sum(parse_times_ms) / len(parse_times_ms)) if parse_times_ms else 0.0

    logger.info(
        f"Shared prints scan: {len(records)} record(s) "
        f"[full={counts['full_parse']} shallow={counts['shallow_index']}] "
        f"| skipped low={counts['skipped_low_priority']} "
        f"img={counts['skipped_image_pdf']} "
        f"date={counts['date_skipped']} "
        f"| avg {avg_ms:.1f} ms/file"
    )
    return records


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

_JOB_META_COLS = [
    "source_file", "filename", "modified_datetime",
    "job_number", "part_number", "drawing_number",
    "revision", "material", "raw_material_text", "normalized_material",
    "work_centers", "operation_count",
    "extraction_confidence", "extraction_notes",
    # Phase 6A-Optimize fields
    "filename_classification", "content_classification",
    "parse_priority", "parse_action",
    "text_extractable", "sampled_text_length",
]

_PRINT_INDEX_COLS = [
    "source_file", "filename", "modified_datetime",
    "part_number", "drawing_number",
    "revision", "material", "raw_material_text", "normalized_material",
    "file_size_bytes",
    "extraction_confidence", "extraction_notes",
    # Phase 6A-Optimize fields
    "filename_classification", "content_classification",
    "parse_priority", "parse_action",
    "text_extractable", "sampled_text_length",
]

_ROUTER_OPS_COLS = [
    "source_file", "job_number", "operation_sequence", "operation_number",
    "work_center", "work_center_code", "work_center_type", "machine",
    "operation_description", "operation_notes",
]


def export_job_metadata(records: list[dict], exports_dir: Path, timestamp: str) -> Path:
    out = exports_dir / f"job_metadata_{timestamp}.csv"
    df = (
        pd.DataFrame(records, columns=_JOB_META_COLS)
        if records
        else pd.DataFrame(columns=_JOB_META_COLS)
    )
    assert_safe_write(out)
    df.to_csv(out, index=False)
    logger.info(f"Job metadata    -> {out}  ({len(df)} row(s))")
    return out


def export_shared_print_index(records: list[dict], exports_dir: Path, timestamp: str) -> Path:
    out = exports_dir / f"shared_print_index_{timestamp}.csv"
    df = (
        pd.DataFrame(records, columns=_PRINT_INDEX_COLS)
        if records
        else pd.DataFrame(columns=_PRINT_INDEX_COLS)
    )
    assert_safe_write(out)
    df.to_csv(out, index=False)
    logger.info(f"Shared prints   -> {out}  ({len(df)} row(s))")
    return out


def export_router_operations(records: list[dict], exports_dir: Path, timestamp: str) -> Path:
    out = exports_dir / f"router_operations_{timestamp}.csv"
    df = (
        pd.DataFrame(records, columns=_ROUTER_OPS_COLS)
        if records
        else pd.DataFrame(columns=_ROUTER_OPS_COLS)
    )
    assert_safe_write(out)
    df.to_csv(out, index=False)
    logger.info(f"Router ops      -> {out}  ({len(df)} row(s))")
    return out


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_job_scan(
    job_root: Path = JOB_FOLDERS_ROOT,
    prints_root: Path = SHARED_PRINTS_ROOT,
    exports_dir: Path | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> tuple[Path, Path, Path]:
    """Full Phase 6A pipeline: scan job folders + shared prints, export CSVs.

    Returns (job_metadata_path, shared_print_index_path, router_operations_path).
    Never writes to source folders. Never overwrites existing exports.
    """
    if exports_dir is None:
        exports_dir = Path(__file__).parent.parent / "exports"
    assert_safe_write(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== Phase 6A — Job Metadata Scanner ===")
    logger.info(f"Job folders root  : {job_root}")
    logger.info(f"Shared prints root: {prints_root}")
    logger.info(f"Exports dir       : {exports_dir}")
    logger.info(f"Lookback window   : {lookback_days} days")

    meta_records, op_records = scan_job_folders(job_root,   lookback_days=lookback_days)
    print_records             = scan_shared_prints(prints_root, lookback_days=lookback_days)

    meta_path   = export_job_metadata(meta_records,   exports_dir, timestamp)
    prints_path = export_shared_print_index(print_records, exports_dir, timestamp)
    ops_path    = export_router_operations(op_records, exports_dir, timestamp)

    # Summary metrics
    job_metrics   = compute_scan_metrics(meta_records)
    print_metrics = compute_scan_metrics(print_records)
    logger.info(
        f"Job metadata  : total={job_metrics['total_records']} "
        f"full={job_metrics.get('total_full_parsed',0)} "
        f"shallow={job_metrics.get('total_shallow_indexed',0)}"
    )
    logger.info(
        f"Shared prints : total={print_metrics['total_records']} "
        f"full={print_metrics.get('total_full_parsed',0)} "
        f"shallow={print_metrics.get('total_shallow_indexed',0)}"
    )

    logger.info("=== Phase 6A complete ===")
    return meta_path, prints_path, ops_path
