"""
tooldb_loader.py — Read Mastercam .tooldb/.TOOLDB SQLite files.

Extracts tool identity, descriptions, geometry, and S/F parameters.
S/F data is never marked usable (sf_usable=False) because TlWorkMaterial
is absent in all known libraries — no material context exists to validate
speeds and feeds against a specific workpiece material.

READ ONLY. Never modifies .tooldb files.
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_MACHINE_ID_RE = re.compile(r'\b(\d{3,4})\b')

_COLUMN_ORDER = [
    "source_tooldb", "machine_id",
    "tool_number", "tool_station",
    "tool_name", "holder_name",
    "tool_category",
    "overall_diameter", "flute_count", "mc_tool_type",
    "insert_shape", "insert_ic_diameter", "insert_corner_radius",
    "feed_rate_ipm", "spindle_rpm",
    "is_metric", "sf_usable", "sf_reject_reason",
]

_MILL_SQL = """
    SELECT
        a.ToolNumber                        AS tool_number,
        a.IsMetric                          AS is_metric,
        t.ToolStation                       AS tool_station,
        ai_t.Name                           AS tool_name,
        COALESCE(ai_h.Name, '')             AS holder_name,
        m.OverallDiameter                   AS overall_diameter,
        m.FluteCount                        AS flute_count,
        m.MCToolType                        AS mc_tool_type,
        COALESCE(op.FeedRate,    0.0)       AS feed_rate_ipm,
        COALESCE(op.SpindleSpeed, 0)        AS spindle_rpm
    FROM TlAssembly a
    JOIN TlAssemblyItem  ai_t ON a.MainTool   = ai_t.ID
    JOIN TlTool          t    ON t.ID          = ai_t.ID
    JOIN TlToolMill      m    ON m.ID          = t.ID
    LEFT JOIN TlOpParams op   ON op.ID         = m.TlOpParamsID
    LEFT JOIN TlAssemblyItem ai_h ON a.MainHolder = ai_h.ID
"""

_LATHE_SQL = """
    SELECT
        a.ToolNumber                            AS tool_number,
        a.IsMetric                              AS is_metric,
        t.ToolStation                           AS tool_station,
        ai_t.Name                               AS tool_name,
        COALESCE(ai_h.Name, '')                 AS holder_name,
        COALESCE(ins.AnsiShapeCode, '')         AS ansi_code,
        COALESCE(ins.ICDiameter,   0.0)         AS ic_diameter,
        COALESCE(ins.CornerRadius, 0.0)         AS corner_radius
    FROM TlAssembly a
    JOIN TlAssemblyItem  ai_t ON a.MainTool   = ai_t.ID
    JOIN TlTool          t    ON t.ID          = ai_t.ID
    JOIN TlToolLathe     l    ON l.ID          = t.ID
    LEFT JOIN TlInsert   ins  ON ins.ID        = l.InsertID
    LEFT JOIN TlAssemblyItem ai_h ON a.MainHolder = ai_h.ID
"""

_UNKNOWN_SQL = """
    SELECT
        a.ToolNumber                        AS tool_number,
        a.IsMetric                          AS is_metric,
        t.ToolStation                       AS tool_station,
        ai_t.Name                           AS tool_name,
        COALESCE(ai_h.Name, '')             AS holder_name
    FROM TlAssembly a
    JOIN TlAssemblyItem  ai_t ON a.MainTool   = ai_t.ID
    JOIN TlTool          t    ON t.ID          = ai_t.ID
    LEFT JOIN TlToolMill m    ON m.ID          = t.ID
    LEFT JOIN TlToolLathe l   ON l.ID          = t.ID
    LEFT JOIN TlAssemblyItem ai_h ON a.MainHolder = ai_h.ID
    WHERE m.ID IS NULL AND l.ID IS NULL
"""


def _extract_machine_ids(path: Path) -> list[str]:
    """Extract 3–4 digit machine IDs from the filename stem."""
    return _MACHINE_ID_RE.findall(path.stem)


def _is_active_tooldb(path: Path) -> bool:
    """Return True for live .tooldb/.TOOLDB files; False for backups and variants."""
    return path.suffix.lower() == ".tooldb"


def _has_material(con: sqlite3.Connection) -> bool:
    """Return True if TlWorkMaterial has any rows."""
    try:
        count = con.execute("SELECT COUNT(*) FROM TlWorkMaterial").fetchone()[0]
        return count > 0
    except Exception:
        return False


def _clean(val) -> str:
    if val is None:
        return ""
    return str(val).strip()


def _ansi_code_to_shape(code_str: str) -> str:
    """Convert ANSI shape code (stored as ASCII decimal) to insert shape letter."""
    try:
        return chr(int(code_str))
    except (ValueError, TypeError):
        return _clean(code_str)


def _query_mill(con: sqlite3.Connection, sf_usable: bool, sf_reject: str) -> list[dict]:
    rows = con.execute(_MILL_SQL).fetchall()
    results = []
    for r in rows:
        feed = r[8] if r[8] else None
        rpm = r[9] if r[9] else None
        results.append({
            "tool_number": r[0],
            "is_metric": bool(r[1]),
            "tool_station": r[2],
            "tool_name": _clean(r[3]),
            "holder_name": _clean(r[4]),
            "tool_category": "mill",
            "overall_diameter": r[5],
            "flute_count": r[6],
            "mc_tool_type": r[7],
            "insert_shape": None,
            "insert_ic_diameter": None,
            "insert_corner_radius": None,
            "feed_rate_ipm": feed,
            "spindle_rpm": rpm,
            "sf_usable": sf_usable,
            "sf_reject_reason": sf_reject,
        })
    return results


def _query_lathe(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(_LATHE_SQL).fetchall()
    results = []
    for r in rows:
        shape = _ansi_code_to_shape(r[5]) if r[5] else None
        ic_dia = r[6] if r[6] else None
        cr = r[7] if r[7] else None
        results.append({
            "tool_number": r[0],
            "is_metric": bool(r[1]),
            "tool_station": r[2],
            "tool_name": _clean(r[3]),
            "holder_name": _clean(r[4]),
            "tool_category": "lathe",
            "overall_diameter": None,
            "flute_count": None,
            "mc_tool_type": None,
            "insert_shape": shape,
            "insert_ic_diameter": ic_dia,
            "insert_corner_radius": cr,
            "feed_rate_ipm": None,
            "spindle_rpm": None,
            "sf_usable": False,
            "sf_reject_reason": "no_sf_data",
        })
    return results


def _query_unknown(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(_UNKNOWN_SQL).fetchall()
    results = []
    for r in rows:
        results.append({
            "tool_number": r[0],
            "is_metric": bool(r[1]),
            "tool_station": r[2],
            "tool_name": _clean(r[3]),
            "holder_name": _clean(r[4]),
            "tool_category": "unknown",
            "overall_diameter": None,
            "flute_count": None,
            "mc_tool_type": None,
            "insert_shape": None,
            "insert_ic_diameter": None,
            "insert_corner_radius": None,
            "feed_rate_ipm": None,
            "spindle_rpm": None,
            "sf_usable": False,
            "sf_reject_reason": "no_sf_data",
        })
    return results


def load_tooldb(path: Path) -> pd.DataFrame:
    """
    Load one .tooldb/.TOOLDB file. Returns a DataFrame with one row per
    (machine_id, tool_assembly). Returns empty DataFrame on any error.

    READ ONLY — never writes to the source file.
    """
    if not path.exists():
        return pd.DataFrame()

    machine_ids = _extract_machine_ids(path)

    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        has_mat = _has_material(con)
        sf_reject = "" if has_mat else "missing_material"
        sf_usable = has_mat  # always False in practice (TlWorkMaterial is empty)

        records: list[dict] = []
        records.extend(_query_mill(con, sf_usable, sf_reject))
        records.extend(_query_lathe(con))
        records.extend(_query_unknown(con))
        con.close()
    except Exception:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    source_stem = path.stem
    for rec in records:
        rec["source_tooldb"] = source_stem

    if machine_ids:
        expanded = []
        for mid in machine_ids:
            for rec in records:
                expanded.append({**rec, "machine_id": mid})
        df = pd.DataFrame(expanded)
    else:
        for rec in records:
            rec["machine_id"] = ""
        df = pd.DataFrame(records)

    for col in _COLUMN_ORDER:
        if col not in df.columns:
            df[col] = None

    df = df[_COLUMN_ORDER]

    # Ensure bool columns hold Python bools (not numpy.bool_) for identity comparisons
    for bool_col in ("sf_usable", "is_metric"):
        df[bool_col] = df[bool_col].astype(object)

    return df


def load_all_tooldb(library_dir: Path) -> pd.DataFrame:
    """
    Scan library_dir for all active .tooldb/.TOOLDB files and load them.
    Returns a combined DataFrame. Returns empty DataFrame if none found.
    """
    if not library_dir.exists():
        return pd.DataFrame()

    paths = sorted(
        p for p in library_dir.iterdir()
        if p.is_file() and _is_active_tooldb(p)
    )

    frames = [load_tooldb(p) for p in paths]
    frames = [f for f in frames if not f.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def export_tooldb_reference(
    df: pd.DataFrame,
    exports_dir: Path,
    timestamp: Optional[str] = None,
) -> Path:
    """Write tooldb_reference_<timestamp>.csv to exports_dir. Returns the output path."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exports_dir.mkdir(parents=True, exist_ok=True)
    out_path = exports_dir / f"tooldb_reference_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    return out_path
