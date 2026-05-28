"""
scanner.py — Folder scanner and manifest exporter for Proven Program Engine.

Scans P:\\*\\Proven\\ (case-insensitive) for CNC programs.
Filters by allowed file extensions and modification date (last 2 years).
Exports a manifest CSV and JSON to the exports/ directory.

READ-ONLY against P:\\ — this module never writes to the source drive.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .safety import assert_safe_write
from .utils import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".EIA", ".NC", ".TXT", ".MIN", ".TAP", ".MPF", ".SPF", ".MAZ", ".PGM",
    ".OP1", ".OP2", ".OP3",
})

_PROVEN_RE = re.compile(r"^proven$", re.IGNORECASE)
_DEFAULT_CUTOFF_DAYS = 730  # 2 years


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_proven_folders(root: Path) -> list[Path]:
    """Return all Proven subfolders found one level under root.

    Expected layout:  root / machine_folder / Proven /
    Matches 'Proven', 'proven', 'PROVEN', etc. case-insensitively.
    """
    results: list[Path] = []

    if not root.exists():
        logger.error(f"Root path does not exist: {root}")
        return results

    for machine_dir in sorted(root.iterdir()):
        if not machine_dir.is_dir():
            continue
        try:
            subdirs = sorted(machine_dir.iterdir())
        except PermissionError as exc:
            logger.warning(f"Permission denied scanning {machine_dir}: {exc}")
            continue

        for subdir in subdirs:
            if subdir.is_dir() and _PROVEN_RE.match(subdir.name):
                logger.debug(f"Proven folder found: {subdir}")
                results.append(subdir)

    logger.info(f"Found {len(results)} Proven folder(s) under {root}")
    return results


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_files(
    root: Path,
    proven_folders: list[Path],
    cutoff_date: datetime,
) -> list[dict]:
    """Scan all Proven folders and build a manifest record for every candidate file.

    Each record contains inclusion status and a skip_reason when excluded.
    """
    records: list[dict] = []
    program_id = 0

    for folder in proven_folders:
        machine_folder = folder.parent.name
        logger.info(f"Scanning {machine_folder} → {folder}")
        folder_total = 0
        folder_included = 0

        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue

            folder_total += 1
            ext = file_path.suffix.upper()
            included = True
            skip_reason = ""
            modified_dt: datetime | None = None
            file_size: int | None = None

            try:
                stat = file_path.stat()
                modified_dt = datetime.fromtimestamp(stat.st_mtime)
                file_size = stat.st_size
            except OSError as exc:
                included = False
                skip_reason = f"stat_error:{exc}"

            if included and ext not in ALLOWED_EXTENSIONS:
                included = False
                skip_reason = f"extension_excluded:{ext}"

            if included and modified_dt is not None and modified_dt < cutoff_date:
                included = False
                skip_reason = f"too_old:{modified_dt.date()}"

            try:
                rel_path = str(file_path.relative_to(root))
            except ValueError:
                rel_path = str(file_path)

            if included:
                folder_included += 1

            program_id += 1
            records.append({
                "program_id": program_id,
                "source_file": str(file_path),
                "relative_path": rel_path,
                "machine_folder": machine_folder,
                "filename": file_path.name,
                "extension": ext,
                "modified_datetime": modified_dt.isoformat() if modified_dt else "",
                "file_size_bytes": file_size if file_size is not None else "",
                "included": included,
                "skip_reason": skip_reason,
            })

        logger.info(
            f"  {machine_folder}: {folder_total} files found, {folder_included} included"
        )

    total = len(records)
    included_count = sum(1 for r in records if r["included"])
    logger.info(
        f"Manifest: {total} total files | {included_count} included | "
        f"{total - included_count} skipped"
    )
    return records


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_manifest(records: list[dict], exports_dir: Path) -> tuple[Path, Path]:
    """Write manifest to timestamped CSV and JSON files.

    Returns (csv_path, json_path).
    """
    assert_safe_write(exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = exports_dir / f"manifest_{timestamp}.csv"
    json_path = exports_dir / f"manifest_{timestamp}.json"

    df = pd.DataFrame(records)
    assert_safe_write(csv_path)
    df.to_csv(csv_path, index=False)
    logger.info(f"Manifest CSV  → {csv_path}  ({len(df)} rows)")

    assert_safe_write(json_path)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    logger.info(f"Manifest JSON → {json_path}")

    return csv_path, json_path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_scan(
    root: Path,
    exports_dir: Path,
    cutoff_days: int = _DEFAULT_CUTOFF_DAYS,
) -> tuple[Path, Path]:
    """Full scan pipeline: discover Proven folders → scan files → export manifest."""
    cutoff_date = datetime.now() - timedelta(days=cutoff_days)
    logger.info(f"Scan root    : {root}")
    logger.info(f"Cutoff date  : {cutoff_date.date()} ({cutoff_days} days back)")

    proven_folders = find_proven_folders(root)
    if not proven_folders:
        logger.warning("No Proven folders discovered. Verify P:\\ is accessible and mapped.")

    records = scan_files(root, proven_folders, cutoff_date)
    return export_manifest(records, exports_dir)
