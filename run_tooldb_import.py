"""
run_tooldb_import.py — Import Mastercam TOOLDB files and export reference CSV.

Usage:
    py run_tooldb_import.py [--library-dir PATH] [--exports-dir PATH]

Defaults:
    --library-dir   M:\\Tooling Data\\Machine Libraries
    --exports-dir   exports/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.tooldb_loader import export_tooldb_reference, load_all_tooldb

_DEFAULT_LIBRARY = Path(r"M:\Tooling Data\Machine Libraries")
_DEFAULT_EXPORTS = Path(__file__).parent / "exports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Mastercam TOOLDB libraries.")
    parser.add_argument("--library-dir", type=Path, default=_DEFAULT_LIBRARY)
    parser.add_argument("--exports-dir", type=Path, default=_DEFAULT_EXPORTS)
    args = parser.parse_args()

    library_dir: Path = args.library_dir
    exports_dir: Path = args.exports_dir

    if not library_dir.exists():
        print(f"ERROR: Library directory not found: {library_dir}", file=sys.stderr)
        return 1

    print(f"Scanning: {library_dir}")
    df = load_all_tooldb(library_dir)

    if df.empty:
        print("No tool records loaded. Check that .tooldb files are present.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = export_tooldb_reference(df, exports_dir, timestamp)

    mill_count = (df["tool_category"] == "mill").sum()
    lathe_count = (df["tool_category"] == "lathe").sum()
    machine_count = df["machine_id"].nunique()

    print(f"Loaded {len(df):,} records from {machine_count} machine IDs")
    print(f"  Mill tools:  {mill_count:,}")
    print(f"  Lathe tools: {lathe_count:,}")
    print(f"  S/F usable:  {df['sf_usable'].sum():,}  (expected 0 — no material data)")
    print(f"Exported: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
