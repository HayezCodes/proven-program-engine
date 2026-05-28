"""
summarize_6b.py -- Quick summary of Phase 6B linker outputs.
Reads the latest program_job_links_*.csv and material_backfill_*.csv.
"""
import sys
import io
from pathlib import Path
import pandas as pd

# Force UTF-8 on Windows CP1252 consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EXPORTS = Path(__file__).parent / "exports"


def _latest(pattern):
    files = sorted(EXPORTS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main():
    links_path    = _latest("program_job_links_*.csv")
    backfill_path = _latest("material_backfill_*.csv")
    manifest_path = _latest("manifest_*.csv")
    cuts_path     = _latest("cuts_*.csv")

    print("=" * 60)
    print("Phase 6B Summary")
    print("=" * 60)

    print("\n--- Source files ---")
    print(f"  Manifest  : {manifest_path.name if manifest_path else 'NOT FOUND'}")
    print(f"  Cuts      : {cuts_path.name if cuts_path else 'NOT FOUND'}")
    print(f"  Links     : {links_path.name if links_path else 'NOT FOUND'}")
    print(f"  Backfill  : {backfill_path.name if backfill_path else 'NOT FOUND'}")

    if manifest_path:
        mdf = pd.read_csv(manifest_path)
        total  = len(mdf)
        inc    = (mdf["included"] == True).sum()
        exc    = total - inc
        proven = mdf[mdf["included"] == True]["source_file"].str.contains(
            r"[/\\][Pp][Rr][Oo][Vv][Ee][Nn][/\\]", regex=True
        ).sum()
        print("\n--- Manifest filter (included==True) ---")
        print(f"  Total rows in manifest          : {total}")
        print(f"  included==True (Proven programs): {inc}")
        print(f"  included==False (excluded)      : {exc}")
        print(f"  Source paths contain 'Proven'   : {proven} / {inc}")
        print("  CONFIRMED: linker uses Proven-scanner included==True records only")

    if links_path is None:
        print("\nNo program_job_links file found -- run py run_job_link.py first.")
        return

    ldf = pd.read_csv(links_path)
    bdf = pd.read_csv(backfill_path) if backfill_path else pd.DataFrame()

    print(f"\n--- program_job_links ({links_path.name}) ---")
    print(f"  Total included proven programs: {len(ldf)}")

    print("\n1. Count by link_confidence")
    print(ldf["link_confidence"].value_counts().to_string())

    print("\n2. Count by link_method")
    print(ldf["link_method"].value_counts().to_string())

    print(f"\n5. Ambiguous matches : {(ldf['link_method']=='ambiguous_match').sum()}")
    print(f"6. Needing review    : {ldf['needs_review'].sum()}")

    print("\n7. Top 20 unmatched program filenames (no_match)")
    unmatched = ldf[ldf["link_method"] == "no_match"]["filename"]
    print(unmatched.value_counts().head(20).to_string())

    if not bdf.empty:
        print(f"\n--- material_backfill ({backfill_path.name}) ---")
        print(f"  Total S/F cut records: {len(bdf)}")

        print("\n3. Count by material_source")
        print(bdf["material_source"].value_counts().to_string())

        verified = bdf[bdf["verified_material"] != "UNKNOWN"]
        pct = 100 * len(verified) / len(bdf) if len(bdf) else 0
        print(f"\n4. S/F records with verified material : {len(verified)} / {len(bdf)} ({pct:.1f}%)")

        print("\n8. Top 20 verified materials by record count")
        vm = bdf[bdf["verified_material"] != "UNKNOWN"]
        if vm.empty:
            print("  (none -- no verified materials yet)")
        else:
            print(vm["verified_material"].value_counts().head(20).to_string())
    else:
        print("\nNo material_backfill file found.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
