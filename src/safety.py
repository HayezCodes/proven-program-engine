"""
safety.py — Production write protection for Proven Program Engine.

CRITICAL SAFETY RULE:
  This system is READ-ONLY against all production/shop folders.
  Allowed write locations:
    - The local project folder  (C:\\Users\\Joshua.Hayes\\Desktop\\proven_program_engine)
    - The OS temp directory     (pytest tmp_path isolation only)

  Production folders (P:\\, G:\\, M:\\) and any other external paths
  must NEVER be written to. This system collects data for analysis only.

Usage:
  Call assert_safe_write(path) immediately before any file open / to_csv /
  FileHandler / mkdir operation. It raises ProductionWriteViolation instantly
  if the target is outside the allowed roots.
"""

import tempfile
from pathlib import Path

# Resolved absolute path to the project root (src/ is one level below)
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

# Canonical allowed write sub-directories (informational — the check uses PROJECT_ROOT)
ALLOWED_WRITE_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "exports",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "data" / "overrides",
)


class ProductionWriteViolation(PermissionError):
    """Raised when a write is attempted outside the permitted project folder.

    Inherits from PermissionError so callers that catch PermissionError also
    catch this — but code specifically checking for accidental production writes
    can catch ProductionWriteViolation.
    """


def assert_safe_write(path) -> None:
    """Assert that *path* is within an allowed write location.

    Resolves the path (following symlinks, normalising ..) before comparing so
    that relative paths and path traversal attempts are caught.

    Allowed roots:
      1. PROJECT_ROOT (the local project folder and all sub-directories)
      2. The OS temp directory (used by pytest tmp_path — never in production)

    Raises:
      ProductionWriteViolation  — if the resolved path is outside both roots.

    This function never writes anything itself; it is purely a guard.
    """
    resolved = Path(path).resolve()

    # Allow: anywhere within the project folder
    try:
        resolved.relative_to(PROJECT_ROOT)
        return
    except ValueError:
        pass

    # Allow: OS temp directory (pytest tmp_path, test isolation)
    try:
        resolved.relative_to(Path(tempfile.gettempdir()).resolve())
        return
    except ValueError:
        pass

    raise ProductionWriteViolation(
        f"\n"
        f"  *** PRODUCTION WRITE BLOCKED ***\n"
        f"  Attempted path : {resolved}\n"
        f"  Project root   : {PROJECT_ROOT}\n"
        f"\n"
        f"  This system is READ-ONLY against all production folders.\n"
        f"  Permitted write locations:\n"
        f"    {PROJECT_ROOT / 'exports'}\n"
        f"    {PROJECT_ROOT / 'logs'}\n"
        f"    {PROJECT_ROOT / 'data' / 'overrides'}\n"
        f"\n"
        f"  Production drives (P:\\, G:\\, M:\\) and all shop/job/program\n"
        f"  folders must never be written to. This system collects data\n"
        f"  for analysis only — it never modifies source files."
    )
