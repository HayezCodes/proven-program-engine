"""
conftest.py — Shared pytest fixtures for Proven Program Engine tests.
"""

import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Sample CNC program — covers all core parser features
# ---------------------------------------------------------------------------

SAMPLE_PROGRAM = """\
%
O1234 (SAMPLE TURNING PROGRAM)
(MATERIAL: NOT SPECIFIED)
(-------------------------------------------)
(TOOL 01 - DNMG 432 ROUGHING INSERT)
T0101
G97 S450 M03
G00 X3.5 Z0.1 M08
G71 U0.100 R0.050
G71 P100 Q200 F0.015
N100 G01 Z-2.0 F0.015
N200 G01 X3.5
G00 X10.0 Z2.0
(-------------------------------------------)
(TOOL 02 - CCMT 32.51 FINISH INSERT)
T0202
G96 S600 M03
G70 P100 Q200 F0.006
G00 X10.0 Z2.0
(-------------------------------------------)
(TOOL 03 - GROOVE TOOL 0.125)
T0303
G97 S1200 M03
G00 X2.0 Z-3.5 M08
G01 X1.5 F0.004
G04 X0.5
G01 X2.0 F0.015
G00 X10.0 Z2.0
M30
%
"""

# Program with G92 spindle limit, G94/G95 feed modes, block-skip lines
SAMPLE_PROGRAM_ADVANCED = """\
%
O2000 (ADVANCED FEATURES TEST)
/N10 G00 G20 G40 G90 G95 F.01
G92 S1200
(TOOL 01 - DRILL)
T0101
G97 S800 M03
G95 G01 Z-1.0 F0.008
G94 G01 X1.0 F12.0
/ G97 S500 M03
G04 F2.5
M30
%
"""

# Program where description is on the line BEFORE the T code
SAMPLE_PROGRAM_LOOKBACK = """\
%
O3000 (LOOKBACK TEST)
(T05 - 1/2 DRILL BIT)
T0505
G97 S1500 M03
G01 Z-2.0 F0.010
M30
%
"""

# .OP1 style program (same syntax, different extension)
SAMPLE_OP1_PROGRAM = """\
%
O4000 (OP1 TEST PROGRAM)
T0101
G97 S900 M03
G01 Z-1.0 F0.012
M30
%
"""


@pytest.fixture
def sample_nc_file(tmp_path: Path) -> Path:
    """Write the standard sample program to a temp .NC file."""
    nc_file = tmp_path / "SAMPLE001.NC"
    nc_file.write_text(SAMPLE_PROGRAM, encoding="utf-8")
    return nc_file


@pytest.fixture
def sample_advanced_nc_file(tmp_path: Path) -> Path:
    """Write the advanced sample program (G92, G94/G95, block-skip) to a temp .NC file."""
    nc_file = tmp_path / "ADVANCED001.NC"
    nc_file.write_text(SAMPLE_PROGRAM_ADVANCED, encoding="utf-8")
    return nc_file


@pytest.fixture
def sample_lookback_nc_file(tmp_path: Path) -> Path:
    """Write the lookback test program to a temp .NC file."""
    nc_file = tmp_path / "LOOKBACK001.NC"
    nc_file.write_text(SAMPLE_PROGRAM_LOOKBACK, encoding="utf-8")
    return nc_file


@pytest.fixture
def sample_op1_file(tmp_path: Path) -> Path:
    """Write the OP1 test program to a .OP1 file."""
    op_file = tmp_path / "PART001.OP1"
    op_file.write_text(SAMPLE_OP1_PROGRAM, encoding="utf-8")
    return op_file


@pytest.fixture
def proven_dir_structure(tmp_path: Path) -> Path:
    """Create a mock P:\\ directory tree with two Proven folders.

    Layout:
        tmp_path / 421, 423, 424 / Proven / PART001.NC
        tmp_path / 432, 437      / proven / PART002.EIA
    """
    for machine, proven_name, filename in [
        ("421, 423, 424", "Proven", "PART001.NC"),
        ("432, 437",      "proven", "PART002.EIA"),
    ]:
        folder = tmp_path / machine / proven_name
        folder.mkdir(parents=True)
        (folder / filename).write_text(SAMPLE_PROGRAM, encoding="utf-8")

    return tmp_path


@pytest.fixture
def proven_dir_with_op_files(tmp_path: Path) -> Path:
    """Proven folder structure that includes .OP1, .OP2, .OP3 files."""
    folder = tmp_path / "652" / "Proven"
    folder.mkdir(parents=True)
    for ext, content in [
        ("PART001.OP1", SAMPLE_OP1_PROGRAM),
        ("PART001.OP2", SAMPLE_OP1_PROGRAM),
        ("PART001.OP3", SAMPLE_OP1_PROGRAM),
        ("PART002.NC",  SAMPLE_PROGRAM),
    ]:
        (folder / ext).write_text(content, encoding="utf-8")
    return tmp_path
