"""Tests for src/tooldb_loader.py"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.tooldb_loader import (
    _ansi_code_to_shape,
    _clean,
    _extract_machine_ids,
    _has_material,
    _is_active_tooldb,
    export_tooldb_reference,
    load_all_tooldb,
    load_tooldb,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> bytes:
    return os.urandom(16)


def _make_tooldb(path: Path, mill_tools=None, lathe_tools=None, add_material=False) -> None:
    """Create a minimal TOOLDB SQLite file for testing.

    mill_tools: list of dicts with keys:
        tool_number, tool_station, name, holder_name,
        diameter, flute_count, mc_tool_type, feed_rate, spindle_rpm, is_metric
    lathe_tools: list of dicts with keys:
        tool_number, tool_station, name, holder_name,
        ansi_code, ic_diameter, corner_radius, is_metric
    """
    mill_tools = mill_tools or []
    lathe_tools = lathe_tools or []

    con = sqlite3.connect(str(path))

    con.executescript("""
        CREATE TABLE TlAssembly (
            ID BLOB PRIMARY KEY,
            ToolNumber INTEGER DEFAULT 0,
            IsMetric INTEGER DEFAULT 0,
            MainTool BLOB,
            MainHolder BLOB,
            Name TEXT DEFAULT '',
            Description TEXT DEFAULT ''
        );
        CREATE TABLE TlAssemblyItem (
            ID BLOB PRIMARY KEY,
            Name TEXT DEFAULT '',
            Description TEXT DEFAULT ''
        );
        CREATE TABLE TlTool (
            ID BLOB PRIMARY KEY,
            ToolNumber INTEGER DEFAULT 0,
            ToolStation INTEGER DEFAULT 0
        );
        CREATE TABLE TlToolMill (
            ID BLOB PRIMARY KEY,
            MCToolType INTEGER DEFAULT 0,
            OverallDiameter REAL DEFAULT 0.0,
            FluteCount INTEGER DEFAULT 0,
            TlOpParamsID BLOB,
            DiameterOffsetNum INTEGER DEFAULT 0,
            LengthOffsetNum INTEGER DEFAULT 0
        );
        CREATE TABLE TlToolLathe (
            ID BLOB PRIMARY KEY,
            HolderID BLOB,
            InsertID BLOB,
            ToolWidth REAL DEFAULT 0.0,
            ToolHeight REAL DEFAULT 0.0,
            ToolClearanceAngle REAL DEFAULT 0.0,
            ToolRakeAngle REAL DEFAULT 0.0,
            MachineSideConnectionType INTEGER DEFAULT 0,
            RequiresToolLocator INTEGER DEFAULT 0
        );
        CREATE TABLE TlOpParams (
            ID BLOB PRIMARY KEY,
            FeedRate REAL DEFAULT 0.0,
            SpindleSpeed REAL DEFAULT 0,
            MaterialSFMAdjust REAL DEFAULT 100.0,
            MaterialFPTAdjust REAL DEFAULT 100.0,
            RetractRate REAL DEFAULT 50.0,
            PlungeRate REAL DEFAULT 25.0,
            IsMetric INTEGER DEFAULT 0
        );
        CREATE TABLE TlInsert (
            ID BLOB PRIMARY KEY,
            AnsiShapeCode TEXT DEFAULT '',
            ICDiameter REAL DEFAULT 0.0,
            CornerRadius REAL DEFAULT 0.0,
            Thickness REAL DEFAULT 0.0,
            Length REAL DEFAULT 0.0,
            Width REAL DEFAULT 0.0,
            TlGradeID BLOB,
            IsCustom INTEGER DEFAULT 0,
            Is3d INTEGER DEFAULT 0,
            CuttingSide INTEGER DEFAULT 0
        );
        CREATE TABLE TlWorkMaterial (
            ID BLOB PRIMARY KEY,
            ISOGroup TEXT DEFAULT '',
            Density REAL DEFAULT 0.0,
            HardnessUnit INTEGER DEFAULT 0,
            HardnessValue REAL DEFAULT 0.0
        );
        CREATE TABLE TlHolder (
            ID BLOB PRIMARY KEY,
            HolderType INTEGER DEFAULT 0
        );
    """)

    for t in mill_tools:
        tool_id = _new_id()
        holder_item_id = _new_id()
        asm_id = _new_id()
        op_id = _new_id()

        con.execute(
            "INSERT INTO TlOpParams(ID, FeedRate, SpindleSpeed) VALUES (?,?,?)",
            (op_id, t.get("feed_rate", 10.0), t.get("spindle_rpm", 1000)),
        )
        con.execute(
            "INSERT INTO TlToolMill(ID, MCToolType, OverallDiameter, FluteCount, TlOpParamsID)"
            " VALUES (?,?,?,?,?)",
            (tool_id, t.get("mc_tool_type", 19), t.get("diameter", 0.5),
             t.get("flute_count", 4), op_id),
        )
        con.execute(
            "INSERT INTO TlAssemblyItem(ID, Name) VALUES (?,?)",
            (tool_id, t.get("name", "Tool")),
        )
        con.execute(
            "INSERT INTO TlTool(ID, ToolNumber, ToolStation) VALUES (?,?,?)",
            (tool_id, t.get("tool_number", 1), t.get("tool_station", 0)),
        )
        con.execute(
            "INSERT INTO TlAssemblyItem(ID, Name) VALUES (?,?)",
            (holder_item_id, t.get("holder_name", "Default Holder")),
        )
        con.execute(
            "INSERT INTO TlAssembly(ID, ToolNumber, IsMetric, MainTool, MainHolder)"
            " VALUES (?,?,?,?,?)",
            (asm_id, t.get("tool_number", 1), int(t.get("is_metric", False)),
             tool_id, holder_item_id),
        )

    for t in lathe_tools:
        tool_id = _new_id()
        holder_item_id = _new_id()
        holder_geom_id = _new_id()
        insert_id = _new_id()
        asm_id = _new_id()

        con.execute(
            "INSERT INTO TlInsert(ID, AnsiShapeCode, ICDiameter, CornerRadius) VALUES (?,?,?,?)",
            (insert_id, t.get("ansi_code", "68"),
             t.get("ic_diameter", 0.375), t.get("corner_radius", 0.016)),
        )
        con.execute(
            "INSERT INTO TlHolder(ID) VALUES (?)",
            (holder_geom_id,),
        )
        con.execute(
            "INSERT INTO TlToolLathe(ID, HolderID, InsertID) VALUES (?,?,?)",
            (tool_id, holder_geom_id, insert_id),
        )
        con.execute(
            "INSERT INTO TlAssemblyItem(ID, Name) VALUES (?,?)",
            (tool_id, t.get("name", "Lathe Tool")),
        )
        con.execute(
            "INSERT INTO TlTool(ID, ToolNumber, ToolStation) VALUES (?,?,?)",
            (tool_id, t.get("tool_number", 1), t.get("tool_station", 0)),
        )
        con.execute(
            "INSERT INTO TlAssemblyItem(ID, Name) VALUES (?,?)",
            (holder_item_id, t.get("holder_name", "Default Holder")),
        )
        con.execute(
            "INSERT INTO TlAssembly(ID, ToolNumber, IsMetric, MainTool, MainHolder)"
            " VALUES (?,?,?,?,?)",
            (asm_id, t.get("tool_number", 1), int(t.get("is_metric", False)),
             tool_id, holder_item_id),
        )

    if add_material:
        con.execute(
            "INSERT INTO TlWorkMaterial(ID, ISOGroup) VALUES (?,?)",
            (_new_id(), "P"),
        )

    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Unit: _extract_machine_ids
# ---------------------------------------------------------------------------

def test_extract_single_id():
    assert _extract_machine_ids(Path("655.tooldb")) == ["655"]


def test_extract_multiple_ids():
    assert _extract_machine_ids(Path("421, 423, 424.tooldb")) == ["421", "423", "424"]


def test_extract_id_with_text():
    assert _extract_machine_ids(Path("652 Makino.TOOLDB")) == ["652"]


def test_extract_no_id():
    assert _extract_machine_ids(Path("ARROW TOOLS.TOOLDB")) == []
    assert _extract_machine_ids(Path("Mill_Inch.tooldb")) == []


def test_extract_multi_with_comma_spacing():
    assert _extract_machine_ids(Path("430, 431.tooldb")) == ["430", "431"]


def test_extract_four_digit_id():
    assert _extract_machine_ids(Path("1234.tooldb")) == ["1234"]


# ---------------------------------------------------------------------------
# Unit: _is_active_tooldb
# ---------------------------------------------------------------------------

def test_is_active_lower():
    assert _is_active_tooldb(Path("432.tooldb")) is True


def test_is_active_upper():
    assert _is_active_tooldb(Path("429.TOOLDB")) is True


def test_not_active_bak():
    assert _is_active_tooldb(Path("432.tooldbbak")) is False


def test_not_active_variant_x():
    assert _is_active_tooldb(Path("432.tooldbX15v28")) is False


def test_not_active_dot_bak():
    assert _is_active_tooldb(Path("432.bak")) is False


def test_not_active_uppercase_variant():
    assert _is_active_tooldb(Path("432.TOOLDBX16v29")) is False


# ---------------------------------------------------------------------------
# Unit: _clean, _ansi_code_to_shape
# ---------------------------------------------------------------------------

def test_clean_strips_whitespace():
    assert _clean("  ENDMILL  ") == "ENDMILL"


def test_clean_none_returns_empty():
    assert _clean(None) == ""


def test_ansi_code_d():
    assert _ansi_code_to_shape("68") == "D"


def test_ansi_code_c():
    assert _ansi_code_to_shape("67") == "C"


def test_ansi_code_g():
    assert _ansi_code_to_shape("71") == "G"


def test_ansi_code_invalid_passthrough():
    assert _ansi_code_to_shape("XX") == "XX"


def test_ansi_code_empty():
    assert _ansi_code_to_shape("") == ""


def test_ansi_code_none():
    assert _ansi_code_to_shape(None) == ""


# ---------------------------------------------------------------------------
# Unit: _has_material
# ---------------------------------------------------------------------------

def test_has_material_false(tmp_path):
    db = tmp_path / "test.tooldb"
    _make_tooldb(db, add_material=False)
    con = sqlite3.connect(str(db))
    assert _has_material(con) is False
    con.close()


def test_has_material_true(tmp_path):
    db = tmp_path / "test.tooldb"
    _make_tooldb(db, add_material=True)
    con = sqlite3.connect(str(db))
    assert _has_material(con) is True
    con.close()


# ---------------------------------------------------------------------------
# load_tooldb — mill tools
# ---------------------------------------------------------------------------

def test_load_mill_tool_basic(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{
        "tool_number": 1, "name": "1/2 ENDMILL",
        "diameter": 0.5, "flute_count": 4, "mc_tool_type": 19,
        "feed_rate": 25.0, "spindle_rpm": 2200,
    }])
    df = load_tooldb(db)
    assert not df.empty
    row = df.iloc[0]
    assert row["tool_number"] == 1
    assert row["tool_name"] == "1/2 ENDMILL"
    assert row["tool_category"] == "mill"
    assert row["overall_diameter"] == pytest.approx(0.5)
    assert row["flute_count"] == 4
    assert row["feed_rate_ipm"] == pytest.approx(25.0)
    assert row["spindle_rpm"] == 2200


def test_load_mill_tool_sf_not_usable(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "DRILL"}])
    df = load_tooldb(db)
    row = df.iloc[0]
    assert row["sf_usable"] is False
    assert row["sf_reject_reason"] == "missing_material"


def test_load_mill_tool_sf_usable_when_material_present(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "DRILL"}], add_material=True)
    df = load_tooldb(db)
    row = df.iloc[0]
    assert row["sf_usable"] is True
    assert row["sf_reject_reason"] == ""


def test_load_mill_source_tooldb(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "X"}])
    df = load_tooldb(db)
    assert df.iloc[0]["source_tooldb"] == "655"


def test_load_mill_machine_id(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "X"}])
    df = load_tooldb(db)
    assert df.iloc[0]["machine_id"] == "655"


def test_load_mill_no_machine_id(tmp_path):
    db = tmp_path / "Mill_Inch.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "X"}])
    df = load_tooldb(db)
    assert df.iloc[0]["machine_id"] == ""


def test_load_mill_multiple_machine_ids(tmp_path):
    db = tmp_path / "421, 423, 424.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "ENDMILL"}])
    df = load_tooldb(db)
    assert len(df) == 3
    assert set(df["machine_id"].tolist()) == {"421", "423", "424"}
    assert all(df["tool_name"] == "ENDMILL")


def test_load_mill_holder_name(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{
        "tool_number": 1, "name": "ENDMILL",
        "holder_name": "CAT40 ER32",
    }])
    df = load_tooldb(db)
    assert df.iloc[0]["holder_name"] == "CAT40 ER32"


def test_load_mill_columns_present(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "X"}])
    df = load_tooldb(db)
    from src.tooldb_loader import _COLUMN_ORDER
    assert list(df.columns) == _COLUMN_ORDER


def test_load_mill_is_metric_false(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "X", "is_metric": False}])
    df = load_tooldb(db)
    assert df.iloc[0]["is_metric"] is False


def test_load_mill_multiple_tools(tmp_path):
    db = tmp_path / "655.tooldb"
    tools = [
        {"tool_number": 1, "name": "ENDMILL", "diameter": 0.5},
        {"tool_number": 2, "name": "DRILL", "diameter": 0.25},
    ]
    _make_tooldb(db, mill_tools=tools)
    df = load_tooldb(db)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# load_tooldb — lathe tools
# ---------------------------------------------------------------------------

def test_load_lathe_tool_basic(tmp_path):
    db = tmp_path / "432.tooldb"
    _make_tooldb(db, lathe_tools=[{
        "tool_number": 1, "name": "DCMT 3|2.5|1-MF",
        "ansi_code": "68", "ic_diameter": 0.375, "corner_radius": 0.016,
    }])
    df = load_tooldb(db)
    assert not df.empty
    row = df.iloc[0]
    assert row["tool_category"] == "lathe"
    assert row["tool_name"] == "DCMT 3|2.5|1-MF"
    assert row["insert_shape"] == "D"
    assert row["insert_ic_diameter"] == pytest.approx(0.375)
    assert row["insert_corner_radius"] == pytest.approx(0.016)


def test_load_lathe_no_sf(tmp_path):
    db = tmp_path / "432.tooldb"
    _make_tooldb(db, lathe_tools=[{"tool_number": 1, "name": "INSERT"}])
    df = load_tooldb(db)
    row = df.iloc[0]
    assert row["feed_rate_ipm"] is None
    assert row["spindle_rpm"] is None
    assert row["sf_usable"] is False
    assert row["sf_reject_reason"] == "no_sf_data"


def test_load_lathe_no_mill_fields(tmp_path):
    db = tmp_path / "432.tooldb"
    _make_tooldb(db, lathe_tools=[{"tool_number": 1, "name": "INSERT"}])
    df = load_tooldb(db)
    row = df.iloc[0]
    assert row["overall_diameter"] is None
    assert row["flute_count"] is None
    assert row["mc_tool_type"] is None


# ---------------------------------------------------------------------------
# load_tooldb — mixed mill + lathe
# ---------------------------------------------------------------------------

def test_load_mixed_mill_and_lathe(tmp_path):
    db = tmp_path / "432.tooldb"
    _make_tooldb(
        db,
        mill_tools=[{"tool_number": 10, "name": "ENDMILL", "diameter": 0.25}],
        lathe_tools=[{"tool_number": 1, "name": "INSERT"}],
    )
    df = load_tooldb(db)
    assert len(df) == 2
    cats = set(df["tool_category"])
    assert cats == {"mill", "lathe"}


# ---------------------------------------------------------------------------
# load_tooldb — edge cases
# ---------------------------------------------------------------------------

def test_load_missing_file_returns_empty(tmp_path):
    df = load_tooldb(tmp_path / "nonexistent.tooldb")
    assert df.empty


def test_load_empty_tooldb_returns_empty(tmp_path):
    db = tmp_path / "empty.tooldb"
    _make_tooldb(db)
    df = load_tooldb(db)
    assert df.empty


# ---------------------------------------------------------------------------
# load_all_tooldb
# ---------------------------------------------------------------------------

def test_load_all_combines_files(tmp_path):
    _make_tooldb(
        tmp_path / "655.tooldb",
        mill_tools=[{"tool_number": 1, "name": "ENDMILL"}],
    )
    _make_tooldb(
        tmp_path / "432.tooldb",
        lathe_tools=[{"tool_number": 1, "name": "INSERT"}],
    )
    df = load_all_tooldb(tmp_path)
    assert len(df) == 2
    sources = set(df["source_tooldb"])
    assert "655" in sources
    assert "432" in sources


def test_load_all_skips_backup(tmp_path):
    _make_tooldb(
        tmp_path / "655.tooldb",
        mill_tools=[{"tool_number": 1, "name": "ENDMILL"}],
    )
    # Create backup files — should be skipped
    (tmp_path / "655.tooldbbak").write_bytes(b"")
    (tmp_path / "655.tooldbX15v28").write_bytes(b"")
    df = load_all_tooldb(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["source_tooldb"] == "655"


def test_load_all_missing_dir_returns_empty(tmp_path):
    df = load_all_tooldb(tmp_path / "nonexistent")
    assert df.empty


def test_load_all_empty_dir_returns_empty(tmp_path):
    df = load_all_tooldb(tmp_path)
    assert df.empty


def test_load_all_upper_extension(tmp_path):
    _make_tooldb(
        tmp_path / "429.TOOLDB",
        mill_tools=[{"tool_number": 1, "name": "DRILL"}],
    )
    df = load_all_tooldb(tmp_path)
    assert not df.empty
    assert df.iloc[0]["source_tooldb"] == "429"


def test_load_all_multi_machine_filename(tmp_path):
    _make_tooldb(
        tmp_path / "421, 423, 424.tooldb",
        mill_tools=[{"tool_number": 5, "name": "ENDMILL"}],
    )
    df = load_all_tooldb(tmp_path)
    # 3 machine IDs × 1 tool
    assert len(df) == 3
    assert set(df["machine_id"]) == {"421", "423", "424"}


# ---------------------------------------------------------------------------
# export_tooldb_reference
# ---------------------------------------------------------------------------

def test_export_creates_file(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "ENDMILL"}])
    df = load_tooldb(db)

    exports = tmp_path / "exports"
    out = export_tooldb_reference(df, exports, timestamp="20250101_120000")
    assert out.exists()
    assert out.name == "tooldb_reference_20250101_120000.csv"


def test_export_content_matches(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 5, "name": "5/8 ENDMILL", "diameter": 0.625}])
    df = load_tooldb(db)

    exports = tmp_path / "exports"
    out = export_tooldb_reference(df, exports, timestamp="ts")
    loaded = pd.read_csv(out)
    assert loaded.iloc[0]["tool_number"] == 5
    assert loaded.iloc[0]["tool_name"] == "5/8 ENDMILL"
    assert loaded.iloc[0]["overall_diameter"] == pytest.approx(0.625)


def test_export_auto_timestamp(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "X"}])
    df = load_tooldb(db)
    exports = tmp_path / "exports"
    out = export_tooldb_reference(df, exports)
    assert out.name.startswith("tooldb_reference_")
    assert out.name.endswith(".csv")


def test_export_creates_dir(tmp_path):
    db = tmp_path / "655.tooldb"
    _make_tooldb(db, mill_tools=[{"tool_number": 1, "name": "X"}])
    df = load_tooldb(db)
    nested = tmp_path / "a" / "b" / "exports"
    export_tooldb_reference(df, nested, timestamp="t")
    assert nested.exists()
