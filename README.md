# Proven Program Engine

A deterministic scanner and parser for extracting **real, proven** spindle speeds
and feedrates from production CNC programs.

---

## Purpose

Reads proven CNC programs from production machine folders and builds a searchable
database of:

- Tool numbers and descriptions
- Spindle speeds (S values, with G96/G97 mode)
- Feedrates (F values)
- Source program names and machine folders
- Surrounding code context for every extracted value

> **All data is real, proven production machining data.**
> No values are inferred, estimated, or fabricated.

---

## Project Structure

```
proven_program_engine/
├── src/
│   ├── utils.py        # Logging and file-reading utilities
│   ├── scanner.py      # Proven-folder scanner and manifest exporter
│   └── parser.py       # CNC program parser (T/S/F extraction)
├── exports/            # Manifest CSV/JSON and cuts CSV outputs
├── logs/               # Daily rotating log files
├── tests/              # Pytest test suite
│   ├── conftest.py
│   ├── test_scanner.py
│   └── test_parser.py
├── data/
│   └── sample_programs/
│       └── SAMPLE001.NC   # Sample program for local testing
├── run_scan.py         # Entry point: scan P:\ and export manifest
├── run_parse.py        # Entry point: parse files from manifest
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Scan for proven programs

```
python run_scan.py
```

Scans `P:\*\Proven\` and writes:
- `exports/manifest_YYYYMMDD_HHMMSS.csv`
- `exports/manifest_YYYYMMDD_HHMMSS.json`

### 3. Parse programs

```
python run_parse.py
```

Reads the latest manifest and parses all included files, writing:
- `exports/cuts_YYYYMMDD_HHMMSS.csv`

### 4. Run tests

```
pytest tests/ -v
```

---

## Scanner Details

| Setting | Value |
|---------|-------|
| Scan root | `P:\` |
| Target pattern | `P:\{machine_folder}\Proven\` (case-insensitive) |
| Date filter | Modified within last 2 years (730 days) |
| Allowed extensions | `.EIA .NC .TXT .MIN .TAP .MPF .SPF .MAZ .PGM` |

---

## Manifest Fields

| Field | Description |
|-------|-------------|
| `program_id` | Sequential integer ID |
| `source_file` | Full path to the program file |
| `relative_path` | Path relative to `P:\` |
| `machine_folder` | Name of the machine folder |
| `filename` | Filename only |
| `extension` | Uppercased extension |
| `modified_datetime` | ISO 8601 last-modified timestamp |
| `file_size_bytes` | File size in bytes |
| `included` | `True` if the file passed all filters |
| `skip_reason` | Why the file was excluded (if applicable) |

---

## Parser Output Fields (cuts CSV)

| Field | Description |
|-------|-------------|
| `record_id` | Sequential within this export |
| `program_id` | From the manifest |
| `source_file` | Full path to source file |
| `machine_folder` | Machine folder name |
| `filename` | Filename |
| `line_number` | 1-based line number of the extracted value |
| `active_t_code` | Most recently seen T code (e.g. `T0101`) |
| `tool_number` | Extracted tool slot (e.g. `01`) |
| `tool_description` | Comment from the tool-change line |
| `s_value` | Spindle speed (float, or blank if none on this line) |
| `s_mode` | `CSS` (G96) or `RPM` (G97) |
| `f_value` | Feedrate (float, or blank if none on this line) |
| `raw_line` | The exact CNC line as it appears in the file |
| `context_json` | JSON array — ±3 lines of surrounding code |

---

## Parser Extraction Rules

- One record is emitted per line that contains at least one S or F value.
- Comments `( )` and `;` are stripped before extraction — values inside comments
  are never extracted.
- T codes are detected with `(?<![A-Za-z])T(\d{1,4})(?!\d)` — handles 4-digit
  Fanuc codes (T0101), 2-digit (T01), and T codes followed immediately by M codes
  (T0101M06).
- G96 sets mode `CSS` (constant surface speed); G97 sets mode `RPM`.
  Mode persists until the next G96/G97 is seen.
- Material is never inferred. `material` is not a field.

---

## Important Safety Notes

- **Read-only against `P:\`** — this system never writes, moves, or deletes
  production files.
- No network calls, no external APIs, no machine learning.
- Fully deterministic and auditable.

---

## Phase Roadmap

| Phase | Goal |
|-------|------|
| **1 (current)** | Scanner + manifest + basic parser |
| **2** | Tool-centric database with search |
| **3** | Reporting, comparison, and range views |
