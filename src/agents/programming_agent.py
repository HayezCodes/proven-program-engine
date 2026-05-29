"""
programming_agent.py — Read-only Programming Analyst Agent.

Answers programming questions using proven shop S/F data:
  - What should I start at for 4140 HR HT on a 417 lathe?
  - What are our proven speeds and feeds for this material?
  - How do we normally run this tool type?
  - What is our most common range?
  - Are there any outlier programs?
  - How confident is this recommendation?

READ-ONLY — this agent never modifies data, exports, programs, or routers.
It only reads from proven_sf_lookup_*.csv and proven_sf_database_*.csv.

Future agents (tooling_agent, print_agent, manufacturing_agent) live in the
same package hierarchy once built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """Proven S/F recommendation for a material + optional filters."""

    material: str
    machine: str | None
    tool_type: str | None
    tool_description: str | None

    # Speed range (spindle — RPM or SFM)
    S_low:  float | None
    S_mid:  float | None
    S_high: float | None
    s_mode: str           # "CSS" / "RPM" / "UNKNOWN" (most common in matching groups)

    # Feed range
    F_low:  float | None
    F_mid:  float | None
    F_high: float | None
    f_mode: str           # "IPR" / "IPM" / "UNKNOWN"

    # Evidence
    confidence:            str   # HIGH / MEDIUM / LOW / NONE
    occurrence_count:      int   # total proven cut records matched
    program_count:         int   # distinct programs
    machine_count:         int   # distinct machine folders
    matching_groups:       int   # rows from the lookup table

    # Confidence breakdown
    high_conf_occurrences:   int
    medium_conf_occurrences: int

    # Tool context
    tool_types_represented: list[str]
    top_tool_descriptions:  list[str]

    # Human-readable output
    summary:  str
    warnings: list[str] = field(default_factory=list)


@dataclass
class OutlierRow:
    """One record identified as an S or F outlier within its peer group."""

    source_file:    str
    machine_folder: str
    tool_description: str
    S:  float | None
    F:  float | None
    s_mode: str
    f_mode: str
    group_S_median: float | None
    group_F_median: float | None
    outlier_flags:  list[str]   # human-readable flag strings
    confidence:     str


@dataclass
class OutlierReport:
    """Outlier analysis result for a given filter set."""

    material:     str
    machine:      str | None
    tool_type:    str | None
    total_records_checked: int
    outlier_count: int
    outliers:     list[OutlierRow]
    summary:      str


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProgrammingAgent:
    """Read-only programming analysis agent.

    Loads proven S/F lookup and full database exports, then provides:
      - ``recommend(material, ...)``  — S/F starting ranges with confidence
      - ``detect_outliers(...)``      — flags records outside ±2× group median
      - ``list_materials()``          — available materials in the lookup
      - ``list_tool_types(material)`` — tool types for a material
      - ``list_machines(material)``   — machines with data for a material

    Designed for Streamlit integration and future agent orchestration.
    This class never writes files or modifies any data structure in-place.
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        lookup_df: pd.DataFrame,
        db_df: pd.DataFrame | None = None,
    ) -> None:
        self._lookup: pd.DataFrame = lookup_df.copy() if lookup_df is not None and not lookup_df.empty else pd.DataFrame()
        self._db:     pd.DataFrame = db_df.copy()     if db_df     is not None and not db_df.empty     else pd.DataFrame()

    @classmethod
    def from_exports(cls, exports_dir: Path | None = None) -> "ProgrammingAgent":
        """Instantiate from the latest available export files. Read-only."""
        if exports_dir is None:
            # src/agents/programming_agent.py → parents[2] = project root
            exports_dir = Path(__file__).parents[2] / "exports"

        def _latest(pat: str) -> Path | None:
            try:
                files = sorted(exports_dir.glob(pat), key=lambda p: p.stat().st_mtime)
                return files[-1] if files else None
            except OSError:
                return None

        def _load(pat: str) -> pd.DataFrame:
            p = _latest(pat)
            if p is None:
                return pd.DataFrame()
            try:
                return pd.read_csv(p, low_memory=False)
            except Exception:
                return pd.DataFrame()

        lookup = _load("proven_sf_lookup_*.csv")
        db     = _load("proven_sf_database_*.csv")
        return cls(lookup, db)

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_materials(self) -> list[str]:
        """Return all materials in the lookup (excluding UNKNOWN), sorted."""
        if self._lookup.empty or "material" not in self._lookup.columns:
            return []
        vals = self._lookup["material"].dropna().astype(str).unique()
        return sorted(v for v in vals if v not in ("UNKNOWN", "nan", ""))

    def list_tool_types(self, material: str | None = None) -> list[str]:
        """Return tool types available for a material (or all if None)."""
        df = self._filter_lookup(material=material)
        if df.empty or "tool_type" not in df.columns:
            return []
        return sorted(df["tool_type"].dropna().astype(str).unique().tolist())

    def list_machines(self, material: str | None = None) -> list[str]:
        """Return machine folders with data for a material (or all if None)."""
        df = self._filter_lookup(material=material)
        if df.empty or "machine_folder" not in df.columns:
            return []
        return sorted(df["machine_folder"].dropna().astype(str).unique().tolist())

    # ── Recommendation ────────────────────────────────────────────────────────

    def recommend(
        self,
        material: str,
        machine: str | None = None,
        tool_type: str | None = None,
        tool_description: str | None = None,
    ) -> Recommendation:
        """Return a proven S/F recommendation.

        material is required.  machine, tool_type, and tool_description
        are optional filters that narrow the result set.

        Ranges are computed across all matching lookup groups:
          S_low  = minimum S_low  in matching groups
          S_mid  = occurrence-weighted mean of S_mid across groups
          S_high = maximum S_high in matching groups
          (same logic for F)
        """
        warnings: list[str] = []

        resolved, mat_warn = self._resolve_material(material)
        if mat_warn:
            warnings.append(mat_warn)
        if resolved is None:
            return self._empty_rec(
                material, machine, tool_type, tool_description,
                [f"Material '{material}' not found in lookup data."],
            )

        rows = self._filter_lookup(
            material=resolved,
            machine=machine,
            tool_type=tool_type,
            tool_description=tool_description,
        )

        if rows.empty:
            parts = [f"material='{resolved}'"]
            if machine:          parts.append(f"machine='{machine}'")
            if tool_type:        parts.append(f"tool_type='{tool_type}'")
            if tool_description: parts.append(f"tool='{tool_description}'")
            return self._empty_rec(
                resolved, machine, tool_type, tool_description,
                [f"No proven records found for {', '.join(parts)}."],
            )

        total_occ = int(pd.to_numeric(rows.get("occurrence_count", 1), errors="coerce").fillna(1).sum())
        if total_occ < 5:
            warnings.append(f"Low occurrence count ({total_occ}). Treat as indicative only.")

        S_low, S_mid, S_high = self._agg_ranges(rows, "S_low", "S_mid", "S_high")
        F_low, F_mid, F_high = self._agg_ranges(rows, "F_low", "F_mid", "F_high")

        confidence  = self._best_confidence(rows)
        prog_count  = int(rows.get("program_count", pd.Series(dtype=float)).fillna(0).sum()) if "program_count" in rows.columns else 0
        mach_count  = rows["machine_folder"].nunique() if "machine_folder" in rows.columns else 0
        high_occ    = self._occ_for_conf(rows, "HIGH")
        medium_occ  = self._occ_for_conf(rows, "MEDIUM")
        s_mode      = self._dominant_mode(rows, "s_mode")
        f_mode      = self._dominant_mode(rows, "f_mode")

        tool_types_repr = (
            sorted(rows["tool_type"].dropna().astype(str).unique().tolist())
            if "tool_type" in rows.columns else []
        )
        top_descs = (
            rows.sort_values("occurrence_count", ascending=False)["tool_description"]
            .dropna().astype(str).unique()[:5].tolist()
            if ("tool_description" in rows.columns and "occurrence_count" in rows.columns)
            else []
        )

        f_modes = self._distinct_modes(rows, "f_mode")
        if len(f_modes) > 1:
            warnings.append(
                f"Feed units/modes are mixed ({', '.join(f_modes)}); narrow by tool_type or machine before using the full F range."
            )

        s_modes = self._distinct_modes(rows, "s_mode")
        if len(s_modes) > 1:
            warnings.append(
                f"Spindle modes are mixed ({', '.join(s_modes)}); narrow by tool_type or machine before using the full S range."
            )

        if not any((machine, tool_type, tool_description)) and len(tool_types_repr) > 1:
            warnings.append(
                f"Material-only result spans {len(tool_types_repr)} tool types; filter by tool_type or machine for a narrower recommendation."
            )

        summary = self._fmt_summary(
            resolved, machine, tool_type, tool_types_repr,
            S_low, S_mid, S_high, s_mode,
            F_low, F_mid, F_high, f_mode,
            confidence, total_occ, len(rows),
        )

        return Recommendation(
            material=resolved,
            machine=machine,
            tool_type=tool_type,
            tool_description=tool_description,
            S_low=S_low, S_mid=S_mid, S_high=S_high, s_mode=s_mode,
            F_low=F_low, F_mid=F_mid, F_high=F_high, f_mode=f_mode,
            confidence=confidence,
            occurrence_count=total_occ,
            program_count=prog_count,
            machine_count=mach_count,
            matching_groups=len(rows),
            high_conf_occurrences=high_occ,
            medium_conf_occurrences=medium_occ,
            tool_types_represented=tool_types_repr,
            top_tool_descriptions=top_descs,
            summary=summary,
            warnings=warnings,
        )

    # ── Outlier detection ─────────────────────────────────────────────────────

    def detect_outliers(
        self,
        material: str | None = None,
        machine: str | None = None,
        tool_type: str | None = None,
        s_threshold: float = 2.0,
        f_threshold: float = 2.0,
    ) -> OutlierReport:
        """Flag S/F outliers within peer groups.

        Uses the full SF database if available; falls back to the lookup.

        A record is flagged when its S or F value is:
          > threshold × group median    (high outlier)
          < group median / threshold    (low outlier)

        Groups require at least 3 records before flagging outliers.
        Returns an OutlierReport. Nothing is modified.
        """
        source = self._db if not self._db.empty else self._lookup
        if source.empty:
            return OutlierReport(
                material=material or "all", machine=machine, tool_type=tool_type,
                total_records_checked=0, outlier_count=0, outliers=[],
                summary="No data available for outlier detection.",
            )

        df = source.copy()

        # ── Apply filters ────────────────────────────────────────────────
        if material:
            mat_col = "verified_material" if "verified_material" in df.columns else "material"
            df = df[df[mat_col].astype(str).str.contains(material, case=False, na=False)]

        if machine and "machine_folder" in df.columns:
            df = df[df["machine_folder"].astype(str).str.contains(machine, case=False, na=False)]

        # Ensure tool_type column exists (classify on demand for full DB)
        if "tool_type" not in df.columns and "resolved_tool_name" in df.columns:
            from src.tool_classifier import classify_tool_type
            df = df.copy()
            df["tool_type"] = df["resolved_tool_name"].fillna("").astype(str).apply(classify_tool_type)

        if tool_type and "tool_type" in df.columns:
            df = df[df["tool_type"].astype(str) == tool_type]

        if df.empty:
            return OutlierReport(
                material=material or "all", machine=machine, tool_type=tool_type,
                total_records_checked=0, outlier_count=0, outliers=[],
                summary="No matching records found.",
            )

        # S and F column names differ between full DB and lookup
        s_col = "S"      if "S"      in df.columns else "S_mid"
        f_col = "F"      if "F"      in df.columns else "F_mid"
        name_col = "resolved_tool_name" if "resolved_tool_name" in df.columns else "tool_description"

        group_keys = [c for c in ("machine_folder", "tool_type", "s_mode", "f_mode") if c in df.columns]

        outlier_rows: list[OutlierRow] = []

        groups = df.groupby(group_keys, dropna=False) if group_keys else [("_all_", df)]
        for _, grp in groups:
            s_vals = pd.to_numeric(grp.get(s_col, pd.Series(dtype=float)), errors="coerce")
            f_vals = pd.to_numeric(grp.get(f_col, pd.Series(dtype=float)), errors="coerce")

            s_med = s_vals.dropna().median() if s_vals.dropna().__len__() >= 3 else None
            f_med = f_vals.dropna().median() if f_vals.dropna().__len__() >= 3 else None

            for idx, row in grp.iterrows():
                flags: list[str] = []
                sv = pd.to_numeric(row.get(s_col, None), errors="coerce")
                fv = pd.to_numeric(row.get(f_col, None), errors="coerce")

                if s_med is not None and pd.notna(sv):
                    if sv > s_threshold * s_med:
                        flags.append(
                            f"S_above_{s_threshold}x_median "
                            f"({sv:.0f} vs median {s_med:.0f})"
                        )
                    elif sv < s_med / s_threshold:
                        flags.append(
                            f"S_below_{int(100/s_threshold)}pct_median "
                            f"({sv:.0f} vs median {s_med:.0f})"
                        )

                if f_med is not None and pd.notna(fv):
                    if fv > f_threshold * f_med:
                        flags.append(
                            f"F_above_{f_threshold}x_median "
                            f"({fv:.5f} vs median {f_med:.5f})"
                        )
                    elif fv < f_med / f_threshold:
                        flags.append(
                            f"F_below_{int(100/f_threshold)}pct_median "
                            f"({fv:.5f} vs median {f_med:.5f})"
                        )

                if flags:
                    outlier_rows.append(OutlierRow(
                        source_file=str(row.get("source_file", "")),
                        machine_folder=str(row.get("machine_folder", "")),
                        tool_description=str(row.get(name_col, "")),
                        S=float(sv) if pd.notna(sv) else None,
                        F=float(fv) if pd.notna(fv) else None,
                        s_mode=self._display_mode(row.get("s_mode", "")),
                        f_mode=self._display_mode(row.get("f_mode", "")),
                        group_S_median=float(s_med) if s_med is not None else None,
                        group_F_median=float(f_med) if f_med is not None else None,
                        outlier_flags=flags,
                        confidence=str(row.get("sf_record_confidence", row.get("confidence", ""))),
                    ))

        summary = (
            f"Checked {len(df):,} records"
            + (f" for '{material}'" if material else "")
            + f". Found {len(outlier_rows):,} outlier(s) "
            f"(threshold: S>{s_threshold}×median or <{int(100/s_threshold)}%, "
            f"F>{f_threshold}×median or <{int(100/f_threshold)}%)."
        )

        return OutlierReport(
            material=material or "all",
            machine=machine,
            tool_type=tool_type,
            total_records_checked=len(df),
            outlier_count=len(outlier_rows),
            outliers=outlier_rows,
            summary=summary,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _filter_lookup(
        self,
        material: str | None = None,
        machine: str | None = None,
        tool_type: str | None = None,
        tool_description: str | None = None,
    ) -> pd.DataFrame:
        df = self._lookup
        if df.empty:
            return df
        if material and "material" in df.columns:
            df = df[df["material"].astype(str) == material]
        if machine and "machine_folder" in df.columns:
            df = df[df["machine_folder"].astype(str).str.contains(machine, case=False, na=False)]
        if tool_type and "tool_type" in df.columns:
            df = df[df["tool_type"].astype(str) == tool_type]
        if tool_description and "tool_description" in df.columns:
            df = df[df["tool_description"].astype(str).str.contains(
                tool_description, case=False, na=False
            )]
        return df

    def _resolve_material(self, material: str) -> tuple[str | None, str | None]:
        """Return (resolved, warning_or_None). Supports exact / case-insensitive / substring."""
        available = self.list_materials()
        if not available:
            return None, "Lookup data is empty — run py run_build_sf_database.py."
        if material in available:
            return material, None
        low = material.lower()
        for m in available:
            if m.lower() == low:
                return m, None
        matches = [m for m in available if low in m.lower() or m.lower() in low]
        if len(matches) == 1:
            return matches[0], f"Resolved '{material}' → '{matches[0]}'."
        if len(matches) > 1:
            return matches[0], (
                f"'{material}' matched multiple materials "
                f"({[m for m in matches[:4]]}); using '{matches[0]}'."
            )
        return None, None

    def _agg_ranges(
        self,
        rows: pd.DataFrame,
        low_col: str,
        mid_col: str,
        high_col: str,
    ) -> tuple[float | None, float | None, float | None]:
        """Aggregate low/mid/high across lookup rows, weighted by occurrence_count."""
        weights = pd.to_numeric(
            rows.get("occurrence_count", pd.Series(dtype=float)), errors="coerce"
        ).fillna(1.0)

        lo = pd.to_numeric(rows.get(low_col,  pd.Series(dtype=float)), errors="coerce")
        mi = pd.to_numeric(rows.get(mid_col,  pd.Series(dtype=float)), errors="coerce")
        hi = pd.to_numeric(rows.get(high_col, pd.Series(dtype=float)), errors="coerce")

        low_val  = float(lo.min())  if lo.notna().any() else None
        high_val = float(hi.max())  if hi.notna().any() else None

        mask = mi.notna()
        if mask.any():
            w = weights[mask]
            mid_val = float((mi[mask] * w).sum() / w.sum()) if w.sum() > 0 else float(mi[mask].median())
        else:
            mid_val = None

        def _r(v: float | None, decimals: int) -> float | None:
            return round(v, decimals) if v is not None else None

        # S values: round to whole numbers; F values: 5 decimal places
        decimals = 0 if "S" in mid_col.upper() else 5
        return _r(low_val, decimals), _r(mid_val, decimals), _r(high_val, decimals)

    def _best_confidence(self, rows: pd.DataFrame) -> str:
        if "confidence" not in rows.columns:
            return "LOW"
        vals = set(rows["confidence"].dropna().astype(str).str.upper())
        for lvl in ("HIGH", "MEDIUM", "LOW"):
            if lvl in vals:
                return lvl
        return "LOW"

    def _occ_for_conf(self, rows: pd.DataFrame, level: str) -> int:
        if "confidence" not in rows.columns or "occurrence_count" not in rows.columns:
            return 0
        mask = rows["confidence"].astype(str).str.upper() == level
        return int(pd.to_numeric(rows.loc[mask, "occurrence_count"], errors="coerce").fillna(0).sum())

    def _dominant_mode(self, rows: pd.DataFrame, col: str) -> str:
        if col not in rows.columns:
            return "UNKNOWN"
        modes = rows[col].apply(self._display_mode)
        occ_col = "occurrence_count"
        if occ_col in rows.columns:
            mode_rows = pd.DataFrame({
                "mode": modes,
                "occurrence_count": pd.to_numeric(rows[occ_col], errors="coerce").fillna(1),
            })
            agg = mode_rows.groupby("mode")["occurrence_count"].sum().sort_values(ascending=False)
        else:
            agg = modes.value_counts()
        if agg.empty:
            return "UNKNOWN"
        return str(agg.index[0])

    def _distinct_modes(self, rows: pd.DataFrame, col: str) -> list[str]:
        if col not in rows.columns:
            return []
        return sorted(rows[col].apply(self._display_mode).unique().tolist())

    def _display_mode(self, val) -> str:
        try:
            if pd.isna(val):
                return "UNKNOWN"
        except (TypeError, ValueError):
            pass
        mode = str(val or "").strip().upper()
        return "UNKNOWN" if mode in ("", "UNKNOWN", "NAN", "NONE") else mode

    def _fmt_summary(
        self,
        material: str,
        machine: str | None,
        tool_type: str | None,
        tool_types_repr: list[str],
        S_low: float | None, S_mid: float | None, S_high: float | None, s_mode: str,
        F_low: float | None, F_mid: float | None, F_high: float | None, f_mode: str,
        confidence: str,
        occurrence_count: int,
        matching_groups: int,
    ) -> str:
        ctx = [f"Material: {material}"]
        if machine:
            ctx.append(f"Machine: {machine}")
        if tool_type:
            ctx.append(f"Tool type: {tool_type}")
        elif tool_types_repr:
            ctx.append(f"Tool types: {', '.join(tool_types_repr[:3])}")

        s_str = ""
        if S_mid is not None:
            s_str = f"S {S_low:.0f}–{S_high:.0f} (mid {S_mid:.0f})"
            if s_mode:
                s_str += f" {s_mode}"

        f_str = ""
        if F_mid is not None:
            f_str = f"F {F_low:.5f}–{F_high:.5f} (mid {F_mid:.5f})"
            if f_mode:
                f_str += f" {f_mode}"

        sf_line = "  |  ".join(filter(None, [s_str, f_str])) or "No S/F data in matching groups."

        return (
            f"{' | '.join(ctx)}\n"
            f"Proven range: {sf_line}\n"
            f"Confidence: {confidence}  |  "
            f"Occurrences: {occurrence_count:,}  |  "
            f"Groups: {matching_groups}"
        )

    def _empty_rec(
        self,
        material: str,
        machine: str | None,
        tool_type: str | None,
        tool_description: str | None,
        warnings: list[str],
    ) -> Recommendation:
        return Recommendation(
            material=material, machine=machine,
            tool_type=tool_type, tool_description=tool_description,
            S_low=None, S_mid=None, S_high=None, s_mode="UNKNOWN",
            F_low=None, F_mid=None, F_high=None, f_mode="UNKNOWN",
            confidence="NONE",
            occurrence_count=0, program_count=0,
            machine_count=0, matching_groups=0,
            high_conf_occurrences=0, medium_conf_occurrences=0,
            tool_types_represented=[], top_tool_descriptions=[],
            summary=f"No data found for '{material}'.",
            warnings=warnings,
        )
