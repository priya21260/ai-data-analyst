"""Core data-analysis logic. Pure pandas/numpy - no UI, no AI calls."""
from __future__ import annotations

import csv
import io
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

TEXT_EXTS = (".csv", ".tsv", ".txt")
EXCEL_EXTS = (".xlsx", ".xlsm", ".xls")
SUPPORTED_EXTS = TEXT_EXTS + EXCEL_EXTS

_NA_TOKENS = {"n/a", "nan", "null", "none", "#n/a", "--"}
_SUM_HINT = re.compile(
    r"(sales|revenue|amount|quantity|qty|units|orders|total|profit|cost|spend|"
    r"clicks|views|count|income|expense)",
    re.I,
)
_ID_HINT = re.compile(r"(^|[\s_\-])(id|uuid|guid|key|index|idx)$", re.I)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_py(x: Any) -> Any:
    """Convert numpy/pandas scalars (and containers of them) to plain JSON-friendly Python."""
    if x is None or x is pd.NaT or x is pd.NA:
        return None
    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_py(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def fmt_num(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return "n/a"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.4g}"


def _is_texty(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _to_object(s: pd.Series) -> pd.Series:
    """Object-dtype copy with real NaN for missing (avoids pd.NA surprises downstream)."""
    return s.astype(object).where(s.notna(), np.nan)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def read_table(data: bytes, filename: str) -> dict[str, pd.DataFrame]:
    """Read a CSV/TSV/TXT/Excel upload. Returns {sheet_name: DataFrame}."""
    name = filename.lower()
    if not data:
        raise ValueError("The uploaded file is empty.")

    if name.endswith(TEXT_EXTS):
        text = ""
        for enc in ("utf-8-sig", "cp1252", "latin-1"):  # latin-1 never fails
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        try:
            try:
                df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            except (csv.Error, pd.errors.ParserError):
                try:
                    df = pd.read_csv(io.StringIO(text))
                except pd.errors.ParserError:
                    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        except pd.errors.EmptyDataError as e:
            raise ValueError("The file has no readable data.") from e
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Could not read this file as CSV: {e}") from e
        return {"Data": df}

    if name.endswith(EXCEL_EXTS):
        engine = "xlrd" if name.endswith(".xls") else "openpyxl"
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine=engine)
        except ImportError as e:
            raise ValueError(
                "Reading old .xls files needs the 'xlrd' package. "
                "Install it (pip install xlrd) or re-save the file as .xlsx."
            ) from e
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Could not read this Excel file: {e}") from e
        sheets = {str(k): v for k, v in sheets.items() if not v.dropna(how="all").empty}
        if not sheets:
            raise ValueError("The workbook has no sheets with data.")
        return sheets

    raise ValueError(f"Unsupported file type. Please upload one of: {', '.join(SUPPORTED_EXTS)}")


def _try_numeric(s_obj: pd.Series) -> pd.Series | None:
    nn = s_obj.dropna().astype(str).str.strip()
    if nn.empty:
        return None
    # keep leading-zero codes (zip codes, product codes) as text
    if nn.str.match(r"^0\d+$").mean() > 0.05:
        return None
    cleaned = nn.str.replace(r"[,\s$€£₹%]", "", regex=True)
    conv = pd.to_numeric(cleaned, errors="coerce")
    if conv.notna().mean() < 0.95:
        return None
    full = s_obj.astype(object).where(s_obj.notna(), np.nan)
    full = full.map(lambda v: re.sub(r"[,\s$€£₹%]", "", str(v)) if isinstance(v, str) else v)
    out = pd.to_numeric(full, errors="coerce").astype("float64")
    if out.notna().all() and (out % 1 == 0).all():
        out = out.astype("int64")
    return out


def _try_datetime(s_obj: pd.Series) -> pd.Series | None:
    nn = s_obj.dropna()
    if nn.empty:
        return None
    step = max(1, len(nn) // 300)
    sample = nn.iloc[::step].astype(str)
    if sample.str.contains(r"\d", regex=True).mean() < 0.9:
        return None
    if sample.str.contains(r"[-/:.]|\d\s?[A-Za-z]{3}|[A-Za-z]{3}\s?\d", regex=True).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() < 0.9:
            return None
        full = pd.to_datetime(s_obj, errors="coerce", format="mixed")
    if not pd.api.types.is_datetime64_any_dtype(full):
        return None
    if getattr(full.dt, "tz", None) is not None:
        full = full.dt.tz_localize(None)
    return full


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Tidy column names, trim text, detect numbers/dates stored as text."""
    notes: list[str] = []
    df = df.copy()

    cols, seen = [], {}
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols

    n_cols, n_rows = df.shape[1], df.shape[0]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.shape[1] < n_cols:
        notes.append(f"Removed {n_cols - df.shape[1]} completely empty column(s).")
    if df.shape[0] < n_rows:
        notes.append(f"Removed {n_rows - df.shape[0]} completely empty row(s).")
    df = df.reset_index(drop=True)

    converted_num, converted_dt = [], []
    for col in df.columns:
        s = df[col]
        if not _is_texty(s):
            continue
        s = _to_object(s)
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(
            lambda v: np.nan
            if isinstance(v, str) and (v == "" or v.lower() in _NA_TOKENS)
            else v
        )
        num = _try_numeric(s)
        if num is not None:
            df[col] = num
            converted_num.append(col)
            continue
        dt = _try_datetime(s)
        if dt is not None:
            df[col] = dt
            converted_dt.append(col)
            continue
        df[col] = s

    if converted_num:
        notes.append("Converted text to numbers: " + ", ".join(converted_num))
    if converted_dt:
        notes.append("Detected date columns: " + ", ".join(converted_dt))
    return df, notes


# --------------------------------------------------------------------------- #
# profiling
# --------------------------------------------------------------------------- #
def classify_column(s: pd.Series) -> str:
    """numeric | categorical | datetime | boolean | identifier | text"""
    n = int(s.notna().sum())
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nunique = int(s.nunique(dropna=True))
    if pd.api.types.is_numeric_dtype(s):
        if n > 0 and pd.api.types.is_integer_dtype(s):
            if _ID_HINT.search(str(s.name)) and nunique >= 0.9 * n:
                return "identifier"
            if n >= 20 and nunique == n and s.dropna().is_monotonic_increasing:
                return "identifier"
        return "numeric"
    if n == 0:
        return "text"
    if nunique <= 50 or nunique / n <= 0.05:
        return "categorical"
    avg_len = s.dropna().astype(str).str.len().mean()
    if avg_len > 40:
        return "text"
    if nunique / n > 0.9:
        return "identifier"
    return "categorical"


def choose_freq(span_days: float) -> str:
    if span_days <= 2:
        return "h"
    if span_days <= 60:
        return "D"
    if span_days <= 800:
        return "MS"
    if span_days <= 3650:
        return "QS"
    return "YS"


_PRIORITY = [
    re.compile(r"(revenue|sales|amount|income|profit)", re.I),
    re.compile(r"(cost|spend|expense|total)", re.I),
    re.compile(r"(quantity|qty|units|orders|clicks|views|count)", re.I),
]


def rank_numeric(cols: list[str]) -> list[str]:
    """Put 'business measure' columns (revenue, sales, units...) ahead of ages/ratings/etc."""

    def key(c: str) -> int:
        for i, rx in enumerate(_PRIORITY):
            if rx.search(c):
                return i
        return len(_PRIORITY)

    return sorted(cols, key=key)


def default_agg(col: str) -> str:
    return "sum" if _SUM_HINT.search(col) else "mean"


def aggregate_series(
    df: pd.DataFrame, dt_col: str, val_col: str, freq: str | None = None, agg: str | None = None
) -> tuple[pd.Series, str, str]:
    """Aggregate val_col over time. Returns (series, freq, agg)."""
    d = df[[dt_col, val_col]].dropna()
    agg = agg or default_agg(val_col)
    if d.empty:
        return pd.Series(dtype=float), freq or "D", agg
    span = (d[dt_col].max() - d[dt_col].min()).total_seconds() / 86400
    freq = freq or choose_freq(span)
    s = d.set_index(dt_col)[val_col].sort_index().resample(freq).agg(agg)
    return s, freq, agg


def _trend_for(s: pd.Series) -> dict | None:
    y = s.dropna()
    if len(y) < 4 or y.nunique() < 2:
        return None
    vals = y.values.astype(float)
    x = np.arange(len(vals))
    slope, _ = np.polyfit(x, vals, 1)
    r = np.corrcoef(x, vals)[0, 1]
    r2 = float(r**2) if not np.isnan(r) else 0.0
    mean = float(vals.mean())
    rel = slope / abs(mean) if mean else 0.0
    first, last = float(vals[0]), float(vals[-1])
    change = (last - first) / abs(first) * 100 if first else None
    if r2 < 0.25 or abs(rel) < 0.005:
        direction = "flat / no clear trend"
    else:
        direction = "increasing" if slope > 0 else "decreasing"

    def label(i):
        return str(i.date()) if hasattr(i, "date") else str(i)

    return {
        "periods": int(len(vals)),
        "direction": direction,
        "first_value": first,
        "last_value": last,
        "change_pct": change,
        "slope_per_period": float(slope),
        "r2": r2,
        "peak_period": label(y.idxmax()),
        "peak_value": float(vals.max()),
        "low_period": label(y.idxmin()),
        "low_value": float(vals.min()),
    }


def profile_dataset(df: pd.DataFrame, prep_notes: list[str] | None = None) -> dict:
    """Full automatic analysis. Returns a JSON-friendly dict."""
    n_rows, n_cols = df.shape
    kinds = {c: classify_column(df[c]) for c in df.columns}
    prof: dict[str, Any] = {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "kinds": kinds,
        "prep_notes": prep_notes or [],
    }

    # ---- columns ---------------------------------------------------------- #
    cols_info = []
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        miss = int(s.isna().sum())
        cols_info.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "kind": kinds[c],
                "non_null": int(len(nn)),
                "missing": miss,
                "missing_pct": miss / n_rows * 100 if n_rows else 0.0,
                "unique": int(s.nunique(dropna=True)),
                "example": str(nn.iloc[0])[:40] if len(nn) else "",
            }
        )
    prof["columns_info"] = cols_info

    # ---- missing ---------------------------------------------------------- #
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    prof["missing"] = {
        "total_cells": int(total_cells),
        "missing_cells": missing_cells,
        "missing_pct": missing_cells / total_cells * 100 if total_cells else 0.0,
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "columns_with_missing": [
            {"name": ci["name"], "missing": ci["missing"], "missing_pct": ci["missing_pct"]}
            for ci in sorted(cols_info, key=lambda d: -d["missing"])
            if ci["missing"] > 0
        ],
    }

    # ---- duplicates ------------------------------------------------------- #
    try:
        dup_mask = df.duplicated(keep="first")
    except TypeError:  # unhashable cell values
        dup_mask = df.astype(str).duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    prof["duplicates"] = {
        "count": dup_count,
        "pct": dup_count / n_rows * 100 if n_rows else 0.0,
        "row_indices": [int(i) for i in df.index[dup_mask][:200]],
    }

    # ---- numeric ---------------------------------------------------------- #
    num_cols = [c for c, k in kinds.items() if k == "numeric"]
    numeric_stats, outliers = {}, {}
    for c in num_cols:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        std = s.std()
        mean = s.mean()
        numeric_stats[c] = {
            "count": int(s.count()),
            "mean": float(mean),
            "std": None if np.isnan(std) else float(std),
            "min": float(s.min()),
            "q1": float(q1),
            "median": float(med),
            "q3": float(q3),
            "max": float(s.max()),
            "sum": float(s.sum()),
            "skew": float(s.skew()) if len(s) > 2 else None,
            "cv": float(std / abs(mean)) if mean and not np.isnan(std) else None,
        }
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            cnt = int(((s < lo) | (s > hi)).sum())
            if cnt:
                outliers[c] = {
                    "count": cnt,
                    "pct": cnt / len(s) * 100,
                    "lower_bound": float(lo),
                    "upper_bound": float(hi),
                }
    prof["numeric_stats"] = to_py(numeric_stats)
    prof["outliers"] = to_py(outliers)

    # ---- categorical ------------------------------------------------------ #
    cat_stats = {}
    for c, k in kinds.items():
        if k in ("categorical", "boolean"):
            vc = df[c].value_counts(dropna=True)
            total = int(vc.sum())
            cat_stats[c] = {
                "unique": int(len(vc)),
                "top": [
                    {"value": str(v), "count": int(n), "pct": n / total * 100 if total else 0.0}
                    for v, n in vc.head(5).items()
                ],
            }
    prof["categorical_stats"] = to_py(cat_stats)

    # ---- datetime --------------------------------------------------------- #
    dt_info = {}
    for c, k in kinds.items():
        if k == "datetime":
            s = df[c].dropna()
            if len(s):
                dt_info[c] = {
                    "min": s.min().isoformat(),
                    "max": s.max().isoformat(),
                    "span_days": float((s.max() - s.min()).total_seconds() / 86400),
                }
    prof["datetime_info"] = dt_info

    # ---- correlations ----------------------------------------------------- #
    corrs = []
    use = num_cols[:40]
    if len(use) >= 2:
        cm = df[use].astype(float).corr()
        for i, a in enumerate(use):
            for b in use[i + 1 :]:
                r = cm.loc[a, b]
                if pd.notna(r) and abs(r) >= 0.5:
                    corrs.append({"a": a, "b": b, "r": float(r)})
        corrs.sort(key=lambda d: -abs(d["r"]))
    prof["correlations"] = corrs[:10]

    # ---- trends ----------------------------------------------------------- #
    trends = []
    if dt_info and num_cols:
        dt_col = max(dt_info, key=lambda c: df[c].notna().sum())
        for c in num_cols[:8]:
            series, freq, agg = aggregate_series(df, dt_col, c)
            t = _trend_for(series)
            if t:
                t.update({"column": c, "date_column": dt_col, "freq": freq, "agg": agg})
                trends.append(t)
    prof["trends"] = to_py(trends)

    # ---- quality score ---------------------------------------------------- #
    notes, score = [], 100.0
    mp = prof["missing"]["missing_pct"]
    if mp > 0:
        score -= min(40.0, mp * 1.5)
        notes.append(f"{mp:.1f}% of all cells are missing.")
    dp = prof["duplicates"]["pct"]
    if dp > 0:
        score -= min(25.0, dp * 2)
        notes.append(f"{dup_count:,} duplicate row(s) ({dp:.1f}%).")
    if outliers:
        score -= min(15.0, len(outliers) * 2.5)
        notes.append(f"Outliers detected in {len(outliers)} numeric column(s).")
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        score -= min(10.0, len(const) * 3)
        notes.append("Constant (single-value) columns: " + ", ".join(const[:5]))
    heavy = [d["name"] for d in prof["missing"]["columns_with_missing"] if d["missing_pct"] > 50]
    if heavy:
        score -= min(10.0, len(heavy) * 3)
        notes.append(">50% missing in: " + ", ".join(heavy[:5]))
    prof["quality_score"] = max(0.0, round(score, 1))
    prof["quality_notes"] = notes
    return prof


# --------------------------------------------------------------------------- #
# text summary for the LLM
# --------------------------------------------------------------------------- #
def profile_to_text(profile: dict, df: pd.DataFrame | None = None, include_sample: bool = True) -> str:
    """Compact plain-text description of the dataset (this is what Claude sees)."""
    p = profile
    L = [f"DATASET: {p['rows']:,} rows x {p['columns']} columns. Quality score {p['quality_score']}/100."]
    if p.get("prep_notes"):
        L.append("PREP: " + " ".join(p["prep_notes"]))
    L.append("COLUMNS:")
    num, cat = p["numeric_stats"], p["categorical_stats"]
    for ci in p["columns_info"][:80]:
        c = ci["name"]
        line = f"- {c} [{ci['kind']}, {ci['dtype']}]: missing {ci['missing_pct']:.1f}%, unique {ci['unique']}"
        if c in num:
            s = num[c]
            line += (
                f", mean {fmt_num(s['mean'])}, median {fmt_num(s['median'])}, "
                f"min {fmt_num(s['min'])}, max {fmt_num(s['max'])}, sum {fmt_num(s['sum'])}"
            )
            if c in p["outliers"]:
                line += f", outliers {p['outliers'][c]['count']}"
        elif c in cat:
            line += ", top: " + "; ".join(f"{t['value']} ({t['pct']:.0f}%)" for t in cat[c]["top"][:4])
        elif c in p["datetime_info"]:
            d = p["datetime_info"][c]
            line += f", from {d['min'][:10]} to {d['max'][:10]}"
        L.append(line)
    if p["columns"] > 80:
        L.append(f"... and {p['columns'] - 80} more columns")
    L.append(
        f"MISSING: {p['missing']['missing_cells']:,} cells ({p['missing']['missing_pct']:.1f}%); "
        f"{p['missing']['rows_with_missing']:,} rows have at least one gap."
    )
    L.append(f"DUPLICATES: {p['duplicates']['count']:,} exact duplicate rows ({p['duplicates']['pct']:.1f}%).")
    if p["correlations"]:
        L.append("CORRELATIONS (|r|>=0.5): " + "; ".join(f"{c['a']} vs {c['b']}: {c['r']:+.2f}" for c in p["correlations"]))
    if p["trends"]:
        L.append("TRENDS:")
        for t in p["trends"]:
            ch = f"{t['change_pct']:+.1f}%" if t["change_pct"] is not None else "n/a"
            L.append(
                f"- {t['column']} ({t['agg']} per {t['freq']} by {t['date_column']}): {t['direction']}, "
                f"first->last {ch}, peak {t['peak_period']}, low {t['low_period']}"
            )
    if include_sample and df is not None and len(df):
        sample = df.head(5).copy()
        for c in sample.columns:
            sample[c] = sample[c].astype(str).str.slice(0, 40)
        L.append("SAMPLE ROWS (first 5):\n" + sample.to_csv(index=False))
    return "\n".join(L)
