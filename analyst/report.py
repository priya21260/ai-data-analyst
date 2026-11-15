"""Downloadable reports: a self-contained HTML report and an Excel workbook."""
from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd
from openpyxl.styles import Font

try:
    import markdown as _md
except ImportError:  # pragma: no cover
    _md = None

CSS = """
body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#111827;max-width:1100px;margin:0 auto;padding:32px 24px;line-height:1.55;background:#fff}
h1{font-size:30px;margin:0 0 4px}h2{font-size:21px;margin:36px 0 12px;padding-bottom:6px;border-bottom:2px solid #E5E7EB}
h3{font-size:16px;margin:22px 0 8px}.sub{color:#6B7280;margin-bottom:24px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}.card{flex:1 1 150px;background:#F5F3FF;border-radius:12px;padding:14px 16px}
.card .v{font-size:24px;font-weight:700;color:#4F46E5}.card .l{font-size:12px;color:#6B7280;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}th,td{border:1px solid #E5E7EB;padding:6px 10px;text-align:left}
th{background:#F9FAFB}tr:nth-child(even) td{background:#FCFCFD}.chart{margin:18px 0;page-break-inside:avoid}
.cap{color:#6B7280;font-size:13px}.note{background:#FFFBEB;border-left:4px solid #F59E0B;padding:8px 12px;margin:6px 0;border-radius:4px}
.qa{background:#F9FAFB;border-radius:10px;padding:12px 16px;margin:12px 0}.q{font-weight:600}
code,pre{background:#F3F4F6;border-radius:6px;font-size:12px}pre{padding:10px;overflow:auto}
footer{margin-top:48px;color:#9CA3AF;font-size:12px}
@media print{body{padding:0}}
"""


def _md_to_html(text: str) -> str:
    safe = html.escape(text or "", quote=False)
    if _md is not None:
        return _md.markdown(safe, extensions=["tables", "sane_lists"])
    return "<p>" + safe.replace("\n\n", "</p><p>").replace("\n", "<br>") + "</p>"


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    return df.head(max_rows).to_html(index=False, border=0, na_rep="", float_format=lambda v: f"{v:,.4g}", escape=True)


def _cards(items: list[tuple[str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div></div>'
        for l, v in items
    ) + "</div>"


def build_html_report(
    filename: str,
    df: pd.DataFrame,
    profile: dict,
    figures: list[tuple[str, object, str]] | None = None,
    insights_md: str | None = None,
    qa_log: list[dict] | None = None,
    embed_plotly: bool = False,
) -> str:
    """Return a single self-contained HTML document.

    figures: [(title, plotly_figure, caption)]. embed_plotly=True makes the file work fully offline
    (adds ~4 MB); otherwise Plotly.js is loaded from a CDN.
    """
    p = profile
    e = html.escape
    parts: list[str] = [
        f"<h1>Data Analysis Report</h1><div class='sub'>{e(filename)} &middot; generated "
        f"{datetime.now().strftime('%d %b %Y, %H:%M')}</div>",
        _cards([
            ("Rows", f"{p['rows']:,}"),
            ("Columns", str(p["columns"])),
            ("Missing cells", f"{p['missing']['missing_pct']:.1f}%"),
            ("Duplicate rows", f"{p['duplicates']['count']:,}"),
            ("Quality score", f"{p['quality_score']:.0f}/100"),
        ]),
    ]

    if insights_md:
        parts.append("<h2>AI Insights &amp; Recommendations</h2>" + _md_to_html(insights_md))

    parts.append("<h2>Data Quality</h2>")
    if p["quality_notes"]:
        parts += [f"<div class='note'>{e(n)}</div>" for n in p["quality_notes"]]
    else:
        parts.append("<p>No data quality issues detected.</p>")
    if p["missing"]["columns_with_missing"]:
        parts.append("<h3>Missing values</h3>" + _table(pd.DataFrame(p["missing"]["columns_with_missing"]).rename(
            columns={"name": "Column", "missing": "Missing", "missing_pct": "% missing"})))
    if p["outliers"]:
        o = pd.DataFrame([{"Column": k, "Outliers": v["count"], "% of values": v["pct"],
                           "Lower bound": v["lower_bound"], "Upper bound": v["upper_bound"]}
                          for k, v in p["outliers"].items()])
        parts.append("<h3>Outliers (IQR rule)</h3>" + _table(o))

    parts.append("<h2>Columns</h2>" + _table(pd.DataFrame(p["columns_info"]).rename(columns={
        "name": "Column", "dtype": "Type", "kind": "Role", "non_null": "Non-null", "missing": "Missing",
        "missing_pct": "% missing", "unique": "Unique", "example": "Example"})))

    if p["numeric_stats"]:
        s = pd.DataFrame(p["numeric_stats"]).T.reset_index().rename(columns={"index": "Column"})
        parts.append("<h2>Numeric Statistics</h2>" + _table(s))
    if p["correlations"]:
        c = pd.DataFrame(p["correlations"]).rename(columns={"a": "Column A", "b": "Column B", "r": "Correlation (r)"})
        parts.append("<h2>Strong Correlations</h2>" + _table(c))
    if p["trends"]:
        t = pd.DataFrame(p["trends"])[["column", "direction", "change_pct", "agg", "freq", "peak_period", "low_period"]]
        parts.append("<h2>Trends</h2>" + _table(t.rename(columns={
            "column": "Column", "direction": "Direction", "change_pct": "First→last change %",
            "agg": "Aggregation", "freq": "Period", "peak_period": "Peak", "low_period": "Low"})))

    if figures:
        parts.append("<h2>Charts</h2>")
        for i, (title, fig, cap) in enumerate(figures):
            include = (True if embed_plotly else "cdn") if i == 0 else False
            div = fig.to_html(full_html=False, include_plotlyjs=include, config={"displaylogo": False})
            parts.append(f"<div class='chart'><h3>{e(title)}</h3>{div}<div class='cap'>{e(cap or '')}</div></div>")

    if qa_log:
        parts.append("<h2>Questions &amp; Answers</h2>")
        pairs = []
        for m in qa_log:
            if m["role"] == "user":
                pairs.append([m["content"], ""])
            elif pairs:
                pairs[-1][1] = m["content"]
        for q, a in pairs:
            parts.append(f"<div class='qa'><div class='q'>{e(q)}</div>{_md_to_html(a)}</div>")

    parts.append("<h2>Data Preview</h2>" + _table(df.head(20)))
    parts.append("<footer>Generated by AI Data Analyst. AI-written text can contain mistakes - "
                 "verify important numbers against your source data.</footer>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Analysis report - {e(filename)}</title><style>{CSS}</style></head><body>"
            + "\n".join(parts) + "</body></html>")


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_localize(None)
    return df


def build_excel_report(df: pd.DataFrame, profile: dict, insights_md: str | None = None,
                       max_data_rows: int = 200_000) -> bytes:
    p = profile
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary = pd.DataFrame(
            [
                ("Rows", p["rows"]), ("Columns", p["columns"]),
                ("Missing cells", p["missing"]["missing_cells"]),
                ("Missing %", round(p["missing"]["missing_pct"], 2)),
                ("Rows with any missing value", p["missing"]["rows_with_missing"]),
                ("Duplicate rows", p["duplicates"]["count"]),
                ("Quality score (0-100)", p["quality_score"]),
            ] + [("Note", n) for n in p["quality_notes"]],
            columns=["Metric", "Value"],
        )
        summary.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(p["columns_info"]).to_excel(xw, sheet_name="Columns", index=False)
        if p["missing"]["columns_with_missing"]:
            pd.DataFrame(p["missing"]["columns_with_missing"]).to_excel(xw, sheet_name="Missing", index=False)
        if p["duplicates"]["count"]:
            try:
                dups = df[df.duplicated(keep=False)]
            except TypeError:
                dups = df[df.astype(str).duplicated(keep=False)]
            _excel_safe(dups.head(10_000)).to_excel(xw, sheet_name="Duplicates", index=False)
        if p["numeric_stats"]:
            pd.DataFrame(p["numeric_stats"]).T.reset_index().rename(columns={"index": "column"}).to_excel(
                xw, sheet_name="Numeric stats", index=False)
        if p["outliers"]:
            pd.DataFrame(p["outliers"]).T.reset_index().rename(columns={"index": "column"}).to_excel(
                xw, sheet_name="Outliers", index=False)
        if p["correlations"]:
            pd.DataFrame(p["correlations"]).to_excel(xw, sheet_name="Correlations", index=False)
        if p["trends"]:
            pd.DataFrame(p["trends"]).to_excel(xw, sheet_name="Trends", index=False)
        if insights_md:
            pd.DataFrame({"AI insights": insights_md.splitlines()}).to_excel(xw, sheet_name="Insights", index=False)
        _excel_safe(df.head(max_data_rows)).to_excel(xw, sheet_name="Data", index=False)

        for ws in xw.book.worksheets:  # light formatting
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))
    return buf.getvalue()
