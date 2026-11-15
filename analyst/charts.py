"""Interactive Plotly charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .analyzer import aggregate_series, rank_numeric

TEMPLATE = "plotly_white"
PALETTE = ["#4F46E5", "#06B6D4", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899", "#64748B"]
MAX_POINTS = 5000


def _style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        colorway=PALETTE,
        font=dict(family="Inter, system-ui, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def missing_values_chart(profile: dict) -> go.Figure | None:
    rows = profile["missing"]["columns_with_missing"]
    if not rows:
        return None
    d = pd.DataFrame(rows).sort_values("missing_pct")
    fig = px.bar(
        d, x="missing_pct", y="name", orientation="h",
        labels={"missing_pct": "% missing", "name": ""},
        hover_data={"missing": True, "missing_pct": ":.1f"},
        color_discrete_sequence=[PALETTE[4]],
    )
    return _style(fig, "Missing values by column", height=max(260, 28 * len(d) + 100))


def distribution_chart(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, marginal="box", color_discrete_sequence=[PALETTE[0]])
    return _style(fig, f"Distribution of {col}")


def category_chart(df: pd.DataFrame, col: str, top: int = 15) -> go.Figure:
    vc = df[col].value_counts().head(top).reset_index()
    vc.columns = [col, "count"]
    vc[col] = vc[col].astype(str)
    fig = px.bar(vc.iloc[::-1], x="count", y=col, orientation="h", color_discrete_sequence=[PALETTE[1]])
    return _style(fig, f"Top values: {col}", height=max(280, 26 * len(vc) + 100))


def correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> go.Figure | None:
    cols = cols[:25]
    if len(cols) < 2:
        return None
    cm = df[cols].astype(float).corr().round(2)
    fig = px.imshow(cm, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    return _style(fig, "Correlation matrix", height=max(380, 34 * len(cols) + 120))


def timeseries_chart(df: pd.DataFrame, dt_col: str, val_col: str, freq: str | None = None,
                     agg: str | None = None) -> go.Figure | None:
    s, freq, agg = aggregate_series(df, dt_col, val_col, freq, agg)
    s = s.dropna()
    if len(s) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=f"{agg} of {val_col}",
                             line=dict(color=PALETTE[0], width=2)))
    if len(s) >= 4:
        x = np.arange(len(s))
        slope, intercept = np.polyfit(x, s.values.astype(float), 1)
        fig.add_trace(go.Scatter(x=s.index, y=slope * x + intercept, mode="lines", name="Trend",
                                 line=dict(color=PALETTE[4], dash="dash")))
    fig.update_yaxes(title=f"{agg} of {val_col}")
    return _style(fig, f"{val_col} over time ({agg}, by {freq})")


def scatter_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> go.Figure:
    d = df[[c for c in {x, y, color} if c]].dropna()
    if len(d) > MAX_POINTS:
        d = d.sample(MAX_POINTS, random_state=0)
    fig = px.scatter(d, x=x, y=y, color=color, opacity=0.65, color_discrete_sequence=PALETTE)
    if pd.api.types.is_numeric_dtype(d[x]) and pd.api.types.is_numeric_dtype(d[y]) and len(d) > 2:
        slope, intercept = np.polyfit(d[x].astype(float), d[y].astype(float), 1)
        xs = np.array([d[x].min(), d[x].max()], dtype=float)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear fit",
                                 line=dict(color="#111827", dash="dash")))
    return _style(fig, f"{y} vs {x}")


def box_chart(df: pd.DataFrame, num_col: str, cat_col: str | None = None) -> go.Figure:
    d = df
    if cat_col:
        top = df[cat_col].value_counts().head(15).index
        d = df[df[cat_col].isin(top)]
    fig = px.box(d, y=num_col, x=cat_col, color=cat_col, color_discrete_sequence=PALETTE)
    return _style(fig, f"{num_col}" + (f" by {cat_col}" if cat_col else ""))


def build_custom_chart(df: pd.DataFrame, kind: str, x: str | None, y: str | None = None,
                       color: str | None = None, agg: str = "sum") -> go.Figure:
    """Chart-builder used by the UI. kind: Bar | Line | Scatter | Histogram | Box | Pie."""
    if not x:
        raise ValueError("Choose an X column.")
    d = df
    if kind in ("Bar", "Line", "Pie"):
        keys = [x] + ([color] if color and color != x else [])
        if y and agg != "count":
            g = d.groupby(keys, dropna=True, observed=True)[y].agg(agg).reset_index()
            yname = f"{agg} of {y}"
        else:
            g = d.groupby(keys, dropna=True, observed=True).size().reset_index(name="count")
            y, yname = "count", "count"
        if kind == "Line" or pd.api.types.is_datetime64_any_dtype(g[x]):
            g = g.sort_values(x)
        else:
            order = g.groupby(x)[y].sum().sort_values(ascending=False).head(30).index
            g = g[g[x].isin(order)]
            g[x] = g[x].astype(str)
        if kind == "Bar":
            fig = px.bar(g, x=x, y=y, color=color if color != x else None, barmode="group",
                         color_discrete_sequence=PALETTE, labels={y: yname})
        elif kind == "Line":
            fig = px.line(g, x=x, y=y, color=color if color != x else None, markers=True,
                          color_discrete_sequence=PALETTE, labels={y: yname})
        else:
            fig = px.pie(g, names=x, values=y, color_discrete_sequence=PALETTE, hole=0.4)
        return _style(fig, f"{yname} by {x}")
    if kind == "Scatter":
        if not y:
            raise ValueError("Scatter needs a Y column.")
        return scatter_chart(d, x, y, color)
    if kind == "Histogram":
        fig = px.histogram(d, x=x, color=color, nbins=40, color_discrete_sequence=PALETTE, barmode="overlay",
                           opacity=0.8)
        return _style(fig, f"Distribution of {x}")
    if kind == "Box":
        if not y:
            raise ValueError("Box plot needs a numeric Y column.")
        return box_chart(d, y, x)
    raise ValueError(f"Unknown chart type: {kind}")


def auto_charts(df: pd.DataFrame, profile: dict, max_charts: int = 12) -> list[tuple[str, go.Figure, str]]:
    """Pick a sensible set of charts automatically. Returns [(title, figure, caption)]."""
    kinds = profile["kinds"]
    out: list[tuple[str, go.Figure, str]] = []

    num = rank_numeric([c for c, k in kinds.items() if k == "numeric"])
    cat = [c for c, k in kinds.items() if k == "categorical"]
    dts = [c for c, k in kinds.items() if k == "datetime"]

    fig = missing_values_chart(profile)
    if fig is not None:
        out.append(("Missing values", fig, "Columns with gaps, as a percentage of rows."))

    if dts and num:
        dt = max(dts, key=lambda c: df[c].notna().sum())
        # prefer columns that have a real trend
        trend_cols = [t["column"] for t in profile["trends"] if "increasing" in t["direction"]
                      or "decreasing" in t["direction"]]
        for c in (trend_cols + [n for n in num if n not in trend_cols])[:3]:
            f = timeseries_chart(df, dt, c)
            if f is not None:
                out.append((f"{c} over time", f, "Aggregated over time with a fitted linear trend."))

    for c in num[:4]:
        out.append((f"Distribution of {c}", distribution_chart(df, c), "Histogram with box plot to show spread and outliers."))

    for c in cat[:3]:
        if df[c].nunique() >= 2:
            out.append((f"Top values: {c}", category_chart(df, c), "Most frequent categories."))

    if len(num) >= 2:
        hm = correlation_heatmap(df, num)
        if hm is not None:
            out.append(("Correlation matrix", hm, "Pearson correlation between numeric columns."))
        if profile["correlations"]:
            top = profile["correlations"][0]
            out.append((f"{top['a']} vs {top['b']}", scatter_chart(df, top["a"], top["b"]),
                        f"Strongest relationship found (r = {top['r']:+.2f})."))

    if num and cat:
        c0 = min(cat, key=lambda c: df[c].nunique() if df[c].nunique() >= 2 else 10**9)
        if df[c0].nunique() >= 2:
            out.append((f"{num[0]} by {c0}", box_chart(df, num[0], c0), "Spread of values per category."))
    return out[:max_charts]
