"""AI Data Analyst - Streamlit web app.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import hashlib
import hmac
import os

import pandas as pd
import streamlit as st

from analyst import ai_engine, analyzer, charts, report
from analyst.sample import make_sample_data

MAX_AI_CALLS = int(os.getenv("MAX_AI_CALLS_PER_SESSION", "30"))  # applies only to the server's own API key
MAX_ROWS_WARN = 500_000

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.block-container{padding-top:1.6rem;max-width:1250px}
.hero{background:linear-gradient(135deg,#4F46E5 0%,#7C3AED 60%,#06B6D4 100%);color:#fff;border-radius:18px;padding:34px 38px;margin-bottom:22px}
.hero h1{color:#fff;margin:0 0 6px;font-size:2.1rem}.hero p{margin:0;opacity:.92;font-size:1.05rem}
.feat{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:16px 18px;height:100%}
.feat b{font-size:1.02rem}.small{opacity:.7;font-size:.85rem}
div[data-testid="stMetric"]{background:rgba(79,70,229,.07);border-radius:12px;padding:12px 16px}
.trend{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:12px 14px;margin-bottom:10px}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# access + API key
# --------------------------------------------------------------------------- #
def _secret(name: str) -> str | None:
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001 - no secrets file / key missing
        return None


def password_gate() -> None:
    required = _secret("APP_PASSWORD")
    if not required or st.session_state.get("authed"):
        return
    st.markdown("<div class='hero'><h1>📊 AI Data Analyst</h1><p>This app is password protected.</p></div>", unsafe_allow_html=True)
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), required.encode()):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


password_gate()

S = st.session_state
for k, v in {"df": None, "profile": None, "insights": None, "chat": [], "pinned": [], "ai_calls": 0,
             "file_key": None, "report_html": None, "report_xlsx": None, "auto": None, "version": 0,
             "pending_q": None}.items():
    S.setdefault(k, v)


def set_data(df: pd.DataFrame, notes: list[str], filename: str, reset_chat: bool = True) -> None:
    S.df, S.filename = df, filename
    with st.spinner("Analysing your data…"):
        S.profile = analyzer.profile_dataset(df, notes)
    S.insights, S.auto, S.report_html, S.report_xlsx = None, None, None, None
    S.version += 1
    if reset_chat:
        S.chat, S.pinned = [], []


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## 📊 AI Data Analyst")
    st.caption("Upload data → get instant analysis → ask questions in plain English.")

    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=[e.lstrip(".") for e in analyzer.SUPPORTED_EXTS],
        help="Supported: .csv, .tsv, .txt, .xlsx, .xlsm, .xls. Max size is set in .streamlit/config.toml (default 50 MB).",
    )
    if st.button("✨ Try the demo dataset"):
        df0, notes0 = analyzer.prepare_dataframe(make_sample_data())
        S.file_key = "demo"
        S.sheets = None
        set_data(df0, notes0, "demo_sales.csv")
        st.rerun()

    if uploaded is not None:
        data = uploaded.getvalue()
        key = (uploaded.name, hashlib.md5(data).hexdigest())
        if S.file_key != key:
            try:
                S.sheets = analyzer.read_table(data, uploaded.name)
                S.file_key = key
                S.sheet_loaded = None
            except ValueError as e:
                S.file_key = None
                st.error(str(e))
        if S.file_key == key and S.get("sheets"):
            names = list(S.sheets)
            sheet = st.selectbox("Sheet", names) if len(names) > 1 else names[0]
            if S.get("sheet_loaded") != (key, sheet):
                df0, notes0 = analyzer.prepare_dataframe(S.sheets[sheet])
                S.sheet_loaded = (key, sheet)
                set_data(df0, notes0, uploaded.name if len(names) == 1 else f"{uploaded.name} [{sheet}]")

    st.divider()
    provider = st.radio("AI provider", ai_engine.PROVIDERS,
                        index=ai_engine.PROVIDERS.index(ai_engine.DEFAULT_PROVIDER) if ai_engine.DEFAULT_PROVIDER in ai_engine.PROVIDERS else 0,
                        help="Gemini has a free tier (no billing needed). Claude is paid.")
    is_gemini = provider.startswith("Gemini")
    server_key = _secret("GEMINI_API_KEY" if is_gemini else "ANTHROPIC_API_KEY")
    user_key = st.text_input(
        ("Your Gemini API key" if is_gemini else "Your Claude API key") + (" (optional)" if server_key else ""),
        type="password",
        placeholder="AIza…" if is_gemini else "sk-ant-…",
        help=("Free key: aistudio.google.com/apikey" if is_gemini else "Get one at console.anthropic.com")
        + ". Your key is kept only in this browser session.",
    )
    api_key = user_key.strip() or server_key
    using_server_key = bool(api_key) and not user_key.strip()
    choices = ai_engine.GEMINI_MODEL_CHOICES if is_gemini else ai_engine.MODEL_CHOICES
    default_model = ai_engine.GEMINI_DEFAULT_MODEL if is_gemini else ai_engine.DEFAULT_MODEL
    model = st.selectbox(
        "AI model", choices, index=choices.index(default_model) if default_model in choices else 0,
        help="Flash-Lite is lightest on the free quota; Flash is better at tricky questions." if is_gemini
        else "Sonnet is the best balance. Haiku is fastest/cheapest. Opus is strongest on hard questions.",
    )
    send_sample = st.checkbox("Send first 5 rows to the AI as context", value=True,
                              help="Improves answers. Turn off for sensitive data (free-tier prompts may be used by Google to improve its products) - the AI then only sees column "
                                   "statistics and the results of the code it runs.")
    if not api_key:
        st.info("Add an API key to enable AI insights and Q&A. Everything else works without one.")
    st.caption("🔒 Files are processed in memory for your session and are not stored.")


def ai_allowed() -> bool:
    if not api_key:
        st.warning("Enter your API key in the sidebar to use this feature.")
        return False
    if using_server_key and S.ai_calls >= MAX_AI_CALLS:
        st.warning(f"This session reached the {MAX_AI_CALLS}-request limit for the shared key. "
                   "Add your own API key in the sidebar to continue.")
        return False
    return True


# --------------------------------------------------------------------------- #
# landing page
# --------------------------------------------------------------------------- #
if S.df is None:
    st.markdown(
        "<div class='hero'><h1>Your AI Data Analyst</h1>"
        "<p>Drop in a spreadsheet and get statistics, charts, data-quality checks, written insights "
        "and answers to your questions - in seconds.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    feats = [
        ("🔍 Auto-analysis", "Types, statistics, trends, correlations and outliers detected automatically."),
        ("🧹 Data quality", "Missing values and duplicate records found, with one-click cleanup."),
        ("💬 Ask anything", "Questions in plain English; the AI writes and runs the pandas code."),
        ("📄 Reports", "Download an HTML report or Excel workbook to share."),
    ]
    for col, (t, d) in zip(cols, feats):
        col.markdown(f"<div class='feat'><b>{t}</b><br><span class='small'>{d}</span></div>", unsafe_allow_html=True)
    st.markdown("")
    st.info("👈 Upload a CSV or Excel file in the sidebar, or click **Try the demo dataset** to explore.")
    st.stop()

df: pd.DataFrame = S.df
prof: dict = S.profile

# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
st.markdown(f"### 📁 {S.filename}")
if prof["rows"] > MAX_ROWS_WARN:
    st.warning(f"Large dataset ({prof['rows']:,} rows). Some views may be slow.")
for n in prof["prep_notes"]:
    st.caption(f"ℹ️ {n}")

m = st.columns(5)
m[0].metric("Rows", f"{prof['rows']:,}")
m[1].metric("Columns", prof["columns"])
m[2].metric("Missing cells", f"{prof['missing']['missing_pct']:.1f}%")
m[3].metric("Duplicate rows", f"{prof['duplicates']['count']:,}")
m[4].metric("Quality score", f"{prof['quality_score']:.0f}/100")

tab_over, tab_quality, tab_stats, tab_charts, tab_ask, tab_insights, tab_report = st.tabs(
    ["📋 Overview", "🧹 Data Quality", "📈 Statistics & Trends", "📊 Charts", "💬 Ask Your Data", "💡 Insights", "📄 Report"]
)

# --------------------------------------------------------------------------- #
# overview
# --------------------------------------------------------------------------- #
with tab_over:
    st.subheader("Data preview")
    st.dataframe(df.head(200), hide_index=True)
    st.subheader("Columns")
    ci = pd.DataFrame(prof["columns_info"]).rename(columns={
        "name": "Column", "dtype": "Type", "kind": "Role", "non_null": "Non-null", "missing": "Missing",
        "missing_pct": "% missing", "unique": "Unique", "example": "Example"})
    st.dataframe(ci, hide_index=True, column_config={"% missing": st.column_config.ProgressColumn(
        "% missing", min_value=0, max_value=100, format="%.1f%%")})

# --------------------------------------------------------------------------- #
# data quality
# --------------------------------------------------------------------------- #
with tab_quality:
    for n in prof["quality_notes"]:
        st.warning(n)
    if not prof["quality_notes"]:
        st.success("No data quality issues detected. 🎉")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing values")
        fig = charts.missing_values_chart(prof)
        if fig is None:
            st.success("No missing values.")
        else:
            st.plotly_chart(fig, key=f"miss_{S.version}")
            with st.expander("Details"):
                st.dataframe(pd.DataFrame(prof["missing"]["columns_with_missing"]), hide_index=True)
    with c2:
        st.subheader("Duplicate records")
        d = prof["duplicates"]
        if d["count"] == 0:
            st.success("No duplicate rows.")
        else:
            st.error(f"{d['count']:,} exact duplicate rows ({d['pct']:.1f}%).")
            dup_df = df[df.duplicated(keep=False)] if d["count"] else df.head(0)
            with st.expander("Show duplicated rows"):
                st.dataframe(dup_df.head(500))
        if prof["outliers"]:
            st.subheader("Outliers (IQR rule)")
            st.dataframe(pd.DataFrame([{"Column": k, "Outliers": v["count"], "% of values": round(v["pct"], 2),
                                        "Lower bound": v["lower_bound"], "Upper bound": v["upper_bound"]}
                                       for k, v in prof["outliers"].items()]), hide_index=True)

    st.subheader("Clean up")
    st.caption("Actions change the working dataset for the rest of the session (your original file is untouched). "
               "Insights and reports are refreshed from the cleaned data.")
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Remove duplicate rows", disabled=prof["duplicates"]["count"] == 0):
        set_data(df.drop_duplicates().reset_index(drop=True), prof["prep_notes"] + ["Removed duplicate rows."],
                 S.filename, reset_chat=False)
        st.rerun()
    if a2.button("Drop rows with any missing value", disabled=prof["missing"]["missing_cells"] == 0):
        set_data(df.dropna().reset_index(drop=True), prof["prep_notes"] + ["Dropped rows with missing values."],
                 S.filename, reset_chat=False)
        st.rerun()
    if a3.button("Fill gaps (median / most common)", disabled=prof["missing"]["missing_cells"] == 0):
        filled = df.copy()
        for c, k in prof["kinds"].items():
            if filled[c].isna().any():
                if k == "numeric":
                    filled[c] = filled[c].fillna(filled[c].median())
                elif k != "datetime" and not filled[c].mode().empty:
                    filled[c] = filled[c].fillna(filled[c].mode().iloc[0])
        set_data(filled, prof["prep_notes"] + ["Filled missing values (median for numbers, mode for text)."],
                 S.filename, reset_chat=False)
        st.rerun()
    if a4.button("Reset to original upload"):
        if S.file_key == "demo":
            d0, n0 = analyzer.prepare_dataframe(make_sample_data())
        else:
            d0, n0 = analyzer.prepare_dataframe(S.sheets[S.sheet_loaded[1]])
        set_data(d0, n0, S.filename, reset_chat=False)
        st.rerun()
    st.download_button("⬇️ Download current data as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_data.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# statistics & trends
# --------------------------------------------------------------------------- #
with tab_stats:
    st.subheader("Numeric statistics")
    if prof["numeric_stats"]:
        ns = pd.DataFrame(prof["numeric_stats"]).T
        st.dataframe(ns[["count", "mean", "std", "min", "q1", "median", "q3", "max", "sum", "skew"]])
    else:
        st.info("No numeric columns found.")

    st.subheader("Trends over time")
    if prof["trends"]:
        cols = st.columns(2)
        for i, t in enumerate(prof["trends"]):
            icon = "📈" if "increasing" in t["direction"] else "📉" if "decreasing" in t["direction"] else "➖"
            chg = f"{t['change_pct']:+.1f}%" if t["change_pct"] is not None else "n/a"
            cols[i % 2].markdown(
                f"<div class='trend'><b>{icon} {t['column']}</b> - {t['direction']}<br>"
                f"<span class='small'>{t['agg']} per {t['freq']} by <i>{t['date_column']}</i> · first→last {chg} · "
                f"peak {t['peak_period']} · low {t['low_period']} · R² {t['r2']:.2f}</span></div>",
                unsafe_allow_html=True)
    else:
        st.info("Trends need a date column and at least one numeric column.")

    st.subheader("Strong correlations")
    if prof["correlations"]:
        st.dataframe(pd.DataFrame(prof["correlations"]).rename(columns={"a": "Column A", "b": "Column B", "r": "r"}),
                     hide_index=True)
    else:
        st.info("No strong correlations (|r| ≥ 0.5) found between numeric columns.")

    if prof["categorical_stats"]:
        st.subheader("Categorical columns")
        for c, v in prof["categorical_stats"].items():
            with st.expander(f"{c} - {v['unique']} unique values"):
                st.dataframe(pd.DataFrame(v["top"]).rename(columns={"value": "Value", "count": "Count", "pct": "%"}),
                             hide_index=True)

# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
with tab_charts:
    st.subheader("Automatic charts")
    if S.auto is None or S.auto[0] != S.version:
        with st.spinner("Drawing charts…"):
            S.auto = (S.version, charts.auto_charts(df, prof))
    auto = S.auto[1]
    if not auto:
        st.info("Not enough suitable columns for automatic charts.")
    for i in range(0, len(auto), 2):
        cols = st.columns(2)
        for col, (title, fig, cap) in zip(cols, auto[i:i + 2]):
            with col:
                st.plotly_chart(fig, key=f"auto_{S.version}_{i}_{title}")
                st.caption(cap)

    st.divider()
    st.subheader("Build your own chart")
    all_cols = list(df.columns)
    b1, b2, b3, b4, b5 = st.columns(5)
    kind = b1.selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie"])
    x = b2.selectbox("X axis", all_cols)
    y = b3.selectbox("Y axis (optional)", ["(none)"] + all_cols)
    color = b4.selectbox("Colour by (optional)", ["(none)"] + all_cols)
    agg = b5.selectbox("Aggregate", ["sum", "mean", "median", "count", "min", "max"])
    try:
        custom = charts.build_custom_chart(df, kind, x, None if y == "(none)" else y,
                                           None if color == "(none)" else color, agg)
        st.plotly_chart(custom, key=f"custom_{S.version}")
        if st.button("📌 Add this chart to the report"):
            S.pinned.append((f"{kind}: {x}" + (f" / {y}" if y != "(none)" else ""), custom, "Custom chart"))
            st.toast("Pinned to report")
    except Exception as e:  # noqa: BLE001
        st.info(f"Can't draw that combination: {e}")

# --------------------------------------------------------------------------- #
# ask your data
# --------------------------------------------------------------------------- #
with tab_ask:
    st.subheader("Ask questions about your data")
    st.caption("The AI writes pandas code, runs it on your data and explains the result. Expand “How I got this” to check the work.")

    if not S.chat:
        st.markdown("**Try one of these:**")
        for q in ai_engine.default_questions(prof):
            if st.button(q, key=f"sugg_{hash(q)}"):
                S.pending_q = q
                st.rerun()

    for i, msg in enumerate(S.chat):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for j, f in enumerate(msg.get("figs", [])):
                st.plotly_chart(f, key=f"chat_{S.version}_{i}_{j}")
            if msg.get("steps"):
                with st.expander("How I got this (code)"):
                    for st_ in msg["steps"]:
                        st.code(st_["code"], language="python")
                        (st.error if st_["error"] else st.text)(st_["output"][:1500])

    typed = st.chat_input("e.g. Which region had the highest average order value?")
    question = typed or S.pending_q
    S.pending_q = None
    if question and ai_allowed():
        S.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                try:
                    client = ai_engine.make_client(api_key, provider)
                    ptxt = analyzer.profile_to_text(prof, df, include_sample=send_sample)
                    hist = [{"role": m_["role"], "content": m_["content"]} for m_ in S.chat[:-1]]
                    ans = ai_engine.answer_question(client, model, df, ptxt, hist, question)
                    S.ai_calls += 1
                    S.chat.append({"role": "assistant", "content": ans.text, "figs": ans.figures, "steps": ans.code_steps})
                except ai_engine.AIError as e:
                    S.chat.pop()
                    st.error(str(e))
        if S.chat and S.chat[-1]["role"] == "assistant":
            st.rerun()
    if S.chat and st.button("Clear conversation"):
        S.chat = []
        st.rerun()

# --------------------------------------------------------------------------- #
# insights
# --------------------------------------------------------------------------- #
with tab_insights:
    st.subheader("AI insights & recommendations")
    st.caption("The AI reads the statistical profile of your dataset and writes findings and next steps.")
    if st.button("✨ Generate insights", type="primary"):
        if ai_allowed():
            with st.spinner("The AI is analysing your dataset…"):
                try:
                    client = ai_engine.make_client(api_key, provider)
                    S.insights = ai_engine.generate_insights(
                        client, model, analyzer.profile_to_text(prof, df, include_sample=send_sample))
                    S.ai_calls += 1
                    S.report_html = None
                except ai_engine.AIError as e:
                    st.error(str(e))
    if S.insights:
        st.markdown(S.insights)
        st.caption("AI-generated. Verify important numbers before acting on them.")
    else:
        st.info("Click the button to generate an executive summary, key findings and recommendations.")

# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
with tab_report:
    st.subheader("Download your analysis")
    r1, r2, r3 = st.columns(3)
    inc_charts = r1.checkbox("Include automatic charts", value=True)
    inc_qa = r2.checkbox("Include Q&A", value=bool(S.chat), disabled=not S.chat)
    offline = r3.checkbox("Make HTML fully offline (+4 MB)", value=False)
    st.caption(f"Insights: {'included' if S.insights else 'not generated yet (see the Insights tab)'} · "
               f"Pinned custom charts: {len(S.pinned)}")

    if st.button("🛠️ Build report", type="primary"):
        with st.spinner("Building report…"):
            figs = []
            if inc_charts:
                if S.auto is None or S.auto[0] != S.version:
                    S.auto = (S.version, charts.auto_charts(df, prof))
                figs += S.auto[1]
            figs += S.pinned
            S.report_html = report.build_html_report(
                S.filename, df, prof, figs, S.insights, S.chat if inc_qa else None, embed_plotly=offline
            ).encode("utf-8")
            S.report_xlsx = report.build_excel_report(df, prof, S.insights)
    if S.report_html:
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ HTML report (open in any browser, print to PDF)", S.report_html,
                           file_name="analysis_report.html", mime="text/html")
        d2.download_button("⬇️ Excel workbook", S.report_xlsx, file_name="analysis_report.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("Report ready. Tip: open the HTML file and use your browser's Print → Save as PDF for a PDF copy.")
