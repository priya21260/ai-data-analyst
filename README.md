# 📊 AI Data Analyst

A standalone web app: upload a CSV or Excel file, get an automatic analysis, ask questions in plain English, and download a report. Claude is the AI brain; pandas does all the number-crunching.

## Features

| Feature | How it works |
|---|---|
| CSV / Excel upload | `.csv .tsv .txt .xlsx .xlsm .xls`, multi-sheet workbooks, auto-detects encoding and delimiter |
| Automatic analysis | Column types, text-to-number and date detection, statistics, correlations, outliers (IQR), trends |
| Missing values & duplicates | Detected, charted, listed, with one-click clean-up and cleaned-CSV download |
| Ask your data | Claude writes pandas code, the app runs it in a restricted sandbox, Claude explains the result (and can draw charts) |
| Interactive charts | Plotly: auto-generated set + a custom chart builder |
| Insights & recommendations | Claude turns the statistical profile into findings, issues and next steps |
| Downloadable reports | Self-contained HTML (print to PDF) and an Excel workbook |

## Project structure

```
ai-data-analyst/
├── app.py                  # Streamlit UI
├── analyst/
│   ├── analyzer.py         # loading, cleaning, profiling, trends (pandas)
│   ├── charts.py           # Plotly charts
│   ├── ai_engine.py        # Claude API: insights + Q&A tool-use loop
│   ├── sandbox.py          # restricted execution of Claude's pandas code
│   ├── report.py           # HTML + Excel report builders
│   └── sample.py           # demo dataset generator
├── tests/test_core.py      # unit tests
├── sample_data/sales_demo.csv
├── requirements.txt
├── Dockerfile
└── .streamlit/             # config.toml, secrets.toml.example
```

---

## Part 1 - Run it on your computer

### Step 1: Install the prerequisites
1. **Python 3.11 or newer** - download from python.org. On Windows, tick **"Add Python to PATH"** in the installer. Check with `python --version` (or `python3 --version`).
2. **A Claude API key** - sign in at console.anthropic.com, add a small amount of credit under Billing, then go to **API Keys → Create Key**. Copy it (starts with `sk-ant-`); you only see it once.
3. A code editor such as VS Code (optional but handy).

### Step 2: Open a terminal in the project folder
Unzip the project, then:
```bash
cd ai-data-analyst
```

### Step 3: Create a virtual environment and install packages
macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
(If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` first, or use Command Prompt and `.venv\Scripts\activate.bat`.)

### Step 4: Add your API key
```bash
mkdir -p .streamlit          # already exists in the zip; harmless
cp .streamlit/secrets.toml.example .streamlit/secrets.toml     # Windows: copy ... ...
```
Open `.streamlit/secrets.toml` and replace the placeholder with your real key:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
(Alternative: set an environment variable `ANTHROPIC_API_KEY`, or just paste the key into the app's sidebar each time.)

### Step 5: Start the app
```bash
streamlit run app.py
```
Your browser opens at http://localhost:8501.

### Step 6: Try it
1. Click **✨ Try the demo dataset** (or upload `sample_data/sales_demo.csv`, or your own file).
2. Browse **Overview → Data Quality → Statistics & Trends → Charts**. No API key needed for these.
3. Open **Ask Your Data** and try: *"Which region has the highest total revenue?"* or *"How has revenue changed over time? Show a chart."*
4. Open **Insights** and click **Generate insights**.
5. Open **Report**, click **Build report**, then download the HTML and Excel files.

### Step 7 (optional): Run the tests
```bash
python -m unittest discover -s tests -v
```

---

## Part 2 - Put it online with a public link

### Option A (recommended, free): Streamlit Community Cloud

**1. Put the project on GitHub**
1. Create a free account at github.com and click **New repository** (name it e.g. `ai-data-analyst`; public is simplest).
2. In your project folder:
   ```bash
   git init
   git add .
   git commit -m "AI Data Analyst"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ai-data-analyst.git
   git push -u origin main
   ```
   `.gitignore` already excludes `.streamlit/secrets.toml` and `.env`, so your key is **not** uploaded. Double-check on GitHub that no real key is visible.

**2. Deploy**
1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **Create app → "Yup, I have an app"**.
3. Choose your repository, branch `main`, and entrypoint file `app.py`.
4. Click **Advanced settings**:
   - **Python version**: pick 3.12 (the default at the time of writing).
   - **Secrets**: paste the contents of your `secrets.toml`, e.g.
     ```toml
     ANTHROPIC_API_KEY = "sk-ant-..."
     APP_PASSWORD = "a-strong-password"   # optional, see below
     ```
5. Click **Deploy**. After a few minutes you get a public `https://<name>.streamlit.app` link to share.

To update the app later, just `git push` - it redeploys automatically.

### Option B: Docker (Render, Fly.io, Railway, Google Cloud Run, your own server)
A `Dockerfile` is included. It listens on `$PORT` (default 8501).
```bash
docker build -t ai-data-analyst .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... ai-data-analyst
```
On a host like Render: **New → Web Service → connect your GitHub repo → Runtime: Docker**, then add `ANTHROPIC_API_KEY` under Environment. (The Dockerfile has not been tested on every host; follow your host's Docker deployment guide.)

---

## Part 3 - Before you share the link (important)

Anyone with the link can use the app, and **if you configure your own API key, their questions use your credits.** Pick one or combine:

1. **Spend limit** - in console.anthropic.com set a monthly spend limit / low credit balance for the key you deploy.
2. **Password** - set `APP_PASSWORD` in secrets; visitors must enter it.
3. **Per-session cap** - `MAX_AI_CALLS_PER_SESSION` (default 30) limits Claude requests per visitor when they are using your key.
4. **Bring-your-own-key** - leave `ANTHROPIC_API_KEY` out of the secrets entirely. Visitors paste their own key in the sidebar (it lives only in their browser session). Safest for a public link.
5. **Dedicated key** - create a separate key just for this app so you can revoke it any time.

Other notes:
- **Privacy**: files are held in memory for the visitor's session and not saved by the app. With "Send first 5 rows to Claude" on, five sample rows plus column statistics are sent to the Claude API; with it off, only statistics and the output of code results are. Don't use the public app for confidential data unless you're comfortable with that and with your hosting provider's terms.
- **Code execution**: Claude's pandas code runs in a restricted sandbox (no imports, file or network access, no `while` loops, allow-listed pandas/numpy functions, 20-second limit). This is defence in depth, not a perfect boundary - run it in a container (Option B) and keep nothing sensitive on the server.
- **Upload size** - `maxUploadSize` in `.streamlit/config.toml` (default 50 MB). Free hosting has limited memory, so very large files may be slow.
- **AI can be wrong** - answers come from code Claude ran on your data (shown under "How I got this"), but always sanity-check important numbers.

## Configuration reference

| Setting | Where | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | secrets / env var | none (visitors can paste their own) |
| `APP_PASSWORD` | secrets / env var | none (no password) |
| `MAX_AI_CALLS_PER_SESSION` | secrets / env var | 30 |
| `CLAUDE_MODEL` | env var | `claude-sonnet-5-5` |
| Model choices in sidebar | `analyst/ai_engine.py → MODEL_CHOICES` | Sonnet / Haiku / Opus |
| Max upload size | `.streamlit/config.toml` | 50 MB |

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment (Step 3) or run `python -m streamlit run app.py` |
| "API key was rejected" | Re-copy the key; make sure there are no spaces; check it's active in the Console |
| "Model not found" | Choose a different model in the sidebar, or update `MODEL_CHOICES` to current model names from docs.claude.com |
| Credit / billing error | Add credit in the Anthropic Console |
| `.xls` file fails | `pip install xlrd` (already in requirements) or save the file as `.xlsx` |
| Dates not detected | Make sure the column uses a consistent format, e.g. `2025-03-14` |
| Deployment fails on dependency install | Pick Python 3.12 under Advanced settings and check the build log |

## Ideas for extending it
Forecasting (statsmodels/Prophet), PDF export (e.g. with WeasyPrint), user accounts, saving analyses to a database, scheduled reports, connecting to SQL databases.
