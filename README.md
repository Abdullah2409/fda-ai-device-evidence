# Evidence at Clearance and Safety After

**Do FDA-authorized AI medical devices cleared without reported clinical validation get recalled
more often, or generate more adverse event reports, than devices backed by clinical testing?**

Final project for CMU 90-819 Python Programming II (Fall 2026), Abdullah Arshad.
**Read the report online: <https://fda-ai-device-evidence-report.vercel.app>** (also as
[`report/report.pdf`](report/report.pdf) and [`report/report.html`](report/report.html)).
The analysis follows the submitted [analysis plan](docs/Analysis_Plan_Abdullah_Arshad.pdf).

## Findings

Across 1,521 AI-enabled devices on the FDA's list (103 later recalled):

- **Devices without reported clinical validation had about 4 times the odds of recall** after
  adjusting for risk class, specialty, public company status, review time and years on market
  (validated vs. not: odds ratio 0.24, 95% CI 0.15–0.38). A Cox model of time to first recall
  agrees (hazard ratio 0.28).
- **Adverse event reports** were also lower for validated devices, but the adjusted difference was
  not statistically significant (rate ratio 0.56, 95% CI 0.26–1.18).
- **Public company devices were recalled far more often** (odds ratio 6.2), matching the
  *JAMA Health Forum* (2025) estimate of 5.9.
- The share of summaries reporting clinical validation rose from 38% (2017) to 78% (2024–2026).

These are associations, not causal effects; see the report's limitations.

## Reproduce

Requires Python 3.12+ and [Quarto](https://quarto.org) (the PDF uses Quarto's built-in Typst, so no LaTeX).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name fda-ai-device-evidence
.venv/bin/python run_all.py              # rebuild everything from committed data (~30 s)
.venv/bin/python run_all.py --download   # re-download all sources first (counts may change)
```

`run_all.py` rebuilds the database, analysis table, models, figures and report. It does not re-run
the LLM coding of clinical validation; those results are committed and merged.

Without an openFDA API key, the downloads use about 50 requests, well under the 1,000/day limit. To
re-download the SEC file, either save <https://www.sec.gov/files/company_tickers.json> to
`data/raw/` in a browser, or set `SEC_USER_AGENT="Your Name your@email"` (SEC requires a contact).

## Pipeline

| Step | Script | What it does | Course unit |
|---|---|---|---|
| 1 | `src/01_download.py` | Download the AI device list, 510(k) file, classification file, PMA records and SEC tickers; log each in `data/raw/MANIFEST.csv` | APIs |
| 2 | `src/02_build_db.py` | Load everything into SQLite (`data/interim/fda.sqlite`) and build device-level views with SQL joins | SQL |
| 3 | `src/03_fetch_postmarket.py` | Pull recalls, recall classes and MAUDE adverse event reports from the openFDA API in batched queries | APIs |
| 4 | `src/04_scrape_summaries.py` | Download 1,605 summary PDFs (510(k), De Novo, PMA) and extract their text with pdfplumber | Web scraping |
| 5 | `src/05_code_validation.py` | Keyword/regex classifier of clinical validation (baseline); creates the hand-coding sample | Text analysis |
| 5b | `src/05b_llm_batches.py` | Prepare summary excerpts for LLM coding and merge the results | Programming with AI |
| 6 | `src/06_merge.py` | Public company matching; build `data/processed/devices.csv` | pandas |
| 7 | `src/07_models.py` | Exploratory figures, logistic, negative binomial and Cox models | Visualization, inference |
| | `src/make_coding_tool.py` | Builds `validation/coding_tool.html`, the page used for blind hand-coding | |

## Repository layout

```
data/raw/            downloaded source files, untouched; MANIFEST.csv records URL, pull time, checksum
data/raw/api/        raw openFDA API responses (JSON)
data/interim/        extracted summary text, LLM batches and outputs, classifier outputs, company matches
data/processed/      devices.csv, the analysis table (see data_dictionary.md)
validation/          codebook, hand-coded sample (100 devices), coding tool, company parent table
report/              report.qmd, rendered PDF/HTML, figures/, tables/
docs/                submitted analysis plan
```

Not committed (regenerable): the 1.9 GB of summary PDFs (`data/interim/pdfs/`, rebuilt by step 4)
and the SQLite database (rebuilt by step 2).

## Data sources

All public; pulled on 2026-10-05 (UTC). openFDA refreshes weekly, so a new pull will differ slightly.

| Source | Link |
|---|---|
| FDA AI-Enabled Medical Device List | <https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices> |
| FDA 510(k) clearance file `pmn96cur.zip` | <https://www.fda.gov/medical-devices/510k-clearances/downloadable-510k-files> |
| openFDA device classification, PMA, recall, enforcement, adverse event (MAUDE) | <https://open.fda.gov/apis/device/> |
| 510(k), De Novo and PMA summary documents | <https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm> |
| SEC EDGAR company tickers | <https://www.sec.gov/files/company_tickers.json> |

## How clinical validation was coded

A device counts as validated if its summary reports testing on patients or patient data with a
result (rule: [`validation/codebook.md`](validation/codebook.md)). The testing section of each
summary was coded yes/no by Claude Sonnet 5 subagents in Claude Code, each decision with a
supporting quote (`data/interim/validation_llm.csv`; prompt in `src/05b_llm_batches.py`).
A stratified random sample of 100 summaries was hand-coded blind to the model's output
(`validation/hand_coded_sample.csv`): agreement 90.4%, Cohen's κ = 0.80. A keyword classifier
reached κ = 0.61 on the same sample.

## Use of AI

Under the course's Tier 4 GenAI policy: Claude (Anthropic) coded the summaries as described above,
and assisted with writing the pipeline code and drafting the report. All results were checked
against the data, and the validation coding was checked against blind hand-coding.
