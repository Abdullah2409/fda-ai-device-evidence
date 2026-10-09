"""Step 2: load the raw files into SQLite and build device-level views with SQL.

Tables (one per source, lightly cleaned: ISO dates, trimmed IDs, no other changes)
  ai_devices          FDA AI-Enabled Device List, one row per device (base table)
  pmn                 510(k) and De Novo decisions from pmn96cur (all ~100k rows)
  pma                 PMA approvals and supplements for the AI-list PMA numbers
  classification      product code -> risk class, regulation number
  recalls             openFDA recall records (one row per recalled product)
  recall_submissions  recall record -> submission number (k_numbers / pma_numbers exploded)
  enforcement         recall class (I/II/III), joined on recall_number = product_res_number
  maude_events        MAUDE adverse event reports
  maude_pma_totals    report totals for PMAs whose records were too many to download
  sec_companies       SEC EDGAR public company names and tickers

Views
  v_device_base       ai_devices + review dates + risk class
  v_device_recalls    recall outcomes per device
  v_device_ae         adverse event outcomes per device
  v_devices           all three joined, one row per device

A note on PMA supplements: MAUDE and recalls only record the base PMA number, so an AI
feature approved as a supplement (e.g. P980016/S939) inherits every report and recall of
the whole product line. v_device_base flags these rows with `pma_supplement = 1`.
"""

import json
import sqlite3
import zipfile

import pandas as pd

from common import API_RAW, DB_PATH, RAW


def iso(series: pd.Series, fmt: str) -> pd.Series:
    """Parse dates in a known format and return 'YYYY-MM-DD' strings (None when missing)."""
    parsed = pd.to_datetime(series.replace("", None), format=fmt, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)


def pathway(submission: str) -> str:
    if submission.startswith("DEN"):
        return "De Novo"
    if submission.startswith("P"):
        return "PMA"
    return "510(k)"


def load_ai_devices() -> pd.DataFrame:
    df = pd.read_csv(RAW / "ai_device_list.csv", encoding="utf-8-sig", dtype=str)
    df.columns = ["decision_date", "submission_number", "device_name", "company", "panel", "product_code"]
    df = df.apply(lambda s: s.str.strip())
    df["decision_date"] = iso(df["decision_date"], "%m/%d/%Y")
    df["base_number"] = df["submission_number"].str.split("/").str[0]
    df["supplement_number"] = df["submission_number"].str.partition("/")[2]
    df["pathway"] = df["submission_number"].map(pathway)
    return df


def load_pmn() -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "pmn96cur.zip") as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep="|", dtype=str, encoding="latin-1", keep_default_na=False)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "knumber": "submission_number", "datereceived": "date_received", "decisiondate": "decision_date",
        "reviewadvisecomm": "review_advise_comm", "productcode": "product_code",
        "stateorsumm": "state_or_summ", "classadvisecomm": "class_advise_comm",
        "sspindicator": "ssp_indicator", "thirdparty": "third_party",
        "expeditedreview": "expedited_review", "devicename": "device_name",
    })
    for col in ("date_received", "decision_date"):
        df[col] = iso(df[col], "%m/%d/%Y")
    keep = ["submission_number", "applicant", "country_code", "date_received", "decision_date",
            "decision", "review_advise_comm", "product_code", "state_or_summ", "class_advise_comm",
            "type", "third_party", "expedited_review", "device_name"]
    return df[keep]


def load_pma() -> pd.DataFrame:
    df = pd.DataFrame(json.loads((API_RAW / "pma_ai_devices.json").read_text()))
    for col in ("date_received", "decision_date"):
        df[col] = iso(df[col], "%Y-%m-%d")
    keep = ["pma_number", "supplement_number", "supplement_type", "supplement_reason",
            "date_received", "decision_date", "decision_code", "product_code",
            "advisory_committee", "applicant", "trade_name", "expedited_review_flag"]
    return df[keep]


def load_classification() -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "classification.json.zip") as z:
        records = json.load(z.open(z.namelist()[0]))["results"]
    keep = ["product_code", "device_name", "device_class", "regulation_number", "medical_specialty",
            "medical_specialty_description", "review_panel", "implant_flag",
            "life_sustain_support_flag", "third_party_flag", "submission_type_id"]
    df = pd.DataFrame(records)[keep]
    # A handful of product codes appear twice; keep the first.
    return df.drop_duplicates("product_code")


def load_recalls() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = json.loads((API_RAW / "recalls.json").read_text())
    df = pd.DataFrame(records)
    keep = ["product_res_number", "res_event_number", "cfres_id", "event_date_initiated",
            "event_date_posted", "event_date_terminated", "recall_status", "product_code",
            "recalling_firm", "root_cause_description", "reason_for_recall", "action",
            "product_description"]
    recalls = df[keep].copy()
    for col in ("event_date_initiated", "event_date_posted", "event_date_terminated"):
        recalls[col] = iso(recalls[col], "%Y-%m-%d")

    links = [
        {"product_res_number": r["product_res_number"], "submission_number": num.strip(), "field": field}
        for r in records
        for field in ("k_numbers", "pma_numbers")
        for num in (r.get(field) or [])
    ]
    return recalls, pd.DataFrame(links).drop_duplicates()


def load_enforcement() -> pd.DataFrame:
    df = pd.DataFrame(json.loads((API_RAW / "recall_enforcement.json").read_text()))
    keep = ["recall_number", "event_id", "classification", "recall_initiation_date",
            "center_classification_date", "report_date", "status", "voluntary_mandated"]
    df = df[keep].copy()
    for col in ("recall_initiation_date", "center_classification_date", "report_date"):
        df[col] = iso(df[col], "%Y%m%d")
    return df


def load_maude() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = json.loads((API_RAW / "maude_events.json").read_text())
    rows = []
    for e in records:
        dev = (e.get("device") or [{}])[0]
        rows.append({
            "mdr_report_key": e["mdr_report_key"],
            "pma_pmn_number": (e.get("pma_pmn_number") or "").strip(),
            "date_received": e.get("date_received"),
            "date_of_event": e.get("date_of_event"),
            "event_type": e.get("event_type"),
            "report_source_code": e.get("report_source_code"),
            "adverse_event_flag": e.get("adverse_event_flag"),
            "product_problem_flag": e.get("product_problem_flag"),
            "brand_name": dev.get("brand_name"),
            "device_product_code": dev.get("device_report_product_code"),
            "manufacturer_name": dev.get("manufacturer_d_name"),
        })
    df = pd.DataFrame(rows)
    for col in ("date_received", "date_of_event"):
        df[col] = iso(df[col], "%Y%m%d")
    totals = json.loads((API_RAW / "maude_pma_totals.json").read_text())
    totals = pd.DataFrame(totals.items(), columns=["pma_number", "total_reports"])
    return df, totals


def load_sec() -> pd.DataFrame:
    data = json.loads((RAW / "company_tickers.json").read_text())
    df = pd.DataFrame(data.values()).rename(columns={"cik_str": "cik", "title": "company_name"})
    return df[["cik", "ticker", "company_name"]]


VIEWS = """
DROP VIEW IF EXISTS v_device_base;
CREATE VIEW v_device_base AS
SELECT
    a.submission_number,
    a.base_number,
    a.pathway,
    a.supplement_number <> '' AS pma_supplement,
    a.decision_date,
    a.device_name,
    a.company,
    a.panel,
    a.product_code,
    COALESCE(n.date_received, p.date_received) AS date_received,
    CAST(julianday(a.decision_date) - julianday(COALESCE(n.date_received, p.date_received)) AS INTEGER)
        AS review_days,
    n.third_party,
    n.state_or_summ,
    n.expedited_review,
    n.country_code,
    p.supplement_type AS pma_supplement_type,
    c.device_class AS risk_class,
    c.regulation_number,
    c.implant_flag,
    c.life_sustain_support_flag
FROM ai_devices a
LEFT JOIN pmn n ON n.submission_number = a.submission_number
LEFT JOIN pma p ON p.pma_number = a.base_number AND p.supplement_number = a.supplement_number
LEFT JOIN classification c ON c.product_code = a.product_code;

-- Only recalls initiated on or after the device's decision date count as outcomes.
-- Earlier ones exist because a recall of an older model (e.g. a CT platform) can list
-- the newer 510(k); they are counted separately for diagnostics.
DROP VIEW IF EXISTS v_device_recalls;
CREATE VIEW v_device_recalls AS
SELECT
    a.submission_number,
    COUNT(DISTINCT CASE WHEN r.event_date_initiated >= a.decision_date THEN r.res_event_number END)
        AS n_recall_events,
    COUNT(DISTINCT CASE WHEN r.event_date_initiated >= a.decision_date THEN r.product_res_number END)
        AS n_recall_products,
    MIN(CASE WHEN r.event_date_initiated >= a.decision_date THEN r.event_date_initiated END)
        AS first_recall_date,
    MIN(CASE WHEN r.event_date_initiated >= a.decision_date THEN
            CASE e.classification WHEN 'Class I' THEN 1 WHEN 'Class II' THEN 2 WHEN 'Class III' THEN 3 END
        END) AS most_severe_recall_class,
    COUNT(DISTINCT CASE WHEN r.event_date_initiated < a.decision_date THEN r.res_event_number END)
        AS n_recalls_before_decision
FROM ai_devices a
JOIN recall_submissions s ON s.submission_number = a.base_number
JOIN recalls r ON r.product_res_number = s.product_res_number
LEFT JOIN enforcement e ON e.recall_number = r.product_res_number
GROUP BY a.submission_number;

DROP VIEW IF EXISTS v_device_ae;
CREATE VIEW v_device_ae AS
SELECT
    a.submission_number,
    MAX(COUNT(m.mdr_report_key), COALESCE(t.total_reports, 0)) AS ae_reports,
    SUM(m.event_type = 'Death') AS ae_deaths,
    SUM(m.event_type = 'Injury') AS ae_injuries,
    SUM(m.event_type = 'Malfunction') AS ae_malfunctions,
    MIN(m.date_received) AS first_ae_date,
    COALESCE(t.total_reports, 0) > COUNT(m.mdr_report_key) AS ae_total_only
FROM ai_devices a
LEFT JOIN maude_events m ON m.pma_pmn_number = a.base_number
LEFT JOIN maude_pma_totals t ON t.pma_number = a.base_number
GROUP BY a.submission_number;

DROP VIEW IF EXISTS v_devices;
CREATE VIEW v_devices AS
SELECT
    b.*,
    COALESCE(r.n_recall_events, 0) > 0 AS recalled,
    COALESCE(r.n_recall_events, 0) AS n_recall_events,
    r.first_recall_date,
    r.most_severe_recall_class,
    COALESCE(r.n_recalls_before_decision, 0) AS n_recalls_before_decision,
    ae.ae_reports,
    ae.ae_reports > 0 AS any_ae,
    ae.ae_deaths,
    ae.ae_injuries,
    ae.ae_malfunctions,
    ae.first_ae_date,
    ae.ae_total_only
FROM v_device_base b
LEFT JOIN v_device_recalls r USING (submission_number)
LEFT JOIN v_device_ae ae USING (submission_number);
"""

INDEXES = """
CREATE UNIQUE INDEX ix_ai_submission ON ai_devices(submission_number);
CREATE UNIQUE INDEX ix_pmn_submission ON pmn(submission_number);
CREATE INDEX ix_pma_number ON pma(pma_number, supplement_number);
CREATE UNIQUE INDEX ix_class_code ON classification(product_code);
CREATE UNIQUE INDEX ix_recalls_res ON recalls(product_res_number);
CREATE INDEX ix_recall_sub ON recall_submissions(submission_number);
CREATE UNIQUE INDEX ix_enf_recall ON enforcement(recall_number);
CREATE INDEX ix_maude_pmn ON maude_events(pma_pmn_number);
"""


def check(con: sqlite3.Connection) -> None:
    """Print join coverage so problems show up immediately."""
    q = lambda sql: con.execute(sql).fetchone()
    print("\nJoin checks")
    print("  devices:", q("SELECT COUNT(*), COUNT(DISTINCT submission_number) FROM v_devices"))
    print("  with date_received / review_days:",
          q("SELECT COUNT(date_received), COUNT(review_days) FROM v_devices"))
    print("  with risk_class:", q("SELECT COUNT(risk_class) FROM v_devices")[0])
    print("  negative review_days:", q("SELECT COUNT(*) FROM v_devices WHERE review_days < 0")[0])
    print("  pmn decision_date disagreeing with AI list:", q("""
        SELECT COUNT(*) FROM ai_devices a JOIN pmn n USING (submission_number)
        WHERE n.decision_date <> a.decision_date""")[0])
    print("  recalled devices (all / excl. PMA supplements):", q("""
        SELECT SUM(recalled), SUM(recalled AND NOT pma_supplement) FROM v_devices"""))
    print("  devices with a recall dated before clearance (excluded from outcome):",
          q("SELECT COUNT(*) FROM v_devices WHERE n_recalls_before_decision > 0")[0])
    print("  devices with any AE (all / excl. PMA supplements):", q("""
        SELECT SUM(any_ae), SUM(any_ae AND NOT pma_supplement) FROM v_devices"""))
    print("\nBy pathway: n, recalled, any_ae")
    for row in con.execute("""SELECT pathway, COUNT(*), SUM(recalled), SUM(any_ae)
                              FROM v_devices GROUP BY pathway ORDER BY 2 DESC"""):
        print("  ", row)


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    recalls, recall_links = load_recalls()
    maude, maude_totals = load_maude()
    tables = {
        "ai_devices": load_ai_devices(),
        "pmn": load_pmn(),
        "pma": load_pma(),
        "classification": load_classification(),
        "recalls": recalls,
        "recall_submissions": recall_links,
        "enforcement": load_enforcement(),
        "maude_events": maude,
        "maude_pma_totals": maude_totals,
        "sec_companies": load_sec(),
    }
    with sqlite3.connect(DB_PATH) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False)
            print(f"{name:20s} {len(df):>7,} rows")
        con.executescript(INDEXES)
        con.executescript(VIEWS)
        check(con)
    print(f"\nwrote {DB_PATH}")


if __name__ == "__main__":
    main()
