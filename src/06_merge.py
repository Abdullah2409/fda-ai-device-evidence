"""Step 6: flag public companies and build the analysis table data/processed/devices.csv.

Public company status
  1. Normalize company names (lowercase, drop punctuation and legal suffixes such as Inc,
     GmbH, Ltd) and match them exactly to SEC EDGAR company_tickers.json. Fuzzy matching
     was tried and rejected: every match scoring below 100 was a false pair
     (e.g. "Qure.ai Technologies" -> "OraSure Technologies").
  2. Map subsidiaries and acquired companies to their parent with the hand-made table
     validation/company_parents.csv (regex pattern -> parent, public, listing).
  Two flags result:
    sec_listed      company or parent is an SEC registrant (US exchange or ADR)
    public_company  company or parent is publicly traded on any exchange, as of the pull date
  Every match is written to data/interim/company_matches.csv for review.

devices.csv joins v_devices (SQLite) with the validation coding (LLM, step 5b) and
the company flags, and derives the analysis variables. See data_dictionary.md.
"""

import json
import re
import sqlite3

import numpy as np
import pandas as pd

from common import DB_PATH, INTERIM, PROCESSED, RAW, ROOT

PARENTS = ROOT / "validation" / "company_parents.csv"
LLM_CODES = INTERIM / "validation_llm.csv"
REGEX_CODES = INTERIM / "validation_coding.csv"
MATCHES = INTERIM / "company_matches.csv"
OUT = PROCESSED / "devices.csv"

SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|l l c|gmbh|ag|sa|s a|bv|b v|nv|n v"
    r"|plc|kk|k k|ab|publ|oy|srl|spa|s p a|pty|holdings?|group|the|adr|sas|lp|ulc|ltda|s r l|a s|as)\b")


def normalize(name: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    name = SUFFIXES.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


def pull_date() -> pd.Timestamp:
    """Date the post-market data were pulled (from the manifest), used as the censoring date."""
    manifest = pd.read_csv(RAW / "MANIFEST.csv")
    row = manifest.loc[manifest["file"] == "data/raw/api/recalls.json", "pulled_utc"].iloc[0]
    return pd.Timestamp(row[:10])


def company_flags(companies: pd.Series, sec: pd.DataFrame) -> pd.DataFrame:
    sec = sec.assign(key=sec["company_name"].map(normalize)).drop_duplicates("key")
    sec_by_key = sec.set_index("key")
    sec_tickers = set(sec["ticker"])
    parents = pd.read_csv(PARENTS)
    missing = [l for l in parents["listing"].dropna() if l.startswith("SEC:") and l[4:] not in sec_tickers]
    if missing:
        print("  warning: parent listings not found in SEC file:", missing)

    rows = []
    for company in sorted(companies.unique()):
        key = normalize(company)
        rule = next((r for r in parents.itertuples(index=False) if re.search(r.pattern, key)), None)
        if rule is not None:
            listing = rule.listing if isinstance(rule.listing, str) else ""
            rows.append({"company": company, "key": key, "method": "parent_table", "parent": rule.parent,
                         "listing": listing, "sec_listed": int(listing.startswith("SEC:")),
                         "public_company": int(rule.public), "confidence": rule.confidence})
        elif key in sec_by_key.index:
            hit = sec_by_key.loc[key]
            rows.append({"company": company, "key": key, "method": "sec_exact", "parent": hit["company_name"],
                         "listing": f"SEC:{hit['ticker']}", "sec_listed": 1, "public_company": 1,
                         "confidence": "high"})
        else:
            rows.append({"company": company, "key": key, "method": "none", "parent": "", "listing": "",
                         "sec_listed": 0, "public_company": 0, "confidence": ""})
    return pd.DataFrame(rows)


SPECIALTY = {"Radiology": "Radiology", "Cardiovascular": "Cardiovascular", "Neurology": "Neurology"}


def main() -> None:
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql("SELECT * FROM v_devices", con)
        sec = pd.read_sql("SELECT ticker, company_name FROM sec_companies", con)

    flags = company_flags(df["company"], sec)
    flags.to_csv(MATCHES, index=False)
    df = df.merge(flags[["company", "parent", "listing", "sec_listed", "public_company"]], on="company", how="left")

    llm = pd.read_csv(LLM_CODES)[["submission_number", "llm_validated", "quote"]]
    rx = pd.read_csv(REGEX_CODES)[["submission_number", "validation_level"]]
    df = df.merge(llm, on="submission_number", how="left").merge(rx, on="submission_number", how="left")
    df["validated"] = df["llm_validated"].map({"yes": 1, "no": 0})          # NaN = no usable summary text
    df["validated_regex"] = df["validation_level"].map(
        {"retrospective": 1, "prospective": 1, "none": 0})                    # sensitivity check only
    df = df.rename(columns={"quote": "validation_quote"})

    pulled = pull_date()
    decided = pd.to_datetime(df["decision_date"])
    df["decision_year"] = decided.dt.year
    df["years_on_market"] = ((pulled - decided).dt.days / 365.25).round(3)
    df["specialty"] = df["panel"].map(SPECIALTY).fillna("Other")
    df["recalled"] = df["recalled"].astype(int)
    df["any_ae"] = df["any_ae"].astype(int)
    # SQL SUM over no rows gives NULL; those devices have zero reports. Leave the breakdown
    # missing only where MAUDE records were too many to download (total known, types not).
    known = df["ae_total_only"] == 0
    for col in ("ae_deaths", "ae_injuries", "ae_malfunctions"):
        df.loc[known, col] = df.loc[known, col].fillna(0).astype(int)
    df["days_to_first_recall"] = (pd.to_datetime(df["first_recall_date"]) - decided).dt.days
    df["days_observed"] = np.where(df["recalled"] == 1, df["days_to_first_recall"], (pulled - decided).dt.days)
    # Devices whose post-market records cannot be attributed to the AI feature (base-PMA records).
    df["in_model_sample"] = ((df["pma_supplement"] == 0) & df["validated"].notna()).astype(int)

    cols = ["submission_number", "pathway", "decision_date", "decision_year", "device_name", "company",
            "parent", "listing", "public_company", "sec_listed", "panel", "specialty", "product_code",
            "risk_class", "regulation_number", "date_received", "review_days", "third_party",
            "years_on_market", "validated", "validated_regex", "validation_level", "validation_quote",
            "recalled", "n_recall_events", "first_recall_date", "most_severe_recall_class",
            "days_to_first_recall", "days_observed", "n_recalls_before_decision",
            "ae_reports", "any_ae", "ae_deaths", "ae_injuries", "ae_malfunctions", "ae_total_only",
            "pma_supplement", "in_model_sample"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT, index=False)

    print(f"wrote {OUT}: {len(df)} devices, pull date {pulled.date()}")
    print("company matching (devices):",
          df.merge(flags[["company", "method"]], on="company")["method"].value_counts().to_dict())
    print(f"public_company: {df['public_company'].mean():.1%}  sec_listed: {df['sec_listed'].mean():.1%}")
    print("validated:", df["validated"].value_counts(dropna=False).to_dict())
    print(f"model sample: {df['in_model_sample'].sum()} devices, "
          f"{df.loc[df['in_model_sample'] == 1, 'recalled'].sum()} recalled, "
          f"{df.loc[df['in_model_sample'] == 1, 'any_ae'].sum()} with adverse events")


if __name__ == "__main__":
    main()
