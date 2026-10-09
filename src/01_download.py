"""Step 1: download the base table and reference files into data/raw/.

Files:
  ai_device_list.csv         FDA AI-Enabled Medical Device List (base table)
  pmn96cur.zip               FDA 510(k) clearance file, 1996-current (also covers De Novo)
  classification.json.zip    openFDA device classification bulk file (risk class by product code)
  api/pma_ai_devices.json    openFDA PMA records for the PMA devices on the AI list
  company_tickers.json       SEC EDGAR public company tickers (needs SEC_USER_AGENT)

Every file is logged in data/raw/MANIFEST.csv with its source URL, pull date and checksum.
"""

import json
import os

import pandas as pd

from common import API_RAW, RAW, download, openfda_get, or_query, record_manifest

AI_LIST_URL = "https://www.fda.gov/media/178541/download?attachment"
PMN_URL = "https://www.accessdata.fda.gov/premarket/ftparea/pmn96cur.zip"
CLASSIFICATION_URL = "https://download.open.fda.gov/device/classification/device-classification-0001-of-0001.json.zip"
SEC_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch_pma_records(ai_list: pd.DataFrame) -> None:
    """The 21 PMA devices are not in pmn96cur, so get their dates from the openFDA PMA API."""
    pma_ids = ai_list.loc[ai_list["Submission Number"].str.startswith("P"), "Submission Number"]
    base_ids = sorted({s.split("/")[0] for s in pma_ids})
    results, skip = [], 0
    while True:
        page = openfda_get("device/pma", {"search": or_query("pma_number", base_ids), "limit": 1000, "skip": skip})
        batch = page.get("results", [])
        results.extend(batch)
        skip += len(batch)
        if not batch or skip >= page["meta"]["results"]["total"]:
            break
    dest = API_RAW / "pma_ai_devices.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=1))
    record_manifest(dest, f"https://api.fda.gov/device/pma.json?search={or_query('pma_number', base_ids)}")
    print(f"  {len(results)} PMA records (incl. supplements) for {len(base_ids)} PMA numbers")


def main() -> None:
    print("AI device list")
    download(AI_LIST_URL, RAW / "ai_device_list.csv")
    ai_list = pd.read_csv(RAW / "ai_device_list.csv", encoding="utf-8-sig", dtype=str)
    dates = pd.to_datetime(ai_list["Date of Final Decision"], format="%m/%d/%Y")
    print(f"  {len(ai_list)} devices, decisions {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")

    print("510(k) file (pmn96cur.zip)")
    download(PMN_URL, RAW / "pmn96cur.zip")

    print("Classification bulk file")
    download(CLASSIFICATION_URL, RAW / "classification.json.zip")

    print("PMA records")
    fetch_pma_records(ai_list)

    print("SEC company tickers")
    sec_ua = os.environ.get("SEC_USER_AGENT")
    sec_file = RAW / "company_tickers.json"
    if sec_ua:
        download(SEC_URL, sec_file, headers={"User-Agent": sec_ua})
    elif sec_file.exists():
        # Saved by hand from a browser, which avoids sending a contact email to SEC.
        record_manifest(sec_file, f"{SEC_URL} (saved manually in a browser)")
        print("  using manually saved file")
    else:
        print(f"  skipped: save {SEC_URL} to data/raw/company_tickers.json in a browser,")
        print('  or set SEC_USER_AGENT="Your Name your@email" (SEC requires a name and email)')


if __name__ == "__main__":
    main()
