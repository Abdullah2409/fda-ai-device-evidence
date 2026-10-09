"""Step 3: pull recalls and MAUDE adverse event reports for every AI-list device from the openFDA API.

The bulk files are too big to be worth it here (recall 275 MB, MAUDE 18 GB), and the
API can filter by submission number directly:
  recall: k_numbers (510(k) and De Novo numbers) and pma_numbers
  event:  pma_pmn_number
Recall records have no recall class (I/II/III); that comes from the enforcement
endpoint, joined on recall_number = product_res_number.

Submission numbers are sent 200 at a time in one OR query and pages of 999 are
followed with `skip`. The whole pull is about 25 requests, far below the no-key
limit of 1,000 requests/day. Without a key the event endpoint answers 403 to
`limit=1000` (999 works) and to `count=` combined with `search=`, so we page at
999 and count reports locally.

PMA numbers are queried one at a time. MAUDE only records the base PMA number, so an
AI feature added as a PMA supplement (e.g. P980016/S939, a Medtronic ICD) inherits every
report for the whole product line: tens of thousands, beyond openFDA's skip limit of
25,000. For PMAs above MAX_RECORDS only the total is kept; step 6 decides how to treat them.

Output (raw, untouched API results):
  data/raw/api/recalls.json
  data/raw/api/recall_enforcement.json
  data/raw/api/maude_events.json
  data/raw/api/maude_pma_totals.json   {pma_number: total reports in MAUDE}
"""

import json

import pandas as pd

from common import API_RAW, RAW, openfda_get, or_query, record_manifest

BATCH = 200
PAGE = 999  # the event endpoint rejects limit=1000 without an API key
MAX_RECORDS = 5000  # per PMA; above this keep the total only


def submission_ids() -> tuple[list[str], list[str]]:
    """Return (510(k) and De Novo numbers, base PMA numbers) from the AI list."""
    ai = pd.read_csv(RAW / "ai_device_list.csv", encoding="utf-8-sig", dtype=str)
    ids = ai["Submission Number"].str.strip()
    pmn = sorted(ids[~ids.str.startswith("P")])
    pma = sorted({s.split("/")[0] for s in ids[ids.str.startswith("P")]})
    return pmn, pma


def fetch_all(endpoint: str, field: str, values: list[str]) -> tuple[list[dict], int]:
    """Run batched OR searches with pagination. Returns (records, number of requests)."""
    records, calls = [], 0
    for i in range(0, len(values), BATCH):
        search = or_query(field, values[i:i + BATCH])
        skip = 0
        while True:
            page = openfda_get(endpoint, {"search": search, "limit": PAGE, "skip": skip})
            calls += 1
            batch = page.get("results", [])
            records.extend(batch)
            skip += len(batch)
            if not batch or skip >= page["meta"]["results"]["total"]:
                break
    return records, calls


def fetch_pma_events(pma: list[str]) -> tuple[list[dict], dict[str, int], int]:
    """Fetch MAUDE reports one PMA at a time. Returns (records, totals per PMA, requests)."""
    records, totals, calls = [], {}, 0
    for p in pma:
        first = openfda_get("device/event", {"search": or_query("pma_pmn_number", [p]), "limit": 1})
        calls += 1
        totals[p] = first.get("meta", {}).get("results", {}).get("total", 0)
        if 0 < totals[p] <= MAX_RECORDS:
            recs, c = fetch_all("device/event", "pma_pmn_number", [p])
            records.extend(recs)
            calls += c
        elif totals[p] > MAX_RECORDS:
            print(f"  {p}: {totals[p]:,} reports, total kept but records not downloaded")
    return records, totals, calls


def save(records: list[dict], name: str, source: str) -> None:
    dest = API_RAW / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(records))
    record_manifest(dest, source)


def main() -> None:
    pmn, pma = submission_ids()
    print(f"{len(pmn)} 510(k)/De Novo numbers, {len(pma)} PMA numbers")

    recalls_k, c1 = fetch_all("device/recall", "k_numbers", pmn)
    recalls_p, c2 = fetch_all("device/recall", "pma_numbers", pma)
    # A recall record that lists several AI devices comes back once per batch it matches.
    recalls = list({r["product_res_number"]: r for r in recalls_k + recalls_p}.values())
    save(recalls, "recalls.json",
         "https://api.fda.gov/device/recall.json (search k_numbers / pma_numbers in batches of 200)")
    print(f"recalls: {len(recalls)} unique recall records ({c1 + c2} requests)")

    enforcement, c5 = fetch_all("device/enforcement", "recall_number",
                                sorted({r["product_res_number"] for r in recalls}))
    save(enforcement, "recall_enforcement.json",
         "https://api.fda.gov/device/enforcement.json (search recall_number in batches of 200)")
    print(f"enforcement: {len(enforcement)} reports with recall class ({c5} requests)")

    events, c3 = fetch_all("device/event", "pma_pmn_number", pmn)
    pma_events, pma_totals, c4 = fetch_pma_events(pma)
    events = list({e["mdr_report_key"]: e for e in events + pma_events}.values())
    save(events, "maude_events.json",
         "https://api.fda.gov/device/event.json (search pma_pmn_number; 510(k)/De Novo in batches of 200, PMA one at a time)")
    save(pma_totals, "maude_pma_totals.json",
         "https://api.fda.gov/device/event.json (search pma_pmn_number per PMA, limit=1, meta.results.total)")
    print(f"MAUDE: {len(events)} unique event reports ({c3 + c4} requests)")
    print(f"total requests: {c1 + c2 + c3 + c4 + c5}")


if __name__ == "__main__":
    main()
