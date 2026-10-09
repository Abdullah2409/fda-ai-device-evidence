"""Step 4: download each device's summary PDF and extract its text.

For each AI-list device we try the predictable accessdata.fda.gov URLs first:
  510(k)   cdrh_docs/pdf{YY}/K{YY}####.pdf      (pdf/ for 2001 and earlier)
  De Novo  cdrh_docs/reviews/DEN######.pdf      (some older ones under pdf{YY}/)
  PMA      cdrh_docs/pdf{YY}/P{YY}####B.pdf     (SSED; some only as P{YY}####.pdf)
If none exist, we scrape the device's page in the FDA database (pmn.cfm, denovo.cfm,
pma.cfm) for a link to a summary PDF.

Outputs
  data/interim/pdfs/{id}.pdf          downloaded PDFs (gitignored, ~600 MB)
  data/interim/summaries/{id}.txt     extracted text, pages separated by form feeds
  data/interim/scrape_log.csv         one row per device: url, status, pages, chars, scanned

The script is resumable: PDFs and text files that already exist are not fetched again.
Scanned PDFs (almost no extractable text) are flagged for OCR rather than OCR'd here.
"""

import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
import pdfplumber

from common import DB_PATH, INTERIM, session

DOCS = "https://www.accessdata.fda.gov/cdrh_docs"
DB_PAGES = {
    "510(k)": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={id}",
    "De Novo": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm?ID={id}",
    "PMA": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm?id={id}",
}
PDF_DIR = INTERIM / "pdfs"
TXT_DIR = INTERIM / "summaries"
LOG = INTERIM / "scrape_log.csv"

DOWNLOAD_WORKERS = 4
MIN_CHARS_PER_PAGE = 200  # below this average the PDF is treated as a scanned image


def file_id(submission: str) -> str:
    return submission.replace("/", "")


def year_folder(number: str) -> str:
    """'K231234' -> 'pdf23', 'K003301' -> 'pdf' (FDA keeps 2001 and earlier in pdf/)."""
    yy = int(number[-6:-4])
    return "pdf" if yy <= 1 or yy >= 76 else f"pdf{yy}"


def candidate_urls(submission: str, pathway: str) -> list[str]:
    base, _, supp = submission.partition("/")
    folder = year_folder(base)
    if pathway == "510(k)":
        return [f"{DOCS}/{folder}/{base}.pdf"]
    if pathway == "De Novo":
        return [f"{DOCS}/reviews/{base}.pdf", f"{DOCS}/{folder}/{base}.pdf"]
    name = base + supp  # P130020/S005 -> P130020S005
    return [f"{DOCS}/{folder}/{name}B.pdf", f"{DOCS}/{folder}/{name}.pdf",
            f"{DOCS}/pdf/{name}B.pdf", f"{DOCS}/pdf/{name}.pdf"]


def is_pdf(r) -> bool:
    return r.status_code == 200 and r.content[:5] == b"%PDF-"


def find_on_db_page(submission: str, pathway: str) -> list[str]:
    """Scrape the FDA database page for links to summary PDFs."""
    page_id = submission.replace("/", "") if pathway == "PMA" else submission
    r = session.get(DB_PAGES[pathway].format(id=page_id), timeout=60)
    if r.status_code != 200:
        return []
    links = re.findall(r'href="([^"]*cdrh_docs/[^"]+\.pdf)"', r.text, flags=re.I)
    # Keep only this device's own documents. Older De Novos were filed under a K number,
    # so their decision summary is reviews/K######.pdf; on a De Novo page we accept the
    # reviews/ folder (decision summaries) and prefer it over the 510(k) folder.
    own = file_id(submission).upper()
    def keep(link: str) -> bool:
        name = link.rsplit("/", 1)[-1].upper()
        return name.startswith(own) or (pathway == "De Novo" and "/reviews/" in link.lower())
    links = sorted((l for l in links if keep(l)), key=lambda l: "/reviews/" not in l.lower())
    return [l.replace("http://", "https://") if l.startswith("http") else "https://www.accessdata.fda.gov" + l
            for l in links]


def fetch(submission: str, pathway: str) -> dict:
    """Download one device's summary PDF, trying candidate URLs then the database page."""
    dest = PDF_DIR / f"{file_id(submission)}.pdf"
    row = {"submission_number": submission, "pathway": pathway, "url": None, "status": "missing", "bytes": 0}
    if dest.exists():
        return {**row, "status": "cached", "bytes": dest.stat().st_size}
    tried = set()
    for stage, urls in (("direct", lambda: candidate_urls(submission, pathway)),
                        ("db_page", lambda: find_on_db_page(submission, pathway))):
        try:
            url_list = urls()
        except Exception as e:  # database page unreachable
            row["status"] = f"error: {e.__class__.__name__}"
            continue
        for url in url_list:
            if url in tried:
                continue
            tried.add(url)
            for attempt in range(3):
                try:
                    r = session.get(url, timeout=120)
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                continue
            time.sleep(0.2)
            if is_pdf(r):
                dest.write_bytes(r.content)
                return {**row, "url": url, "status": stage, "bytes": len(r.content)}
    return row


def extract(submission: str) -> dict:
    """Extract text from a downloaded PDF. Runs in a worker process."""
    fid = file_id(submission)
    pdf_path, txt_path = PDF_DIR / f"{fid}.pdf", TXT_DIR / f"{fid}.txt"
    try:
        if not txt_path.exists():
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            txt_path.write_text("\f".join(pages))
        text = txt_path.read_text()
        n_pages = text.count("\f") + 1
        chars = len(text.replace("\f", "").strip())
        return {"submission_number": submission, "pages": n_pages, "chars": chars,
                "scanned": chars / n_pages < MIN_CHARS_PER_PAGE, "extract_error": None}
    except Exception as e:
        return {"submission_number": submission, "pages": None, "chars": None,
                "scanned": None, "extract_error": f"{e.__class__.__name__}: {e}"[:200]}


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        devices = pd.read_sql("SELECT submission_number, pathway FROM ai_devices ORDER BY decision_date DESC", con)

    print(f"Downloading {len(devices)} summaries with {DOWNLOAD_WORKERS} workers")
    fetched, t0 = [], time.time()
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(fetch, s, p) for s, p in devices.itertuples(index=False)]
        for i, f in enumerate(as_completed(futures), 1):
            fetched.append(f.result())
            if i % 100 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} ({time.time() - t0:.0f}s)", flush=True)
    log = pd.DataFrame(fetched)
    print(log["status"].value_counts().to_string())

    have = log.loc[log["status"] != "missing", "submission_number"]
    have = have[~have.str.startswith("error")]
    print(f"\nExtracting text from {len(have)} PDFs")
    with ProcessPoolExecutor() as pool:
        extracted = list(pool.map(extract, have, chunksize=8))
    log = log.merge(pd.DataFrame(extracted), on="submission_number", how="left")

    # Keep the URL from an earlier run for PDFs that were already cached.
    if LOG.exists():
        old = pd.read_csv(LOG, dtype=str).set_index("submission_number")
        cached = log["status"] == "cached"
        log.loc[cached, "url"] = log.loc[cached, "submission_number"].map(old["url"])
        log.loc[cached, "status"] = log.loc[cached, "submission_number"].map(old["status"])
    log.sort_values("submission_number").to_csv(LOG, index=False)

    print(f"\nwrote {LOG}")
    print("by pathway (found / total):")
    found = log["status"].isin(["direct", "db_page"])
    print(log.groupby("pathway").apply(lambda g: f"{found[g.index].sum()} / {len(g)}", include_groups=False).to_string())
    print("scanned (needs OCR):", int(log["scanned"].fillna(False).astype(bool).sum()))
    print("extract errors:", int(log["extract_error"].notna().sum()))


if __name__ == "__main__":
    main()
