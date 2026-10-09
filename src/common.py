"""Shared paths and HTTP helpers for the pipeline scripts."""

import csv
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
API_RAW = RAW / "api"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = RAW / "MANIFEST.csv"
DB_PATH = INTERIM / "fda.sqlite"

OPENFDA = "https://api.fda.gov"
# Optional. Without a key openFDA allows 240 requests/minute and 1,000/day per IP,
# which is enough for this pipeline (about 25 calls). A key is only required for
# `count=` queries combined with `search=`, which we do not use.
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY")

# fda.gov rejects requests without a browser-like User-Agent.
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) fda-ai-device-evidence research project"

session = requests.Session()
session.headers["User-Agent"] = BROWSER_UA


def download(url: str, dest: Path, headers: dict | None = None, retries: int = 3) -> Path:
    """Stream a URL to disk and record it in the manifest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
            break
        except requests.RequestException as e:
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {url}: {e}")
            time.sleep(5 * attempt)
    record_manifest(dest, url)
    return dest


def openfda_get(endpoint: str, params: dict, retries: int = 3) -> dict:
    """GET an openFDA endpoint. Returns {} when there are no matches (openFDA answers 404)."""
    params = dict(params)
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY
    for attempt in range(1, retries + 1):
        r = session.get(f"{OPENFDA}/{endpoint}.json", params=params, timeout=120)
        if r.status_code == 404:
            return {}
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(10 * attempt)
            continue
        r.raise_for_status()
        time.sleep(0.3)  # stay well under 240 requests/minute
        return r.json()
    r.raise_for_status()
    return {}


def or_query(field: str, values: list[str]) -> str:
    """Build an openFDA search like k_numbers:("K1" OR "K2")."""
    return f'{field}:(' + " OR ".join(f'"{v}"' for v in values) + ")"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_manifest(path: Path, source: str) -> None:
    """Add or replace a row in data/raw/MANIFEST.csv (file, source, pull date, size, checksum)."""
    rel = path.relative_to(ROOT).as_posix()
    rows = []
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["file"] != rel]
    rows.append({
        "file": rel,
        "source": source,
        "pulled_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })
    rows.sort(key=lambda r: r["file"])
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "source", "pulled_utc", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
