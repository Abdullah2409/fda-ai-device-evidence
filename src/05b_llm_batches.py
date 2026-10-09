"""Step 5b: prepare summary excerpts for LLM coding of clinical validation, and merge the results.

  prepare  write data/interim/llm_batches/batch_NN.txt: each device's testing section
           (from the first testing heading to the conclusion, capped), ~35k tokens per file,
           wrapped into long lines so a subagent can read a file in two calls
  merge    combine data/interim/llm_coding/batch_NN.jsonl into data/interim/validation_llm.csv

The coding itself was done by Claude subagents (Claude Code, Sonnet), one per batch file,
using the yes/no rule in validation/codebook.md. Each answer carries a supporting quote so
it can be checked. The prompt is in LLM_PROMPT below and is cited in the report.

Usage:  .venv/bin/python src/05b_llm_batches.py prepare | merge
"""

import json
import re
import sqlite3
import sys

import pandas as pd

from common import DB_PATH, INTERIM
from make_coding_tool import prepare as locate_testing

TXT_DIR = INTERIM / "summaries"
BATCH_DIR = INTERIM / "llm_batches"
RESULT_DIR = INTERIM / "llm_coding"
OUT = INTERIM / "validation_llm.csv"

BATCH_CHARS = 140_000      # about 35k tokens per batch file
LINE_CHARS = 1_500         # long lines: fewer lines per file, fewer read calls
MAX_CHARS = 16_000         # about 4k tokens; covers ~90% of 510(k) testing sections whole
MAX_CHARS_NO_HEADING = 24_000
MAX_CHARS_LONG_DOC = 48_000  # De Novo / PMA decision summaries put results far down
CONCLUSION = re.compile(r"(?im)^[^\n]{0,15}?\bconclusions?\b[^\n]{0,60}$")

LLM_PROMPT = """You are coding FDA medical device summaries for a research project.
For each device, answer one question:

Does this summary report testing of the device on patients or patient data, with a result?
  1. Tested on real patient data (scans, signals, records, or enrolled patients)? If not: no.
  2. Reports a result (sensitivity, specificity, AUC, accuracy, Dice, agreement, or met
     predefined acceptance criteria)? If not: no.
  3. Is the result from this submission? If yes: yes.

Borderline rules:
  - Bench, phantom, simulated or synthetic data, software verification/validation only: no
  - Clinicians judged "sample clinical images" acceptable with no endpoint or criteria: no
  - Clinicians scored image quality on a predefined scale with acceptance criteria: yes
  - "Clinical performance unchanged from K######" with no new data here: no
  - "No clinical testing was required" but algorithm accuracy on patient scans is reported: yes
  - A reader study (clinicians read cases with and without the device): yes
  - Ignore boilerplate negations that refer to other things; judge what was actually done.
"""


def excerpt(text: str, pathway: str) -> tuple[str, bool]:
    """Testing section of one summary, and whether it was truncated."""
    body, anchor = locate_testing(text, pathway)
    if anchor >= 0:
        end = CONCLUSION.search(body, anchor + 200)
        section = body[anchor:end.start() if end else len(body)]
        limit = MAX_CHARS
    else:
        section, limit = body, MAX_CHARS_NO_HEADING
    if pathway != "510(k)":
        limit = MAX_CHARS_LONG_DOC
    section = re.sub(r"[ \t]+", " ", section)
    section = re.sub(r"\n\s*\n+", "\n", section)
    return section[:limit], len(section) > limit


def prepare() -> None:
    with sqlite3.connect(DB_PATH) as con:
        devices = pd.read_sql("SELECT submission_number, pathway FROM ai_devices ORDER BY submission_number", con)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.txt"):
        old.unlink()
    items = []
    for sub, pathway in devices.itertuples(index=False):
        path = TXT_DIR / f"{sub.replace('/', '')}.txt"
        if not path.exists():
            continue
        text, truncated = excerpt(path.read_text(), pathway)
        if len(text.strip()) < 200:  # scanned PDF with no text
            continue
        items.append((sub, pathway, text, truncated))
    batches, current, size = [], [], 0
    for item in items:
        if current and size + len(item[2]) > BATCH_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item[2])
    batches.append(current)
    for n, chunk in enumerate(batches):
        parts = [f"##### DEVICE {sub} | {pathway}{' | excerpt truncated' if trunc else ''}\n{wrap(text)}\n"
                 for sub, pathway, text, trunc in chunk]
        (BATCH_DIR / f"batch_{n:02d}.txt").write_text("\n".join(parts))
    total = sum(len(t) for _, _, t, _ in items)
    print(f"{len(items)} devices in {len(batches)} batches, ~{total / 4 / 1e6:.1f}M tokens")


def wrap(text: str) -> str:
    """Join lines into ~LINE_CHARS-long lines (line breaks become ' / '), never over 1,900 chars."""
    flat = " / ".join(p for p in text.split("\n") if p.strip())
    out = []
    while flat:
        if len(flat) <= 1_900:
            out.append(flat)
            break
        cut = flat.rfind(" ", LINE_CHARS, 1_900)
        cut = cut if cut > 0 else 1_900
        out.append(flat[:cut])
        flat = flat[cut:].lstrip()
    return "\n".join(out)


# Checked by hand after the run: the extracted text had no sponsor summary (only FDA's
# clearance letter, or the decision summary is not yet published), so the model's answer
# reflects missing text, not missing validation.
NO_SUMMARY_TEXT = {
    "K200852", "K241847", "K250754", "K251901", "K252099", "K252708",  # letter-only text, coded "no"
    "DEN250007", "DEN250028", "DEN250057",  # FDA has not posted the decision summary yet
}


def merge() -> None:
    rows = []
    for f in sorted(RESULT_DIR.glob("batch_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append({**json.loads(line), "batch": f.stem})
    df = pd.DataFrame(rows).drop_duplicates("submission_number", keep="last")
    df["llm_validated_raw"] = df["llm_validated"]
    df.loc[df["submission_number"].isin(NO_SUMMARY_TEXT), "llm_validated"] = "not_available"
    # "unsure" here almost always means unreadable or truncated text (scanned pages, broken fonts)
    df.loc[df["llm_validated"] == "unsure", "llm_validated"] = "not_available"
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} devices")
    print(df["llm_validated"].value_counts().to_string())


if __name__ == "__main__":
    {"prepare": prepare, "merge": merge}[sys.argv[1]]()
