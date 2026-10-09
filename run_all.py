"""Reproduce the analysis end to end.

    python run_all.py              rebuild from the committed raw data (no downloads)
    python run_all.py --download   also re-download every source first (openFDA refreshes weekly,
                                   so counts can change) and re-scrape the summary PDFs

The LLM coding of clinical validation (step 5b) is not re-run: it was done once by Claude
subagents and its outputs are committed in data/interim/llm_coding/. This script only merges them.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

DOWNLOAD = ["src/01_download.py", "src/03_fetch_postmarket.py"]
BUILD = [
    ["src/02_build_db.py"],
    ["src/05_code_validation.py"],          # regex classifier (comparison only)
    ["src/05b_llm_batches.py", "merge"],    # LLM coding results -> validation_llm.csv
    ["src/06_merge.py"],
    ["src/07_models.py"],
]


def run(args: list[str]) -> None:
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main() -> None:
    if "--download" in sys.argv:
        for script in DOWNLOAD:
            run([script])
        run(["src/02_build_db.py"])          # step 4 reads submission numbers from the database
        run(["src/04_scrape_summaries.py"])
    for args in BUILD:
        run(args)
    print("\n=== rendering report/report.qmd", flush=True)
    subprocess.run(["quarto", "render", "report.qmd"], cwd=ROOT / "report", check=True,
                   env={**__import__("os").environ, "QUARTO_PYTHON": PY})


if __name__ == "__main__":
    main()
