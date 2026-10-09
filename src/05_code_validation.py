"""Step 5: code each summary's reported clinical validation with a rule-based text classifier.

validation_level (plan.md section 4)
  none           only bench, software verification or other non-clinical testing
  retrospective  performance measured on existing patient data (stored scans, records)
  prospective    patients enrolled going forward, or a reader study (MRMC)
  not_available  no summary text (missing PDF, statement instead of summary, scanned image)

How it works
  1. Drop FDA's clearance letter and the Indications for Use form: for 510(k)s, start
     at the "510(k) Summary" heading. De Novo and PMA documents are used whole.
  2. Split into sentences and look for evidence terms in each.
  3. A term in a negated sentence ("no clinical testing was required", "no prospectively
     gathered data") does not count; the negation is recorded as its own flag.
  4. Highest level with evidence wins: prospective > retrospective > none.

Outputs
  data/interim/validation_coding.csv     one row per device: level, flags, evidence snippets
  validation/hand_coded_sample.csv       created once: 100 devices to hand-code (stratified)
  If the hand-coded sample is filled in, prints Cohen's kappa and per-level precision/recall.
"""

import re
import sqlite3

import pandas as pd

from common import DB_PATH, INTERIM, ROOT

TXT_DIR = INTERIM / "summaries"
SCRAPE_LOG = INTERIM / "scrape_log.csv"
OUT = INTERIM / "validation_coding.csv"
SAMPLE = ROOT / "validation" / "hand_coded_sample.csv"
LEVELS = ["none", "retrospective", "prospective"]

NEGATION = re.compile(
    r"\b(no|not|none|neither|nor|without|n/a|did not|was not|were not|is not|are not|"
    r"not required|not necessary|not needed|not applicable)\b", re.I)

PATTERNS = {
    # prospective: enrolment going forward, a clinical trial, or a reader study
    # "prospective gating" / "prospectively gated" are CT acquisition modes, not study designs
    "prospective": re.compile(r"\bprospective(ly)?\b(?! (ecg[- ])?(gat|trigger))", re.I),
    "enrolled": re.compile(r"\b(enrolled|enrollment|enrolment|were recruited)\b", re.I),
    "clinical_trial": re.compile(
        r"\b(clinical trial|pivotal (clinical )?(study|trial)|randomi[sz]ed,? (controlled|clinical|multi-?cent(er|re))"
        r"|randomi[sz]ed (to|into) )", re.I),
    "reader_study": re.compile(
        r"\b(reader study|observer study|multi-?reader|MRMC|reader performance|readers? (were|evaluated|reviewed)"
        r"|\d+ (board[- ]certified )?(readers|radiologists|clinicians|physicians) (read|reviewed|interpreted|evaluated)"
        r"|(radiologists|readers|clinicians) (read|interpreted|reviewed) .{0,40}(with and without|aided|unaided))",
        re.I),
    # retrospective: performance on existing patient data
    "retrospective": re.compile(r"\bretrospective(ly)?\b", re.I),
    "standalone": re.compile(r"\bstand-?alone (performance|testing|study|evaluation|assessment)\b", re.I),
    "metrics": re.compile(
        r"\b(sensitivity|specificity|AUC|ROC|area under the|PPV|NPV|positive predictive|negative predictive"
        r"|dice (coefficient|score|similarity)|mean absolute error|kappa|concordance|agreement rate"
        r"|correlation coefficient|Bland-Altman|ICC|intraclass|mean (absolute )?(error|difference)"
        r"|true positive rate|false positive rate)\b", re.I),
    "patient_data": re.compile(
        r"\b(\d[\d,]*\s+(patients|subjects|cases|scans|studies|exams|examinations|images|participants|records))\b"
        r"|\b(clinical (data|dataset|datasets|images|cases|performance (data|testing|study|evaluation)))\b",
        re.I),
    "test_data": re.compile(
        r"\b(test (data)?sets?|testing datasets?|validation datasets?|ground[- ]truth|reference standard"
        r"|annotated by|expert (annotation|segmentation|consensus|reader)s?|truthed|truthing"
        r"|compared (to|with|against) (the )?(expert|manual|annotat|clinician|radiologist))", re.I),
    "clinical_study": re.compile(r"\bclinical (study|studies|testing|validation|evaluation|investigation)\b", re.I),
    # non-clinical
    "bench": re.compile(
        r"\b(bench testing|non-?clinical|software verification|verification and validation|V&V"
        r"|phantom|IEC 62304|performance bench)\b", re.I),
}
PROSPECTIVE_TERMS = ["prospective", "enrolled", "clinical_trial", "reader_study"]
RETROSPECTIVE_TERMS = ["retrospective", "standalone", "metrics", "test_data", "patient_data", "clinical_study"]

# A line (or page start) with "510(k) Summary" near its beginning, allowing prefixes such as
# "Section 5:" or "Special 510(k)". The clearance letter never starts a line this way.
SUMMARY_START = re.compile(r"(?:^|\f)[^\n\f]{0,30}?510\s*\(\s*k\s*\)\s*summary\b", re.I | re.M)
REFERENCES = re.compile(r"(?:^|\f)\s*(references|bibliography|literature cited)\s*$", re.I | re.M)
NUMBER_N = re.compile(
    r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\s+(patients|subjects|cases|scans|studies|exams|images|participants)\b", re.I)


def summary_text(text: str, pathway: str) -> tuple[str, bool]:
    """Return the part of the document written by the sponsor, and whether a summary heading was found."""
    if pathway != "510(k)":
        return text, True
    m = SUMMARY_START.search(text)
    if m:
        return text[m.start():], True
    # Fall back to dropping the first page or two (letter + indications form).
    pages = text.split("\f")
    return "\f".join(pages[2:]) if len(pages) > 3 else text, False


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+(?=[A-Z(•\-])", text) if len(s.strip()) > 20]


def classify(text: str, pathway: str) -> dict:
    body, found = summary_text(text, pathway)
    # Reference lists cite other people's trials; drop them.
    ref = REFERENCES.search(body)
    if ref:
        body = body[:ref.start()]
    hits = {k: 0 for k in PATTERNS}
    negated = {k: 0 for k in PATTERNS}
    evidence = {}
    for sent in sentences(body):
        neg = bool(NEGATION.search(sent))
        retro = bool(PATTERNS["retrospective"].search(sent))
        for name, pat in PATTERNS.items():
            # "a retrospective, multicenter pivotal study" is a retrospective study
            if name == "clinical_trial" and retro:
                continue
            if pat.search(sent):
                if neg and name not in ("metrics", "patient_data", "test_data"):
                    negated[name] += 1
                else:
                    hits[name] += 1
                    evidence.setdefault(name, sent[:240])

    if any(hits[t] for t in PROSPECTIVE_TERMS):
        level = "prospective"
    elif (hits["retrospective"] or hits["standalone"]
          or (hits["metrics"] and (hits["patient_data"] or hits["clinical_study"] or hits["test_data"]))
          or (hits["test_data"] and hits["patient_data"])):
        level = "retrospective"
    else:
        level = "none"

    sizes = [int(n.replace(",", "")) for n, _ in NUMBER_N.findall(body)]
    # Snippet for the strongest evidence found, so a reviewer can check the call quickly.
    top = next((k for k in PROSPECTIVE_TERMS + RETROSPECTIVE_TERMS + ["bench"] if k in evidence), None)
    return {
        "validation_level": level,
        "summary_found": found,
        "max_n": max(sizes) if sizes else None,
        "clinical_not_required": negated["clinical_study"] > 0 or negated["prospective"] > 0,
        **{f"hit_{k}": v for k, v in hits.items()},
        "evidence_term": top,
        "evidence": evidence.get(top, ""),
    }


def make_sample(coded: pd.DataFrame, devices: pd.DataFrame, n: int = 100, seed: int = 90819) -> None:
    """Draw a hand-coding sample stratified by decision period and radiology vs. other panels."""
    df = devices.merge(coded[["submission_number", "validation_level"]], on="submission_number")
    df = df[df["validation_level"] != "not_available"].copy()
    df["period"] = pd.cut(pd.to_datetime(df["decision_date"]).dt.year,
                          [0, 2017, 2020, 2022, 2024, 2100], labels=["<=2017", "2018-20", "2021-22", "2023-24", "2025-26"])
    df["radiology"] = df["panel"].eq("Radiology")
    per_cell = n // (df["period"].nunique() * 2)
    sample = (df.groupby(["period", "radiology"], observed=True, group_keys=False)
                .apply(lambda g: g.sample(min(per_cell, len(g)), random_state=seed), include_groups=False))
    if len(sample) < n:  # top up from the remaining devices if a cell was small
        rest = df.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n - len(sample), random_state=seed)])
    out = sample[["submission_number", "pathway", "decision_date", "device_name", "company", "panel", "url"]].copy()
    out["hand_validated"] = ""  # yes / no / unsure, filled in with validation/coding_tool.html
    out["notes"] = ""
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("submission_number").to_csv(SAMPLE, index=False)
    print(f"wrote {SAMPLE} ({len(out)} devices) - code it with src/make_coding_tool.py, blind to the classifier")


def evaluate(coded: pd.DataFrame) -> None:
    """Compare the hand codes (yes/no) with the classifier: yes = retrospective or prospective."""
    from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

    hand = pd.read_csv(SAMPLE, dtype=str)
    hand = hand[hand["hand_validated"].isin(["yes", "no"])]
    if hand.empty:
        print("hand-coded sample not filled in yet; skipping evaluation")
        return
    df = hand.merge(coded, on="submission_number")
    df["auto_validated"] = df["validation_level"].isin(["retrospective", "prospective"]).map({True: "yes", False: "no"})
    labels = ["yes", "no"]
    print(f"\nAgreement on {len(df)} hand-coded devices")
    print("  percent agreement:", f"{(df['hand_validated'] == df['auto_validated']).mean():.1%}")
    print("  Cohen's kappa:", round(cohen_kappa_score(df["hand_validated"], df["auto_validated"]), 3))
    print(pd.DataFrame(confusion_matrix(df["hand_validated"], df["auto_validated"], labels=labels),
                       index=["hand_yes", "hand_no"], columns=["auto_yes", "auto_no"]))
    print(classification_report(df["hand_validated"], df["auto_validated"], labels=labels, zero_division=0))

def main() -> None:
    with sqlite3.connect(DB_PATH) as con:
        devices = pd.read_sql("""SELECT submission_number, pathway, decision_date, device_name, company, panel,
                                        state_or_summ FROM v_device_base""", con)
    log = pd.read_csv(SCRAPE_LOG, dtype={"submission_number": str})
    devices = devices.merge(log[["submission_number", "url", "scanned"]], on="submission_number", how="left")

    rows = []
    for d in devices.itertuples(index=False):
        path = TXT_DIR / f"{d.submission_number.replace('/', '')}.txt"
        if not path.exists() or d.scanned is True or str(d.scanned) == "True":
            reason = "scanned" if path.exists() else ("statement" if d.state_or_summ == "Statement" else "no_pdf")
            rows.append({"submission_number": d.submission_number, "validation_level": "not_available",
                         "unavailable_reason": reason})
            continue
        rows.append({"submission_number": d.submission_number, "unavailable_reason": None,
                     **classify(path.read_text(), d.pathway)})
    coded = pd.DataFrame(rows)
    coded.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(coded["validation_level"].value_counts().to_string())
    print(coded["unavailable_reason"].value_counts().to_string())

    if not SAMPLE.exists():
        make_sample(coded, devices)
    evaluate(coded)


if __name__ == "__main__":
    main()
