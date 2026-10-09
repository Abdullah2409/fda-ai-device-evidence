# Codebook: reported clinical validation (yes/no)

Used for hand-coding `hand_coded_sample.csv` and as the target for the classifier in
`src/05_code_validation.py`. The definition follows the analysis plan: a device is
**validated** if its summary reports a study on patients or patient data.

Code with `coding_tool.html` (build it with `.venv/bin/python src/make_coding_tool.py`).
It shows the sponsor's summary opened at the testing section and never shows the
classifier's output. Do not open `data/interim/validation_coding.csv` while coding.

## The question

**Does this summary report testing of the device on patients or patient data, with a result?**

1. Tested on **real patient data** (scans, signals, records, or enrolled patients)? If not: **no**.
2. Reports a **result** (sensitivity, specificity, AUC, accuracy, Dice, agreement, or met
   predefined acceptance criteria)? If not: **no**.
3. Is the result from **this** submission? If yes: **yes**.

## Borderline cases

| Situation | Code |
|---|---|
| Bench, phantom, simulated or synthetic data, software verification and validation only | no |
| Clinicians judged "sample clinical images" acceptable, with no endpoint or criteria | no |
| Clinicians scored image quality on a predefined scale with acceptance criteria | yes |
| "Clinical performance unchanged from K######", with no new data here | no (add a note) |
| "No clinical testing was required", but algorithm accuracy on patient scans is reported | yes |
| A reader study (clinicians read cases with and without the device) | yes |
| Can't decide within about a minute | unsure, then resolve at the end with the full PDF |

## Columns

- `hand_validated`: yes / no / unsure (resolve every unsure before analysis)
- `notes`: optional; a few quoted words for borderline cases
