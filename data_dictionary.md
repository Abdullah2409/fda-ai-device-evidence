# Data dictionary: `data/processed/devices.csv`

One row per device on the FDA AI-Enabled Medical Device List (1,614 rows). Built by
`src/06_merge.py` from the SQLite view `v_devices` (`src/02_build_db.py`), the validation coding
(`src/05_code_validation.py`, `src/05b_llm_batches.py`) and the company table. Data pulled
2026-10-05; see `data/raw/MANIFEST.csv`.

Binary flags are 1 = yes, 0 = no. Dates are `YYYY-MM-DD`. Read `regulation_number` as a string
(`pd.read_csv(..., dtype={"regulation_number": str})`), or trailing zeros are lost.

**Model sample:** rows with `in_model_sample == 1` (1,521 devices) are used in all models.

## Device and authorization

| Column | Type | Description | Source |
|---|---|---|---|
| `submission_number` | string | FDA submission number: `K######` (510(k)), `DEN######` (De Novo), `P######` or `P######/S###` (PMA or PMA supplement). Unique key. | AI list |
| `pathway` | string | `510(k)`, `De Novo` or `PMA`, from the submission number prefix | derived |
| `decision_date` | date | Date of FDA's final decision | AI list |
| `decision_year` | int | Year of `decision_date` | derived |
| `device_name` | string | Device name as listed | AI list |
| `company` | string | Manufacturer as listed (spelling varies across rows) | AI list |
| `panel` | string | Lead review panel (18 values, e.g. Radiology, Cardiovascular) | AI list |
| `specialty` | string | `panel` grouped: Radiology, Cardiovascular, Neurology, Other | derived |
| `product_code` | string | FDA three-letter product code | AI list |
| `risk_class` | string | Device class `1`, `2`, `3`, or `U` (unclassified) | openFDA classification, by product code |
| `regulation_number` | string | 21 CFR regulation number (missing for 12 unclassified codes) | openFDA classification |
| `date_received` | date | Date FDA received the submission | `pmn96cur` (510(k), De Novo); openFDA PMA |
| `review_days` | int | `decision_date` minus `date_received` | derived |
| `third_party` | string | `Y` if a third party reviewed the 510(k); missing for PMA | `pmn96cur` |
| `years_on_market` | float | Years from `decision_date` to the pull date (2026-10-05) | derived |
| `pma_supplement` | 0/1 | 1 if the device is a PMA supplement. Its recalls and adverse events are recorded under the base PMA and so cover the whole product line | derived |

## Company

| Column | Type | Description |
|---|---|---|
| `parent` | string | Parent company used for the public-company flag; blank if no match |
| `listing` | string | Where the parent is listed, e.g. `SEC:PHG`, `XETRA:SHL`, `KRX:005930`; blank if none |
| `public_company` | 0/1 | Company or its parent is publicly traded on any exchange, as of the pull date |
| `sec_listed` | 0/1 | Company or its parent is an SEC registrant (US exchange or ADR) |

Matching method (`src/06_merge.py`): names are normalized (lowercase, punctuation and legal suffixes
removed) and matched exactly to SEC `company_tickers.json`; subsidiaries, acquired firms and
foreign-listed parents come from `validation/company_parents.csv`. Every match is listed in
`data/interim/company_matches.csv`.

## Clinical validation (main predictor)

| Column | Type | Description |
|---|---|---|
| `validated` | 0/1, missing | **Main predictor.** 1 if the device summary reports testing on patients or patient data with a result (rule in `validation/codebook.md`), coded by Claude subagents. Missing (93) where no usable summary text existed. Agreement with blind hand-coding of 100 devices: 90.4%, κ = 0.80 |
| `validation_quote` | string | Verbatim text from the summary supporting the LLM's decision |
| `validated_regex` | 0/1, missing | Same question answered by the keyword classifier (`src/05_code_validation.py`); comparison only (κ = 0.61 against hand-coding) |
| `validation_level` | string | Keyword classifier's three-level output: `none`, `retrospective`, `prospective`, `not_available` |

## Recalls (outcome 1)

Only recalls initiated on or after `decision_date` count. Recall records link to devices through
their `k_numbers` / `pma_numbers` fields.

| Column | Type | Description |
|---|---|---|
| `recalled` | 0/1 | **Outcome.** 1 if any recall event began on or after authorization |
| `n_recall_events` | int | Number of distinct recall events (`res_event_number`) after authorization |
| `first_recall_date` | date | Start date of the first recall after authorization |
| `most_severe_recall_class` | 1/2/3 | Most severe FDA recall class (1 = most serious), from enforcement reports; missing if not recalled or no matching enforcement record |
| `days_to_first_recall` | float | `first_recall_date` minus `decision_date` |
| `days_observed` | float | Follow-up for the Cox model: days to first recall if recalled, otherwise days to the pull date |
| `n_recalls_before_decision` | int | Recall events dated before authorization (older models listing this 510(k)); excluded from `recalled` |

## Adverse events (outcome 2)

From MAUDE reports whose `pma_pmn_number` equals the device's submission number (base number for PMA).

| Column | Type | Description |
|---|---|---|
| `ae_reports` | int | **Outcome.** Number of MAUDE reports |
| `any_ae` | 0/1 | 1 if `ae_reports > 0` |
| `ae_deaths`, `ae_injuries`, `ae_malfunctions` | int | Reports by event type. Missing for one PMA supplement where only the total was downloadable |
| `ae_total_only` | 0/1 | 1 if MAUDE held too many reports to download (over 5,000; one device), so only the total is known |

## Analysis sample

| Column | Type | Description |
|---|---|---|
| `in_model_sample` | 0/1 | 1 if not a PMA supplement and `validated` is known. 1,521 devices: 103 recalled, 105 with adverse event reports |
