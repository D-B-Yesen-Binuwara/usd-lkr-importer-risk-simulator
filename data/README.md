# Data sources and preparation

## Data register

| File | Description | Source | Licence / reuse | Date accessed |
|---|---|---|---|---|
| raw/CBSL_USD_TT_Selling.csv | Local CSV labelled as CBSL USD Telegraphic Transfer daily buying and selling rates. The model input is the selling rate. | Official CBSL reference page: [Daily Buying and Selling Exchange Rate (2005–latest), Telegraphic Transfers](https://www.cbsl.gov.lk/en/node/146). The local file's exact download URL and upstream lineage have not been matched to a specific CBSL export. | Not verified for this local file. Do not assume redistribution rights. | Local download/access date not recorded. The CBSL page returned HTTP 403 during this repository review on 2026-11-21. |
| raw/DEXSLUS.csv and raw/DEXSLUS.xlsx | FRED DEXSLUS series retained for reference only; not used as the modelling input. | [FRED DEXSLUS series page](https://fred.stlouisfed.org/series/DEXSLUS). The page identifies the series as daily and not seasonally adjusted, with noon buying rates in New York City for cable transfers. | Review the FRED series notes before sharing. The exact provenance and reuse status of the local copies are not independently verified. | Local file retrieval date not recorded. FRED page checked 2026-11-21. |

The FRED definition differs from the selected CBSL TT selling-rate measure. FRED files are not substituted for the modelling series.

## Supplied CBSL CSV snapshot

- Source columns: Currency, Date, Buy Rate, Sell Rate.
- Source date format: dd/mm/yyyy.
- The snapshot has 3,382 parsed data rows, including one completely blank spacer row, and 3,381 usable USD observations.
- Observation span: 2010-05-07 through 2026-11-19.
- USD selling-rate range in this snapshot: 91.6728–377.4967 LKR per USD.
- SHA-256: b76b443cee6db1bcaa6ffcb515490aafa8e8c2ef339fa411a328cf287af74969.
- The raw file is retained unchanged. Its digest is emitted in data/processed/data_quality_report.json and checked before and after preparation.

## Canonical data and processing decisions

data/processed/cleaned_usd_lkr.csv contains the canonical fields:

| Field | Meaning |
|---|---|
| date | Source quote date in ISO YYYY-MM-DD format. |
| usd_lkr_rate | Positive USD TT selling rate in LKR per USD. |
| source | CBSL USD TT Selling Rate. |

The loader validates the expected source headers, filters USD, parses dates explicitly and sorts observations ascending. Completely blank spacer rows, non-USD rows, missing dates and missing selling rates are counted in the audit. Invalid nonblank dates/rates, nonpositive selling rates and conflicting selling rates for the same date raise an error rather than being silently changed. Equal-selling-rate duplicate dates are reduced to one quote and counted. The buying rate is used for audit only.

Two pre-calibration records have the buying rate above the selling rate: 2010-07-19 (113.5755 versus 111.9651) and 2011-02-28 (109.948 versus 91.6728). They are retained and listed in the quality report. The source also contains a Saturday quote (2017-12-30). These records are not silently removed.

The large March 2022 movements are retained. The calibration-return file shows +0.124872 on 2022-03-08 and +0.122608 on 2022-03-10. No crisis-period observations are removed as outliers.

## Returns and period files

data/processed/daily_log_returns.csv calculates each return as ln(current observed selling rate / previous observed selling rate). It uses consecutive valid quote observations. calendar_gap_days makes elapsed calendar time explicit; a multi-day return is not divided by the gap length or relabelled as a one-calendar-day move.

- Calibration quote and return files cover 2019-01-01 to 2024-12-31. A calibration return is included only when both quote dates are in this period.
- Evaluation quote files cover 2025-01-01 through the available source end date in 2026. An evaluation return is selected by its current quote date; the first observation may use the immediately preceding 2024 quote as the return anchor. No future observation is used.
- The source currently has 1,445 calibration quotes and 457 evaluation quotes. Return counts are 1,444 and 457, respectively, in the recorded Stage 2 preparation run.

Generated files:

- data/processed/cleaned_usd_lkr.csv
- data/processed/daily_log_returns.csv
- data/processed/calibration_quotes_2019_2024.csv
- data/processed/evaluation_quotes_2025_2026.csv
- data/processed/calibration_returns_2019_2024.csv
- data/processed/evaluation_returns_2025_2026.csv
- data/processed/data_quality_report.json
- figures/usd_lkr_selling_rate.png

## Limitations

This is a selected-bank daily TT selling-rate series labelled as CBSL in the local CSV. The exact file lineage and source download date are not recorded. The local CSV's reuse terms are unresolved. The official CBSL page could not be fetched during this review, so its contents were not independently rechecked here.

Holidays and irregular source intervals remain in the data. The 2025–2026 evaluation period is a temporal holdout from the initial calibration window; this preparation step does not fit or validate a forecasting or risk model. The 2026 portion ends at the last date supplied in the snapshot.
