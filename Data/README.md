# Data sources and preparation

## Primary source

The modelling input is [`raw/CBSL_USD_TT_Selling.csv`](raw/CBSL_USD_TT_Selling.csv), a local CSV labelled as the Central Bank of Sri Lanka (CBSL) USD Telegraphic Transfer (TT) daily buying and selling rate series. The official CBSL exchange-rate page is [Daily Buying and Selling Exchange Rate (2005–latest), Telegraphic Transfers](https://www.cbsl.gov.lk/en/node/146). CBSL describes its daily TT rates as based on 9:30 a.m. quotes from selected licensed banks.

The exact download URL, download date, and dataset-specific reuse or redistribution terms for this local CSV are not embedded in the file and were not independently verified. The source CSV is retained as supplied; this repository does not claim permission to redistribute it.

`DEXSLUS.csv` and `DEXSLUS.xlsx` are retained as FRED reference files only. They are not substituted for the selected CBSL TT selling-rate series.

## Supplied CSV snapshot

- Columns: `Currency`, `Date`, `Buy Rate`, `Sell Rate`.
- Date format in the source: `dd/mm/yyyy`.
- Snapshot contains 3,382 parsed data rows including one completely blank spacer row, and 3,381 usable USD observations.
- Observation span: 2010-05-07 through 2026-11-19.
- USD selling-rate range in this snapshot: 91.6728–377.4967 LKR per USD.
- SHA-256: `b76b443cee6db1bcaa6ffcb515490aafa8e8c2ef339fa411a328cf287af74969`.
- The raw file is preserved unchanged. The digest is emitted in `processed/data_quality_report.json` and checked before/after processing.

## Canonical data and processing decisions

`processed/cleaned_usd_lkr.csv` contains the canonical fields required by the project blueprint:

| Field | Meaning |
| --- | --- |
| `date` | Source quote date in ISO `YYYY-MM-DD` format. |
| `usd_lkr_rate` | USD TT **Selling Rate**, in LKR per USD, as a positive float. |
| `source` | `CBSL USD TT Selling Rate`. |

The loader validates the expected source headers, filters the USD currency, parses dates explicitly, and sorts observations ascending. Completely blank spacer rows, non-USD rows, missing dates, and missing selling rates are counted in the audit. Invalid nonblank dates/rates, nonpositive selling rates, and conflicting selling rates for the same date raise an error rather than being silently changed. Equal-selling-rate duplicate dates are reduced to one quote and counted. The buying rate is used only for audit; the selling rate is the model input.

Two pre-calibration records have the buying rate above the selling rate: 2010-07-19 (113.5755 versus 111.9651) and 2011-02-28 (109.948 versus 91.6728). They are retained and listed in the quality report. The source also contains a Saturday quote (2017-12-30). These records are not silently removed.

The large March 2022 movements are retained. The calibration return file shows +0.124872 on 2022-03-08 and +0.122608 on 2022-03-10 (log returns between one-day-apart quotes); no crisis-period observations are removed as outliers.

## Returns and period files

`processed/daily_log_returns.csv` calculates each return as `ln(current observed selling rate / previous observed selling rate)`. It uses consecutive valid quote observations. No weekend, holiday, or missing-date rates are forward-filled, interpolated, or synthesized. `calendar_gap_days` makes the elapsed calendar time explicit; a multi-day return is not divided by the gap length or relabelled as a one-calendar-day move.

- Calibration quote and return files cover 2019-01-01 to 2024-12-31. A calibration return is included only when both quote dates are in this period.
- Evaluation quote files cover 2025-01-01 through the available source end date in 2026. An evaluation return is selected by its current quote date; its first observation may use the immediately preceding 2024 quote as the return anchor. No future observation is used.
- The source currently has 1,445 calibration quotes and 457 evaluation quotes. The corresponding return counts are reported in `processed/data_quality_report.json` after the preparation run.

Generated files:

- `processed/cleaned_usd_lkr.csv`
- `processed/daily_log_returns.csv`
- `processed/calibration_quotes_2019_2024.csv`
- `processed/evaluation_quotes_2025_2026.csv`
- `processed/calibration_returns_2019_2024.csv`
- `processed/evaluation_returns_2025_2026.csv`
- `processed/data_quality_report.json`
- `../Figures/usd_lkr_selling_rate.png`

## Limitations

This is a selected-bank daily TT selling-rate series, not a continuous market mid-rate or intraday executable quote. Holidays and irregular source intervals remain in the data. The local CSV's exact upstream file lineage and data reuse terms are unresolved. The 2025–2026 evaluation period is a temporal holdout from the initial calibration window; this preparation step does not fit or validate a forecasting or risk model. The 2026 portion ends at the last date supplied in the snapshot.
