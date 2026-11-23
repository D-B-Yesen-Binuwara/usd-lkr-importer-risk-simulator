# Data sources and preparation

## Data register

| File | Description | Source | Licence / reuse | Date accessed |
|---|---|---|---|---|
| raw/CBSL_USD_TT_Selling.csv | Local CSV labelled as CBSL USD Telegraphic Transfer daily buying and selling rates. The model input is the selling rate. | The downloader reported using the [CBSL daily buy-and-sell page](https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates/daily-buy-and-sell-exchange-rates). The [CBSL exchange-rate explanation](https://www.cbsl.gov.lk/en/node/146) describes selected licensed-bank TT quotes at 9:30 AM. The exact export request and response have not been retained. | Not verified for this local file. Publicly viewable data do not establish permission to redistribute the raw or derived tables. | Downloader cannot recall the exact date. On 2026-11-23 the official pages were visible through search indexing, but direct fetch returned HTTP 403. |
| raw/DEXSLUS.csv and raw/DEXSLUS.xlsx | FRED DEXSLUS series retained for reference only; not used as the modelling input. | [FRED DEXSLUS series page](https://fred.stlouisfed.org/series/DEXSLUS). The page identifies the series as daily and not seasonally adjusted, with noon buying rates in New York City for cable transfers. | Review the FRED series notes before sharing. The exact provenance and reuse status of the local copies are not independently verified. | Local file retrieval date not recorded. FRED page checked 2026-11-21. |

The FRED definition differs from the selected CBSL TT selling-rate measure. FRED files are not substituted for the modelling series.

The downloader supplied the official CBSL daily buy-and-sell page as the site used, but cannot recall the download date. Search indexing exposes a CBSL daily buy/sell query form with a date range and USD selection. This narrows the **reported source page**; it does **not** establish the exact query, CSV response, export timestamp or file identity. The local CSV has no embedded source URL, download date or licence statement. Local filesystem or Git timestamps cannot establish the download date. No applicable exchange-rate CSV reuse licence was found in this review. Preserve an original browser download record, response, or CBSL permission statement if available; until then, cite the reported official page while labelling exact file lineage and redistribution permission unverified.

Search-indexed text from two official CBSL Daily Economic Indicators PDFs partially corroborates local values: the local 2026-02-23 USD row (buy 305.6182, sell 313.1434) matches the [23 February report](https://www.cbsl.gov.lk/sites/default/files/daily_economic_indicators_20260223_e.pdf), and the local 2026-06-02 row (buy 326.7561, sell 336.6059) matches the [2 June report](https://www.cbsl.gov.lk/sites/default/files/daily_economic_indicators_20260602_e.pdf). Direct fetches of those PDFs and the exchange-rate page returned HTTP 403 during this check. Two matching indexed rows support the rate definition but do **not** authenticate every CSV row, establish the export/download path or date, or grant redistribution rights.

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

The primary GBM calibration uses each return's actual interval, `calendar_gap_days/365` years. An invoice due in `H` calendar days has model horizon `H/365` years. This assumes Brownian variance accrues across closures; it does not fill unquoted dates. A separate 252-equal-quote-session comparison will test time-convention sensitivity with its own declared future quotation schedule. The 2019–2024 calibration returns include 357 gaps longer than one day, 57 longer than three, and a six-day maximum. [THEORY.md](../THEORY.md) records the fitted baseline and the 2019–2021, 2022–2024, full-period and extra 2022-excluded comparison plan. No importer payment-risk outputs have yet been calculated.

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
- data/processed/gbm_calibration_2019_2024.json (from `python -m src.calibration`; 1,444 training returns, 2,191 elapsed days, annual `alpha`, `sigma` and `mu`, source and prepared-file hashes)
- data/processed/markov_calibration_2019_2024.json (from `python -m src.markov`; 70th-percentile training-only stable/volatile rule, state and transition counts, row probabilities, residual RMS and input hashes)
- figures/usd_lkr_selling_rate.png

## Limitations

This is a selected-bank daily TT selling-rate series labelled as CBSL in the local CSV. The exact file lineage and source download date are not recorded. The local CSV's reuse terms are unresolved. The official CBSL page's indexed description was checked on 2026-11-23; direct access to the page returned HTTP 403, and the linked source export could not be inspected.

Holidays and irregular source intervals remain in the data. The 2025–2026 evaluation period is a temporal holdout excluded from the fitted 2019–2024 GBM and Markov parameters. For the Markov extension, `e_i=(r_i-alpha*delta_years_i)/sqrt(delta_years_i)` and a training-only linear-interpolated 70th percentile of `|e_i|` separates stable (ties included) from volatile. There are 1,011 stable and 433 volatile returns. Its matrix counts successive published-quote increments, not calendar-day transitions. The weekday future-quote helper is a proxy that ignores holidays and settlement timing. These are historical model estimates, not validation of forecasting accuracy. The 2026 portion ends at the last date supplied in the snapshot.
