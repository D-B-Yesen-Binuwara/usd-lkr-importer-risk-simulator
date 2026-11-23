# USD/LKR Importer Risk Simulator

A Python project that prepares Sri Lankan rupee per US dollar (USD/LKR) quotes for a future importer payment-risk simulation.

**Current scope:** the repository contains the Stage 2 data-preparation pipeline, processed quote/return files, a data-quality report and automated validation tests. A baseline calibration convention and historical-window comparisons are documented in [THEORY.md](THEORY.md). The GBM, Monte Carlo, Markov-chain and interactive simulation components have not been implemented. The docs/ page is a status page, not a live simulator.

[Project page source](docs/index.html) (GitHub Pages deployment is not verified) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D-B-Yesen-Binuwara/usd-lkr-importer-risk-simulator/blob/main/notebook/usd-lkr-importer-risk-simulator.ipynb)

![CBSL USD TT selling-rate snapshot](figures/usd_lkr_selling_rate.png)

## The problem

Sri Lankan importers with USD invoices need to plan for the LKR cost of future payments. The cost rises when the LKR-per-USD rate rises. A risk simulator can later explore that uncertainty, but its result depends on the rate definition, data quality and time basis used for calibration.

This repository currently prepares the historical input series and holdout window. It does not yet estimate future exchange rates or importer costs.

## Quick start

Use Python 3.10 or later.

    python -m venv .venv

Activate the environment, then install the project requirements.

    # Windows PowerShell
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

    # macOS / Linux
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Run the data-preparation pipeline from the repository root:

    python -m src.prepare_data

The pipeline reads data/raw/CBSL_USD_TT_Selling.csv, writes outputs to data/processed/, and refreshes figures/usd_lkr_selling_rate.png. The raw input is checked for changes and is not rewritten.

Run the automated checks:

    python -m pytest -q

Open the prepared-data notebook locally:

The notebook interface is optional and is not listed in requirements.txt. If needed, install it with python -m pip install notebook.

    jupyter notebook notebook/usd-lkr-importer-risk-simulator.ipynb

In Colab, the repository must be accessible to Colab. Install dependencies and run the notebook after cloning the repository.

## Using the project

There is no web-demo control panel yet. The current workflow is the command-line data-preparation pipeline and the notebook that inspects its outputs. The pipeline writes a quality report with source-file hash, exclusions, date gaps and period counts. Full setup details are in [USER_GUIDE.md](USER_GUIDE.md).

## Model and mathematics

For consecutive observed quote rows, the prepared log return is

\[
r_i = \ln\!\left(\frac{S_i}{S_{i-1}}\right),
\]

where \(S_i\) is the USD TT selling rate in LKR per USD. The processed file also records the calendar days between quote dates. Missing calendar dates are not interpolated, and preparation does not convert returns to a per-day rate. The planned **primary** GBM uses actual elapsed gaps (`calendar_gap_days/365` years) and an invoice horizon of `H/365` for `H` calendar days. An equal-quote-step, 252-session convention is a separate sensitivity comparison. See [THEORY.md](THEORY.md). No GBM coefficients have yet been estimated.

## Verification

The Stage 2 suite was rerun in the local Windows virtual environment on 2026-11-23 after correcting a `Data`/`data` test-path mismatch: **9 passed in 1.38s** (`.venv\Scripts\python.exe -m pytest -q`). The checks cover parsing, cleaning rules, return construction, period boundaries and output creation. A Linux/Colab test run has not yet been performed, and passing code tests do not establish source authenticity or GBM suitability.

## Concepts applied

The current work uses time-series data cleaning, source-quality auditing, log-return transformation, chronological calibration/evaluation splits and automated software checks. Stochastic simulation and parameter estimation are future work.

## Results

The supplied snapshot has 3,381 usable USD quotes from 2010-05-07 through 2026-11-19. The initial calibration window contains 1,445 quotes and 1,444 in-window returns. The available evaluation window contains 457 quotes and 457 returns; its first return is anchored to the preceding observed quote. The 2026 window is partial in this snapshot.

The rate plot above is descriptive. No forecasting or risk-model result has been produced.

## Limitations and future work

- The downloader identified the [CBSL daily buy-and-sell page](https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates/daily-buy-and-sell-exchange-rates) as the source site, but the exact export request, download date and reuse terms are not recorded. Confirm redistribution rights before publishing the data.
- The series is a daily TT selling-rate series, not an intraday executable quote or continuous market mid-rate.
- Quote intervals are irregular. The primary calendar-time model will assume variance accrues over weekends and holidays in proportion to elapsed days; the 252-session sensitivity comparison will assess this choice. Neither convention is established as empirically correct by the data-preparation tests.
- The evaluation period ends at the source snapshot's last date and is therefore incomplete for 2026.
- The core calibration comparison retains 2022 in the 2019–2024 baseline and contrasts 2019–2021 with 2022–2024. A separately labelled 2022-excluded sensitivity case is planned without rewriting the source series.

## Key references

1. Central Bank of Sri Lanka, “Daily Buy and Sell Exchange Rates.” [Online]. Available: [CBSL daily rates page](https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates/daily-buy-and-sell-exchange-rates). The downloader reports using this page; the local CSV's exact export and download date are not recorded. See the [CBSL rate definition](https://www.cbsl.gov.lk/en/node/146).
2. Board of Governors of the Federal Reserve System (US), “Sri Lankan Rupees to U.S. Dollar Spot Exchange Rate [DEXSLUS],” retrieved from FRED, Federal Reserve Bank of St. Louis. [Online]. Available: [FRED series page](https://fred.stlouisfed.org/series/DEXSLUS). The FRED files in this repository are reference only.

See [REFERENCES.md](REFERENCES.md) for source notes.

## AI assistance

AI tools assisted with planning, documentation and code drafting. Stage 2 includes automated checks; users should run them in their own environment.

## Licence and credits

The MIT licence in [LICENSE](LICENSE) applies to the project's code and documentation only. The data has separate, unresolved reuse terms; see [data/README.md](data/README.md) before sharing it.
