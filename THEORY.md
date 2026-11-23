# Theory and mathematics

## 1. Problem formulation

The intended application is to examine uncertainty in the LKR cost of future USD-denominated importer payments. This repository prepares historical quote data and records a calibration design. It does not yet estimate a stochastic model or simulate payments.

## 2. Data and assumptions

- The selected variable is the CBSL USD Telegraphic Transfer (TT) selling rate, measured in LKR per USD. The selling side corresponds to the exchange-rate side relevant when an importer buys USD to pay an invoice.
- The local CSV is labelled as a CBSL daily buying/selling series. Its exact download URL, download date and file lineage are not embedded in the file.
- Only valid USD quotes with positive selling rates are retained by the loader.
- The processing keeps adjacent available quote observations. It does not synthesize observations for weekends, holidays or other missing dates.
- The return data records the elapsed calendar days for each adjacent pair. It is not a consistent one-calendar-day sample.

These are data-preparation decisions. They do not establish that the observations are a complete or executable market price series.

## 3. Return calculation

Let \(S_i\) denote the observed USD TT selling rate on quote date \(t_i\). The pipeline computes

\[
r_i = \ln(S_i/S_{i-1}).
\]

The quantity is dimensionless and represents the continuously compounded change between two adjacent observations. `calendar_gap_days` stores \(t_i-t_{i-1}\) in calendar days. The data pipeline does not divide \(r_i\) by that interval or annualize it.

### Selected primary clock: elapsed calendar time

The importer has a calendar due date. The **primary** GBM will use each observed return's actual elapsed interval, \(\Delta t_i=\texttt{calendar\_gap\_days}_i/365\) years. A future invoice due in \(H\) calendar days has horizon \(T=H/365\) years. No weekend or holiday quotes are invented. This choice makes the historical calibration clock and the invoice horizon consistent; it assumes GBM variance accumulates in proportion to elapsed calendar time, including closures. That assumption is a modelling simplification, not a property established by CBSL data.

Let \(\alpha=\mu-\tfrac12\sigma^2\) be annual mean log-rate change. Under a constant-parameter GBM with independent Gaussian increments,

\[
r_i\sim N(\alpha\Delta t_i,\,\sigma^2\Delta t_i),\qquad
\hat\alpha=\frac{\sum_i r_i}{\sum_i\Delta t_i},\qquad
\hat\sigma^2_{\mathrm{ML}}=\frac{1}{n}\sum_i\frac{(r_i-\hat\alpha\Delta t_i)^2}{\Delta t_i},\qquad
\hat\mu=\hat\alpha+\tfrac12\hat\sigma^2_{\mathrm{ML}}.
\]

The variance divisor \(n\) identifies the maximum-likelihood estimate; an \(n-1\) adjustment would need a different label. Implement \(\hat\sigma=\sqrt{\max(\hat\sigma^2_{\mathrm{ML}},0)}\), checking for positive intervals and a usable positive volatility before stochastic simulation. These are **planned Stage 4 formulas**; no parameters are fitted here. The mean of \(r_i/d_i\) and `std(r)*sqrt(252)` are not the primary variable-gap estimator.

The 2019–2024 calibration file has 1,444 returns: 1,087 with a one-day gap, 40 with two, 260 with three, 44 with four, 10 with five, and 3 with six. Thus 357 span more than one calendar day. A descriptive check found sample SD 0.007050 for one-day returns and 0.004459 for three-day returns. Excluding returns touching 2022 gives 0.004213 and 0.004596, respectively. These groups do not exhibit a simple \(\sqrt{3}\) dispersion ratio, but they also represent different market conditions. This warns that uniform calendar-day variance scaling may fit poorly; it does not justify changing the chosen invoice clock after seeing model outputs.

### Quote-session clock: predefined sensitivity comparison

Treat every adjacent published quote pair as one equal step and use 252 steps per model year **only for a separate sensitivity run**. With \(a=\operatorname{mean}(r_i)\) and sample SD \(s_r\) in the selected training window, use \(\sigma_{252}=s_r\sqrt{252}\) and \(\mu_{252}=252a+\tfrac12\sigma_{252}^2\). One transition of the later two-state Markov chain means one published-quote increment. Its future rate increments still use actual assumed quotation gaps. For training-only regime classification, use calendar-gap-standardized residuals \(e_i=(r_i-\hat\alpha\Delta t_i)/\sqrt{\Delta t_i}\), then a training-only magnitude threshold. This prevents longer gaps from automatically appearing more volatile solely because they span more days. The Markov model is not implemented yet.

For a calendar due date, specify an assumed future quotation schedule, count its quote steps \(n_{\mathrm{session}}\), and set the alternative horizon to \(T_{252}=n_{\mathrm{session}}/252\) model years. A Monday–Friday proxy can be used if labelled, with unverified holidays and settlement conventions disclosed. A 90-calendar-day invoice is not automatically 90 quote steps. Keep this schedule separate from the calendar-time primary; do not mix \(\sigma_{252}\) with \(\sqrt{H/365}\) without an explicit conversion rule. Compare the resulting importer VaR and ES later, using the same invoice, due date, starting quote and simulation settings.

The planned geometric Brownian motion model is written as

\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
\]

where the primary \(\mu\) is annual level drift, \(\sigma\) is annual volatility (per square-root year), and \(W_t\) is Brownian motion in elapsed years. This equation is planned theory. The repository has not fitted \(\mu\) or \(\sigma\) or simulated paths.

## 4. Calibration periods and sensitivity design

The baseline uses only returns whose two quote dates fall from 2019-01-01 through 2024-12-31. The **Stage 3 core comparisons, fixed before GBM fitting or risk outputs**, are 2019–2021 (before the 2022 movements), 2022–2024 (crisis and post-crisis), and the full 2019–2024 baseline. These choices follow inspection of the training data; they should not be presented as chosen before any historical data were viewed. The 2025–2026 observations are a temporal evaluation snapshot and must not influence any fit, regime threshold, or choice of the preferred window. The values below were independently recalculated from `data/processed/calibration_returns_2019_2024.csv` on 2026-11-23. Means and sample SDs are **descriptive per-observed-quote statistics**, not calendar-time GBM estimates or prediction results.

| Window | Selection: both quote dates in window | Returns | Sum of observed gaps, days | Mean log return per quote | Sample SD per quote |
|---|---|---:|---:|---:|---:|
| Baseline | 2019–2024 | 1,444 | 2,191 | 0.000329 | 0.006684 |
| Pre-2022 | 2019–2021 | 721 | 1,095 | 0.000132 | 0.003750 |
| Crisis/post-crisis | 2022–2024 | 722 | 1,093 | 0.000527 | 0.008678 |
| Additional excluding-2022 sensitivity | 2019–2021 or 2023–2024, separately | 1,204 | 1,824 | -0.000107 | 0.004604 |

The pre-2022 and 2022–2024 counts sum to 1,443 because the full baseline additionally contains the boundary return from 2021-12-31 to 2022-01-03 (zero in this snapshot). The excluding-2022 case is a filtered set of existing adjacent returns, **not** a price series with a fabricated 2021-to-2023 bridge. For diagnosis, 2022 alone has 238 in-year returns with sample SD 0.012593 per quote. The source's 2022-03-08 and 2022-03-10 log returns are about +0.124872 and +0.122608, each across one calendar day. Retain them in the baseline and crisis/post-crisis run; they may materially affect drift and volatility.

Fit each window with the same variable-gap formula, using only that window's eligible returns. Later, hold invoice amount, starting rate, actual due date, simulation count and random-seed policy fixed while comparing fitted parameters and importer risk. A different window is a sensitivity scenario, not proof that it predicts future FX better. Keep any 2025–2026 retrospective assessment frozen to information known at the end of 2024. A separate contemporary illustration may start at the latest supplied quote (2026-11-19), but its parameters remain trained on 2019–2024 and it must not be labelled a 2025–2026 out-of-sample forecast.

## 5. Numerical method

No stochastic simulation or numerical integration is implemented at this stage. The data pipeline parses source rows, validates the fields, sorts valid USD observations by date, calculates adjacent-observation log returns and records date gaps. See src/data_loader.py, src/preprocess.py and src/prepare_data.py.

## 6. Verification

The Stage 2 tests were rerun in the local Windows virtual environment on 2026-11-23 after fixing the test's `Data`/`data` path: 9 passed in 1.38s. They exercise code behavior using the local source snapshot and small synthetic examples. They do not independently verify the data provider or prove that GBM assumptions fit the exchange-rate series; a Linux/Colab run has not yet been performed.

| Check | Expected | Recorded result |
|---|---|---|
| Stage 2 automated test suite | All included tests pass | 9 passed in 1.38s, rerun 2026-11-23 on Windows |

Run python -m pytest -q to repeat these checks in the current environment.

The Stage 3 evidence above was checked by reading the processed calibration-return file and its date/gap columns. No Stage 3 model test or new GBM result exists. Before using a future model, rerun the Stage 2 suite, confirm positive gaps and window counts, exclude evaluation rows from every fit, and verify the variable-gap maximum-likelihood formula against an independent numerical likelihood calculation. Test equal-gap limiting cases, seed reproducibility, analytic GBM moments and the exact calendar horizon. The separate quote-session comparison needs its own coherent future quotation schedule.

## 7. Concepts applied

| Concept | Use in this project |
|---|---|
| Time-series cleaning | Parse, filter, sort and audit the supplied daily quotes. |
| Log returns | Express changes as adjacent quote ratios for future model calibration. |
| Chronological split | Keep 2019–2024 for initial calibration and 2025–2026 for evaluation. |
| Model clock | Use elapsed calendar time for the primary GBM; compare a separate 252-session convention. |
| Automated validation | Check representative parsing, cleaning, return and output behavior. |
| Stochastic modelling | Planned next-stage work; not implemented or verified. |

## 8. References

See [REFERENCES.md](REFERENCES.md).
