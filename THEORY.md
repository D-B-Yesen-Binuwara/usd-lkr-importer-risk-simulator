# Theory and mathematics

## 1. Problem formulation

The application examines uncertainty in the LKR cost of hypothetical future USD-denominated importer payments. This repository prepares historical quotes, fits a calendar-time GBM and a bounded two-state quote-step Markov volatility extension, simulates rates, and calculates importer payment-risk metrics under a declared scenario.

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

The importer has a calendar due date. The **primary** GBM uses each observed return's actual elapsed interval, \(\Delta t_i=\texttt{calendar\_gap\_days}_i/365\) years. A future invoice due in \(H\) calendar days has horizon \(T=H/365\) years. No weekend or holiday quotes are invented. This choice makes the historical calibration clock and the invoice horizon consistent; it assumes GBM variance accumulates in proportion to elapsed calendar time, including closures. That assumption is a modelling simplification, not a property established by CBSL data.

Let \(\alpha=\mu-\tfrac12\sigma^2\) be annual mean log-rate change. Under a constant-parameter GBM with independent Gaussian increments,

\[
r_i\sim N(\alpha\Delta t_i,\,\sigma^2\Delta t_i),\qquad
\hat\alpha=\frac{\sum_i r_i}{\sum_i\Delta t_i},\qquad
\hat\sigma^2_{\mathrm{ML}}=\frac{1}{n}\sum_i\frac{(r_i-\hat\alpha\Delta t_i)^2}{\Delta t_i},\qquad
\hat\mu=\hat\alpha+\tfrac12\hat\sigma^2_{\mathrm{ML}}.
\]

The variance divisor \(n\) identifies the maximum-likelihood estimate; an \(n-1\) adjustment would need a different label. `src/calibration.py` implements \(\hat\sigma=\sqrt{\max(\hat\sigma^2_{\mathrm{ML}},0)}\), checks positive intervals and requires a positive fitted volatility in its baseline report. The mean of \(r_i/d_i\) and `std(r)*sqrt(252)` are not the primary variable-gap estimator.

The 2019–2024 baseline fit uses 1,444 in-window returns spanning 2,191 calendar days (6.002740 model years). The report in `data/processed/gbm_calibration_2019_2024.json` records the source/prepared-file SHA-256 hashes and these **annual** estimates:

| Parameter | Estimate | Units and meaning |
|---|---:|---|
| \(\hat\alpha\) | 0.079196960112 | Mean log-rate change per year. |
| \(\hat\sigma\) | 0.120814246748 | Rate volatility per square-root year. |
| \(\hat\mu\) | 0.086495001221 | Drift of the rate level per year, \(\hat\alpha+\hat\sigma^2/2\). |

These describe the fitted historical constant-coefficient model, not a reliable prediction of future USD/LKR. The 2022 movements are retained and can strongly affect the fitted drift and volatility. The 2025–2026 evaluation rows are excluded.

The 2019–2024 calibration file has 1,444 returns: 1,087 with a one-day gap, 40 with two, 260 with three, 44 with four, 10 with five, and 3 with six. Thus 357 span more than one calendar day. A descriptive check found sample SD 0.007050 for one-day returns and 0.004459 for three-day returns. Excluding returns touching 2022 gives 0.004213 and 0.004596, respectively. These groups do not exhibit a simple \(\sqrt{3}\) dispersion ratio, but they also represent different market conditions. This warns that uniform calendar-day variance scaling may fit poorly; it does not justify changing the chosen invoice clock after seeing model outputs.

### Quote-session clock: predefined sensitivity comparison

Treat every adjacent published quote pair as one equal step and use 252 steps per model year **only for a separate sensitivity run**. With \(a=\operatorname{mean}(r_i)\) and sample SD \(s_r\) in the selected training window, use \(\sigma_{252}=s_r\sqrt{252}\) and \(\mu_{252}=252a+\tfrac12\sigma_{252}^2\). This separate 252-session GBM comparison is not yet implemented. The implemented Markov chain has one transition per assumed published-quote increment, while its rate increments retain actual assumed calendar gaps. Its training labels use calendar-gap-standardized residuals; see Section 5.

For a calendar due date, specify an assumed future quotation schedule, count its quote steps \(n_{\mathrm{session}}\), and set the alternative horizon to \(T_{252}=n_{\mathrm{session}}/252\) model years. A Monday–Friday proxy can be used if labelled, with unverified holidays and settlement conventions disclosed. A 90-calendar-day invoice is not automatically 90 quote steps. Keep this schedule separate from the calendar-time primary; do not mix \(\sigma_{252}\) with \(\sqrt{H/365}\) without an explicit conversion rule. Compare the resulting importer VaR and ES later, using the same invoice, due date, starting quote and simulation settings.

The planned geometric Brownian motion model is written as

\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,
\]

where \(\mu\) is annual level drift, \(\sigma\) is annual volatility (per square-root year), and \(W_t\) is Brownian motion in elapsed years. The GBM core fits these coefficients and simulates rates; `src/risk.py` converts terminal rates to invoice costs and upper-tail risk.

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

The data pipeline parses, validates and sorts USD quotes, then calculates adjacent-observation log returns and day gaps. See `src/data_loader.py`, `src/preprocess.py` and `src/prepare_data.py`. `src/calibration.py` fits the variable-gap Gaussian likelihood to training rows whose **two** quote dates lie in the requested window. It validates rate/return arithmetic, positive integer gaps and the CBSL TT selling-rate source before fitting.

`src/gbm.py` uses the exact conditional transition, for any positive elapsed interval \(\delta\) in years and independent \(Z\sim N(0,1)\):

\[
S_{t+\delta}=S_t\exp\left(\alpha\delta+\sigma\sqrt{\delta}Z\right).
\]

Terminal-only sampling uses \(\delta=H/365\) directly. A path chart can use daily or caller-supplied positive intervals that sum to the same \(H\) calendar days. Local NumPy generators with explicit seeds give repeatable draws; the exact transition avoids Euler time-step bias. The module also exposes analytical mean, variance and rate quantiles for verification. Payment costs, VaR and ES are computed separately from terminal rates in `src/risk.py`.

For \(T=H/365\), independent analytical benchmarks are

\[
E[S_T]=S_0e^{\mu T},\qquad
\operatorname{Var}(S_T)=S_0^2e^{2\mu T}\bigl(e^{\sigma^2T}-1\bigr),\qquad
q_p(S_T)=S_0\exp\bigl(\alpha T+\sigma\sqrt{T}\,\Phi^{-1}(p)\bigr).
\]

Agreement with these identities verifies the sampler under its assumptions; it does not establish forecasting skill on real exchange rates.

### Bounded two-state Markov volatility extension

`src/markov.py` uses the same 2019–2024 training rows and fitted annual GBM log drift \(\hat\alpha\). For each observed quote-to-quote return, let

\[
e_i=\frac{r_i-\hat\alpha\Delta t_i}{\sqrt{\Delta t_i}},\qquad
\tau=Q_{0.70}(|e_i|),\qquad
X_i=\begin{cases}0\;(\text{stable}),&|e_i|\le\tau,\\1\;(\text{volatile}),&|e_i|>\tau.\end{cases}
\]

The 70th percentile is a declared design choice, computed by NumPy's linear quantile rule on training data only. Ties at the threshold are stable. Standardizing by \(\sqrt{\Delta t_i}\) reduces the automatic effect of longer quote gaps on a magnitude threshold. The residual has annual-volatility units; labels are a heuristic classification of realized returns, not observed or inferred hidden economic regimes. A class with fewer than ten returns fails rather than yielding a fragile fitted state.

Count successive classified published-quote increments \(n_{ab}\), then set \(P_{ab}=n_{ab}/\sum_b n_{ab}\). Both outgoing rows must be observed, contain probabilities in \([0,1]\), and sum to one within \(10^{-12}\); no smoothing row is invented. The state-specific conditional residual RMS is \(\hat\sigma_k=\sqrt{\operatorname{mean}_{i:X_i=k}(e_i^2)}\), using the common \(\hat\alpha\), not a separate drift for each state.

| Training result | Stable (0) | Volatile (1) |
|---|---:|---:|
| Return labels | 1,011 | 433 |
| Residual mean, per square-root year | -0.003817 | 0.015948 |
| Residual RMS \(\hat\sigma_k\), per square-root year | 0.010525 | 0.220040 |

The training threshold is 0.026381719589 per square-root year. Transition counts are \(\left[\begin{smallmatrix}823&187\\187&246\end{smallmatrix}\right]\), so the quote-step matrix in stable/volatile order is

\[
P=\begin{pmatrix}0.814851485149&0.185148514851\\0.431870669746&0.568129330254\end{pmatrix}.
\]

The last training return is labelled stable. For future paths, provide assumed quotation-day offsets measured from the reference date. At each assumed quote, sample the **next** state from the current row of \(P\), then draw an independent normal shock and apply \(\hat\alpha\delta+\hat\sigma_{\mathrm{new}}\sqrt{\delta}Z\), where \(\delta\) is that quote gap divided by 365. If maturity follows the last assumed quote, the residual interval uses the current state without an extra transition. The `weekday_quote_offsets` helper is only a Monday–Friday proxy: it includes unverified weekday holidays and does not model settlement timing. The 90-day smoke check used 64 assumed weekday quotes; its output was checked for positive finite rates, not interpreted as a forecast.

This uses one common **log** drift across states, so the level drift varies with \(\hat\sigma_k^2/2\). The volatile RMS is much larger because the retained 2022 jumps enter that class. Conditional normality, persistence and the future quotation schedule are modelling assumptions; the fitted matrix is not evidence of a crisis predictor.

### Hypothetical importer payment risk

Stage 6 fixes a hypothetical USD invoice \(Q=100{,}000\), the 2024-12-31 training-end selling rate \(S_0=297.0162\) LKR/USD, and a 90-calendar-day due date of 2025-03-31. The reference cost is \(C_0=QS_0=29{,}701{,}620\) LKR. An illustrative budget is 5% higher, \(B=31{,}186{,}701\) LKR. These values are assumptions, not a real importer's bill. The end-2024 anchor makes this a retrospective hypothetical scenario; the 2025–2026 observations are not used to choose its inputs or fit its models.

For each simulated terminal rate \(S_T^{(j)}\), `src/risk.py` calculates

\[
C_T^{(j)}=Q S_T^{(j)},\qquad L^{(j)}=C_T^{(j)}-C_0.
\]

The signed extra cost \(L\) can be negative, meaning a saving relative to the reference conversion. The budget exceedance estimate is \(N^{-1}\sum_j 1\{C_T^{(j)}>B\}\), with a strict inequality. At level \(p=0.95\), importer VaR is the empirical **upper** left quantile of \(L\), and ES is the average of the worst \((1-p)N\) observations, fractionally weighting the boundary observation when needed. This implements the quantile-integral upper-tail convention even when outcomes tie; it does not use the lower 5% of FX returns or clamp losses at zero. The 5th/95th *cost* percentiles use NumPy linear interpolation and are separate from VaR of extra cost.

`src/run_risk.py` uses eight distinct seeds, 42–49, with 10,000 independent paths per seed and model (80,000 pooled paths per model). The GBM draws terminal rates directly over 90/365 years. The Markov draws quote-step paths over an explicit 64-date Monday–Friday proxy; holidays and settlement timing are unverified. Both models share the invoice, starting rate, due date, budget, path count and seed list, but the RNG algorithms do not make their individual paths paired. The report includes each replicate's metrics and pooled metrics.

| Model | Mean cost, LKR | 95% VaR of extra cost, LKR | 95% ES of extra cost, LKR | Pr(cost > budget) |
|---|---:|---:|---:|---:|
| GBM | 30,339,206 | 3,731,194 | 4,582,741 | 0.3129 |
| Markov | 30,345,423 | 3,679,169 | 4,609,172 | 0.3051 |

The pooled mean-cost 95% Monte Carlo intervals are approximately LKR [30,326,560, 30,351,852] for GBM and [30,332,957, 30,357,890] for Markov. They use \(\bar C\pm z_{0.975}s_C/\sqrt N\), conditional on independent finite-variance paths and fixed fitted parameters. The saved report also gives the **between-replicate** sample SD and min/max for VaR and ES; these are observed run-to-run variation, not tail-risk confidence intervals. GBM's simulated mean cost differs from its analytical expected cost by -0.42 pooled Monte Carlo standard errors in this run.

The numerical intervals exclude uncertainty in the historical calibration window, the 2022 regime change, the Markov state rule, time convention and source-data lineage. Fees and importer-specific bank spreads are excluded. Neither model's results are validated predictions of actual future payment costs.

## 6. Verification

The full local Windows suite was rerun after Stage 6 code and test changes on 2026-11-23: **59 passed in 2.38s** using `.venv\Scripts\python.exe -m pytest -q`. This includes nine data-preparation tests, 13 GBM tests, 22 Markov tests and 15 importer-risk tests. It checks code and mathematical/probability identities, not source authenticity or predictive adequacy. A Linux/Colab run has not yet been performed.

| Check | Expected | Recorded result |
|---|---|---|
| Full automated test suite | All included tests pass | 59 passed in 2.38s on Windows, 2026-11-23 |
| Variable-gap likelihood | Closed form agrees with independent numerical optimization | Passed in the above suite |
| Synthetic irregular series | Approximately recovers known coefficients | Passed in the above suite |
| Exact GBM and limiting cases | Seed, zero volatility, horizon partition, moments and quantiles | Passed in the above suite |
| Markov state and matrix fit | Independent counts, row sums, zero-row rejection, training-only holdout isolation | Passed in the above suite |
| Quote-step path simulation | New-state ordering, explicit day gaps, maturity remainder, seeded replay and long-run synthetic transition frequencies | Passed in the above suite |
| Importer risk arithmetic | Hand-calculated costs, signed savings, quantile-integral ES with ties/fractional tails, invoice scaling and budget distinction | Passed in the above suite |
| Reproducible risk report | Same seeds give same outputs; saved report matches fresh run; GBM mean agrees with analytical mean within five MC SE | Passed in the above suite |

Run python -m pytest -q to repeat these checks in the current environment.

The actual baseline coefficients were also recalculated independently using Python's standard-library `csv`, `math.fsum` and `math.sqrt`; the three values agreed with the saved report to absolute tolerance \(10^{-12}\). The test that fits both the calibration-only file and the complete return history obtained identical baseline coefficients, checking that holdout rows are filtered. The 252-session comparison still needs its own coherent future quotation schedule and later risk-output comparison.

## 7. Concepts applied

| Concept | Use in this project |
|---|---|
| Time-series cleaning | Parse, filter, sort and audit the supplied daily quotes. |
| Log returns | Express changes as adjacent quote ratios for future model calibration. |
| Chronological split | Keep 2019–2024 for initial calibration and 2025–2026 for evaluation. |
| Model clock | Use elapsed calendar time for the primary GBM; compare a separate 252-session convention. |
| Automated validation | Check representative parsing, cleaning, return and output behavior. |
| Stochastic modelling | Fit an annual GBM and sample exact positive rate paths under a stated calendar clock. |
| Markov chains (DTMC) | Classify training returns into two volatility states, estimate a quote-step transition matrix and sample state-conditioned rate increments. |
| Monte Carlo integration and LLN | Average independent payment-cost paths, quantify sampling SE, and inspect run-to-run tail estimates. |

## 8. References

See [REFERENCES.md](REFERENCES.md).
