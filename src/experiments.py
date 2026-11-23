"""Frozen-input GBM comparisons and replicated Monte Carlo verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import DAYS_PER_YEAR, baseline_calibration_report, fit_calendar_gbm
from src.gbm import exact_gbm_step, theoretical_terminal_moments
from src.markov import weekday_quote_offsets
from src.risk import PaymentRiskMetrics, between_run_variability, payment_risk_metrics
from src.run_risk import (
    BUDGET_MARGIN, HORIZON_CALENDAR_DAYS, INVOICE_USD, N_PATHS_PER_RUN,
    REPLICATE_SEEDS, RISK_LEVEL, baseline_risk_report,
)

SESSIONS_PER_YEAR = 252
CONVERGENCE_COUNTS = (500, 2_000, 8_000, 32_000)
CONVERGENCE_SEEDS = tuple(range(1_000, 1_032))


@dataclass(frozen=True)
class GBMCase:
    key: str
    label: str
    alpha_per_year: float
    sigma_per_sqrt_year: float
    horizon_years: float
    training_returns: int
    training_elapsed_days: int
    clock: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibration_windows(training_returns: pd.DataFrame) -> dict[str, object]:
    """Fit the predeclared windows without creating a 2022-crossing return."""
    full = fit_calendar_gbm(training_returns, window_start="2019-01-01", window_end="2024-12-31")
    pre = fit_calendar_gbm(training_returns, window_start="2019-01-01", window_end="2021-12-31")
    post = fit_calendar_gbm(training_returns, window_start="2022-01-01", window_end="2024-12-31")
    dates = pd.to_datetime(training_returns["date"], errors="raise")
    prev = pd.to_datetime(training_returns["previous_date"], errors="raise")
    keep = (
        (dates.between("2019-01-01", "2021-12-31") & prev.between("2019-01-01", "2021-12-31"))
        | (dates.between("2023-01-01", "2024-12-31") & prev.between("2023-01-01", "2024-12-31"))
    )
    excluded = fit_calendar_gbm(
        training_returns.loc[keep], window_start="2019-01-01", window_end="2024-12-31"
    )
    return {"full": full, "pre_2022": pre, "post_2022": post, "exclude_2022": excluded}


def quote_session_fit(training_returns: pd.DataFrame, quote_count: int) -> dict[str, float | int]:
    """Separate 252-equal-quotation-step sensitivity, trained only on 2019-2024."""
    validated = fit_calendar_gbm(training_returns, window_start="2019-01-01", window_end="2024-12-31")
    if len(training_returns) != validated.n_returns:
        raise ValueError("252-session input must contain only the selected training returns")
    if quote_count <= 0:
        raise ValueError("quote_count must be positive")
    returns = training_returns["log_return"].to_numpy(dtype=float)
    alpha = float(np.mean(returns) * SESSIONS_PER_YEAR)
    sigma = float(np.std(returns, ddof=1) * sqrt(SESSIONS_PER_YEAR))
    return {
        "n_returns": validated.n_returns,
        "alpha_per_year": alpha,
        "sigma_per_sqrt_year": sigma,
        "mu_per_year": alpha + 0.5 * sigma**2,
        "quote_count": quote_count,
        "horizon_session_years": quote_count / SESSIONS_PER_YEAR,
    }


def case_metrics(
    case: GBMCase, *, initial_rate: float, invoice_usd: float, budget_lkr: float,
    seeds: tuple[int, ...], n_paths_per_run: int,
) -> dict[str, object]:
    """Use common normal draws across cases and independent seeds within a case."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct seeds are required")
    batches: list[np.ndarray] = []
    metrics: list[PaymentRiskMetrics] = []
    for seed in seeds:
        shocks = np.random.default_rng(seed).standard_normal(n_paths_per_run)
        rates = np.asarray(exact_gbm_step(
            initial_rate, case.alpha_per_year, case.sigma_per_sqrt_year,
            case.horizon_years, shocks,
        ))
        batches.append(rates)
        metrics.append(payment_risk_metrics(
            rates, invoice_usd=invoice_usd, reference_rate_lkr_per_usd=initial_rate,
            budget_lkr=budget_lkr, level=RISK_LEVEL,
        ))
    pooled = payment_risk_metrics(
        np.concatenate(batches), invoice_usd=invoice_usd,
        reference_rate_lkr_per_usd=initial_rate, budget_lkr=budget_lkr, level=RISK_LEVEL,
    )
    return {
        "parameters": case.to_dict(),
        "pooled_metrics": pooled.to_dict(),
        "between_run_variability": between_run_variability(metrics),
        "runs": [{"seed": seed, "metrics": metric.to_dict()} for seed, metric in zip(seeds, metrics, strict=True)],
    }


def convergence_study(
    *, initial_rate: float, alpha_per_year: float, sigma_per_sqrt_year: float,
    invoice_usd: float, horizon_days: int,
    counts: tuple[int, ...] = CONVERGENCE_COUNTS,
    seeds: tuple[int, ...] = CONVERGENCE_SEEDS,
) -> dict[str, object]:
    """Replicated RMSE of the GBM sample mean against its analytical mean."""
    if len(counts) < 3 or any(n < 2 for n in counts) or tuple(sorted(set(counts))) != counts:
        raise ValueError("counts must be at least three increasing sample sizes")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct replicate seeds are required")
    mean_rate, variance_rate = theoretical_terminal_moments(
        initial_rate, alpha_per_year, sigma_per_sqrt_year, horizon_days=horizon_days,
    )
    true_mean_cost = invoice_usd * mean_rate
    horizon_years = horizon_days / DAYS_PER_YEAR
    errors = {n: [] for n in counts}
    for seed in seeds:
        shocks = np.random.default_rng(seed).standard_normal(counts[-1])
        rates = np.asarray(exact_gbm_step(
            initial_rate, alpha_per_year, sigma_per_sqrt_year, horizon_years, shocks,
        ))
        costs = invoice_usd * rates
        for n in counts:
            errors[n].append(float(np.mean(costs[:n]) - true_mean_cost))
    points = []
    for n in counts:
        rmse = float(np.sqrt(np.mean(np.square(errors[n]))))
        points.append({
            "n_paths": n,
            "replicates": len(seeds),
            "rmse_mean_cost_lkr": rmse,
            "theoretical_mc_se_lkr": float(invoice_usd * sqrt(variance_rate / n)),
        })
    slope = float(np.polyfit(np.log(counts), np.log([p["rmse_mean_cost_lkr"] for p in points]), 1)[0])
    return {
        "analytic_mean_cost_lkr": true_mean_cost,
        "seed_list": seeds,
        "method": "independent replicate seeds; nested prefixes within each seed; RMSE against analytical GBM mean",
        "points": points,
        "log_log_rmse_slope": slope,
        "theoretical_slope": -0.5,
    }


def comparison_report(project_root: Path) -> dict[str, object]:
    """Reproduce Stage 7 comparisons with training-only fits and frozen exposure."""
    gbm_report = baseline_calibration_report(project_root)
    stage6 = baseline_risk_report(project_root)
    training = pd.read_csv(project_root / "data" / "processed" / "calibration_returns_2019_2024.csv")
    fits = calibration_windows(training)
    baseline = fits["full"]
    prior = gbm_report["calibration"]
    if baseline.to_dict() != prior:
        raise ValueError("full-window fit disagrees with the baseline report")
    scenario = stage6["scenario"]
    initial_rate = scenario["reference_rate_lkr_per_usd"]
    invoice = scenario["invoice_usd"]
    budget = scenario["budget_lkr"]
    seeds = tuple(scenario["replicate_seeds"])
    n_paths = scenario["n_paths_per_run"]
    horizon_days = scenario["horizon_calendar_days"]
    quote_offsets = weekday_quote_offsets(scenario["reference_date"], horizon_days)
    session = quote_session_fit(training, len(quote_offsets))
    calendar_years = horizon_days / DAYS_PER_YEAR

    cases = {
        "full": GBMCase("full", "2019-2024 baseline", baseline.alpha_per_year, baseline.sigma_per_sqrt_year, calendar_years, baseline.n_returns, baseline.total_elapsed_days, "calendar days / 365"),
        "pre_2022": GBMCase("pre_2022", "2019-2021", fits["pre_2022"].alpha_per_year, fits["pre_2022"].sigma_per_sqrt_year, calendar_years, fits["pre_2022"].n_returns, fits["pre_2022"].total_elapsed_days, "calendar days / 365"),
        "post_2022": GBMCase("post_2022", "2022-2024", fits["post_2022"].alpha_per_year, fits["post_2022"].sigma_per_sqrt_year, calendar_years, fits["post_2022"].n_returns, fits["post_2022"].total_elapsed_days, "calendar days / 365"),
        "exclude_2022": GBMCase("exclude_2022", "2019-2021 + 2023-2024", fits["exclude_2022"].alpha_per_year, fits["exclude_2022"].sigma_per_sqrt_year, calendar_years, fits["exclude_2022"].n_returns, fits["exclude_2022"].total_elapsed_days, "calendar days / 365; disjoint return blocks"),
        "vol_0_75": GBMCase("vol_0_75", "0.75 x baseline volatility", baseline.alpha_per_year, 0.75 * baseline.sigma_per_sqrt_year, calendar_years, baseline.n_returns, baseline.total_elapsed_days, "calendar days / 365; assumed volatility stress"),
        "vol_1_50": GBMCase("vol_1_50", "1.50 x baseline volatility", baseline.alpha_per_year, 1.5 * baseline.sigma_per_sqrt_year, calendar_years, baseline.n_returns, baseline.total_elapsed_days, "calendar days / 365; assumed volatility stress"),
        "session_252": GBMCase("session_252", "252 quote sessions/year", session["alpha_per_year"], session["sigma_per_sqrt_year"], session["horizon_session_years"], session["n_returns"], baseline.total_elapsed_days, "64 assumed quotes / 252 sessions per year"),
    }
    results = {
        key: case_metrics(case, initial_rate=initial_rate, invoice_usd=invoice,
                          budget_lkr=budget, seeds=seeds, n_paths_per_run=n_paths)
        for key, case in cases.items()
    }
    if results["full"]["pooled_metrics"] != stage6["models"]["gbm"]["pooled_metrics"]:
        raise ValueError("Stage 7 baseline does not reproduce Stage 6 GBM metrics")
    base_var = results["full"]["pooled_metrics"]["var_lkr"]
    base_es = results["full"]["pooled_metrics"]["es_lkr"]
    for result in results.values():
        result["delta_vs_full_var_lkr"] = result["pooled_metrics"]["var_lkr"] - base_var
        result["delta_vs_full_es_lkr"] = result["pooled_metrics"]["es_lkr"] - base_es

    convergence = convergence_study(
        initial_rate=initial_rate, alpha_per_year=baseline.alpha_per_year,
        sigma_per_sqrt_year=baseline.sigma_per_sqrt_year,
        invoice_usd=invoice, horizon_days=horizon_days,
    )
    return {
        "scenario": scenario,
        "raw_sha256": gbm_report["raw_sha256"],
        "prepared_sha256": gbm_report["prepared_sha256"],
        "methods": {
            "comparison": "same 2024-12-31 rate, USD invoice, 90-day due date, LKR budget, 8 seeds and 10000 paths per seed; GBM cases reuse the same normal draws per seed",
            "holdout": "2025-2026 excluded from every fit and case definition",
            "volatility_stress": "multiply baseline sigma with common alpha; assumed shock, not data filtering",
            "time_alternative": "training quote returns treated as equal sessions, sample SD annualized by sqrt(252), 64 assumed future quotes imply T=64/252; not mixed with H/365",
            "uncertainty": "pooled mean-cost MC interval and per-seed sample SD/min/max of VaR/ES; conditional on fitted parameters",
        },
        "session_fit": session,
        "cases": results,
        "markov_baseline_from_fresh_stage6_run": stage6["models"]["markov"]["pooled_metrics"],
        "convergence": convergence,
        "limitations": [
            "Cases were predeclared from training-data inspection; they are not unbiased model selection or predictive validation.",
            "The 2022-excluded fit uses two disjoint return blocks without fabricating a bridging return.",
            "The 252-session alternative uses a weekday proxy and ignores holidays/settlement.",
            "Monte Carlo error excludes calibration-window and structural-model uncertainty.",
        ],
    }


def _write_figures(report: dict[str, object], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    keys = ("full", "pre_2022", "post_2022", "exclude_2022", "vol_0_75", "vol_1_50", "session_252")
    cases = report["cases"]
    labels = [cases[k]["parameters"]["label"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, metric, title in zip(axes, ("var_lkr", "es_lkr"), ("VaR95", "ES95"), strict=True):
        values = [cases[k]["pooled_metrics"][metric] / 1e6 for k in keys]
        errors = [cases[k]["between_run_variability"][metric]["sample_sd"] / 1e6 for k in keys]
        ax.barh(labels, values, xerr=errors, color="#3274a1", ecolor="#222222", capsize=3)
        ax.set_xlabel("Extra cost (million LKR)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()
    fig.suptitle("90-day hypothetical invoice: GBM scenario sensitivity")
    fig.text(0.5, 0.01, "Error bars: between-seed SD (8 batches); conditional model simulations", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(figure_dir / "stage7_scenario_comparison.png", dpi=160)
    plt.close(fig)

    points = report["convergence"]["points"]
    counts = np.array([p["n_paths"] for p in points])
    rmse = np.array([p["rmse_mean_cost_lkr"] for p in points])
    reference = np.array([p["theoretical_mc_se_lkr"] for p in points])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(counts, rmse, "o-", label="Replicated RMSE")
    ax.loglog(counts, reference, "--", label="Analytical MC SE (slope -1/2)")
    ax.set_xlabel("Independent paths per replicate")
    ax.set_ylabel("Mean-cost error (LKR)")
    ax.set_title(f"GBM Monte Carlo convergence: observed slope {report['convergence']['log_log_rmse_slope']:.2f}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "stage7_gbm_convergence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = comparison_report(root)
    path = root / "data" / "processed" / "stage7_comparisons.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_figures(report, root / "figures")
    for key, case in report["cases"].items():
        print(f"{key}: VaR95 LKR {case['pooled_metrics']['var_lkr']:.0f}; ES95 LKR {case['pooled_metrics']['es_lkr']:.0f}")
    print(f"convergence slope: {report['convergence']['log_log_rmse_slope']:.3f}; saved {path}")


if __name__ == "__main__":
    main()
