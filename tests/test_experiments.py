"""Independent checks for frozen-window sensitivity and MC convergence."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments import (
    calibration_windows, comparison_report, convergence_study, quote_session_fit,
)
from src.gbm import simulate_terminal_rates, theoretical_terminal_moments, theoretical_terminal_quantile

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data" / "processed" / "calibration_returns_2019_2024.csv"


def test_predeclared_windows_use_training_only_and_no_2022_bridge() -> None:
    frame = pd.read_csv(TRAINING)
    fits = calibration_windows(frame)
    assert {key: fit.n_returns for key, fit in fits.items()} == {
        "full": 1444, "pre_2022": 721, "post_2022": 722, "exclude_2022": 1204,
    }
    assert fits["full"].total_elapsed_days == 2191
    assert fits["exclude_2022"].total_elapsed_days == 1824
    assert fits["pre_2022"].last_quote_date <= "2021-12-31"
    assert fits["post_2022"].first_quote_date >= "2022-01-01"
    assert fits["post_2022"].sigma_per_sqrt_year > fits["pre_2022"].sigma_per_sqrt_year

    full_history = pd.read_csv(ROOT / "data" / "processed" / "daily_log_returns.csv")
    assert calibration_windows(full_history) == fits  # 2025-2026 cannot alter the cases.


def test_quote_session_fit_uses_sample_sd_and_explicit_quote_count() -> None:
    frame = pd.read_csv(TRAINING)
    fit = quote_session_fit(frame, 64)
    r = frame["log_return"].to_numpy(dtype=float)
    assert fit["alpha_per_year"] == pytest.approx(float(np.mean(r) * 252))
    assert fit["sigma_per_sqrt_year"] == pytest.approx(float(np.std(r, ddof=1) * sqrt(252)))
    assert fit["horizon_session_years"] == pytest.approx(64 / 252)
    assert fit["horizon_session_years"] != pytest.approx(90 / 365)
    with pytest.raises(ValueError, match="only the selected training"):
        quote_session_fit(pd.read_csv(ROOT / "data" / "processed" / "daily_log_returns.csv"), 64)


def test_gbm_sample_moments_and_quantile_match_analytical_references() -> None:
    c = json.loads((ROOT / "data" / "processed" / "gbm_calibration_2019_2024.json").read_text())["calibration"]
    rates = simulate_terminal_rates(
        c["last_rate_lkr_per_usd"], c["alpha_per_year"], c["sigma_per_sqrt_year"],
        horizon_days=90, n_paths=100_000, seed=2026,
    )
    mean, variance = theoretical_terminal_moments(
        c["last_rate_lkr_per_usd"], c["alpha_per_year"], c["sigma_per_sqrt_year"],
        horizon_days=90,
    )
    q95 = theoretical_terminal_quantile(
        c["last_rate_lkr_per_usd"], c["alpha_per_year"], c["sigma_per_sqrt_year"],
        horizon_days=90, probability=0.95,
    )
    assert abs(float(np.mean(rates)) - mean) < 4 * sqrt(variance / len(rates))
    assert float(np.var(rates, ddof=1)) == pytest.approx(variance, rel=0.03)
    assert float(np.quantile(rates, 0.95)) == pytest.approx(q95, rel=0.015)


def test_replicated_mean_error_scaling_is_close_to_sqrt_n_law() -> None:
    c = json.loads((ROOT / "data" / "processed" / "gbm_calibration_2019_2024.json").read_text())["calibration"]
    study = convergence_study(
        initial_rate=c["last_rate_lkr_per_usd"], alpha_per_year=c["alpha_per_year"],
        sigma_per_sqrt_year=c["sigma_per_sqrt_year"], invoice_usd=100_000.0,
        horizon_days=90,
    )
    points = study["points"]
    assert [p["n_paths"] for p in points] == [500, 2000, 8000, 32000]
    assert all(p["replicates"] == 32 for p in points)
    assert points[-1]["rmse_mean_cost_lkr"] < points[0]["rmse_mean_cost_lkr"]
    assert -0.75 < study["log_log_rmse_slope"] < -0.25
    assert all(0.7 < p["rmse_mean_cost_lkr"] / p["theoretical_mc_se_lkr"] < 1.3 for p in points)


def test_saved_comparison_reproduces_stage6_and_records_uncertainty() -> None:
    saved = json.loads((ROOT / "data" / "processed" / "stage7_comparisons.json").read_text())
    fresh = comparison_report(ROOT)
    assert saved == json.loads(json.dumps(fresh))
    stage6 = json.loads((ROOT / "data" / "processed" / "importer_risk_baseline_2019_2024.json").read_text())
    assert saved["cases"]["full"]["pooled_metrics"] == stage6["models"]["gbm"]["pooled_metrics"]
    assert saved["markov_baseline_from_fresh_stage6_run"] == stage6["models"]["markov"]["pooled_metrics"]
    assert len(saved["cases"]) == 7
    assert all(
        result["between_run_variability"]["var_lkr"]["sample_sd"] > 0
        for result in saved["cases"].values()
    )
    assert saved["cases"]["vol_0_75"]["pooled_metrics"]["var_lkr"] < saved["cases"]["full"]["pooled_metrics"]["var_lkr"]
    assert saved["cases"]["vol_1_50"]["pooled_metrics"]["var_lkr"] > saved["cases"]["full"]["pooled_metrics"]["var_lkr"]
