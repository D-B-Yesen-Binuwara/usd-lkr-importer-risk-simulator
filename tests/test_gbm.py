"""Independent numerical and deterministic checks for the calendar-time GBM."""

from __future__ import annotations

from datetime import date, timedelta
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from src.calibration import baseline_calibration_report, fit_calendar_gbm
from src.data_loader import SOURCE_LABEL
from src.gbm import (
    exact_gbm_step,
    simulate_paths,
    simulate_terminal_rates,
    theoretical_terminal_moments,
    theoretical_terminal_quantile,
)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = ROOT / "data" / "processed" / "calibration_returns_2019_2024.csv"
ALL_RETURNS_FILE = ROOT / "data" / "processed" / "daily_log_returns.csv"


def make_returns(rates: list[float], gaps: list[int], first_date: date) -> pd.DataFrame:
    rows = []
    previous_date = first_date
    for previous_rate, rate, gap in zip(rates[:-1], rates[1:], gaps, strict=True):
        current_date = previous_date + timedelta(days=gap)
        rows.append(
            {
                "date": current_date.isoformat(),
                "previous_date": previous_date.isoformat(),
                "calendar_gap_days": gap,
                "usd_lkr_rate": rate,
                "previous_usd_lkr_rate": previous_rate,
                "log_return": log(rate / previous_rate),
                "source": SOURCE_LABEL,
            }
        )
        previous_date = current_date
    return pd.DataFrame(rows)


def fit_fixture(frame: pd.DataFrame):
    return fit_calendar_gbm(frame, window_start="2019-01-01", window_end="2024-12-31")


def test_equal_gap_fit_reduces_to_standard_gaussian_mle() -> None:
    frame = make_returns([100.0, 102.0, 98.0, 101.0, 99.0], [1, 1, 1, 1], date(2020, 1, 1))
    fit = fit_fixture(frame)
    values = np.log(np.array([102 / 100, 98 / 102, 101 / 98, 99 / 101]))
    expected_alpha = float(values.mean() * 365)
    expected_sigma_squared = float(np.mean(np.square(values - values.mean())) * 365)
    assert fit.alpha_per_year == pytest.approx(expected_alpha, abs=1e-12)
    assert fit.sigma_per_sqrt_year == pytest.approx(sqrt(expected_sigma_squared), abs=1e-12)
    assert fit.mu_per_year == pytest.approx(expected_alpha + expected_sigma_squared / 2, abs=1e-12)
    assert fit.n_returns == 4 and fit.total_elapsed_days == 4


def test_irregular_gap_fit_matches_independent_likelihood_optimizer() -> None:
    frame = pd.read_csv(CALIBRATION_FILE)
    fit = fit_fixture(frame)
    dt = frame["calendar_gap_days"].to_numpy(dtype=float) / 365
    values = frame["log_return"].to_numpy(dtype=float)

    def negative_log_likelihood(params: np.ndarray) -> float:
        alpha, log_variance = params
        variance = np.exp(log_variance) * dt
        return float(0.5 * np.sum(np.log(2 * np.pi * variance) + (values - alpha * dt) ** 2 / variance))

    optimum = minimize(negative_log_likelihood, [0.0, log(0.02)], method="L-BFGS-B")
    assert optimum.success, optimum.message
    assert fit.alpha_per_year == pytest.approx(optimum.x[0], abs=1e-6)
    assert fit.sigma_per_sqrt_year**2 == pytest.approx(exp(optimum.x[1]), abs=1e-7)


def test_baseline_fit_uses_only_training_rows_and_matches_source_audit() -> None:
    baseline = fit_fixture(pd.read_csv(CALIBRATION_FILE))
    full_history = fit_fixture(pd.read_csv(ALL_RETURNS_FILE))
    assert baseline == full_history
    assert baseline.n_returns == 1444
    assert baseline.total_elapsed_days == 2191
    assert baseline.first_quote_date == "2019-01-01"
    assert baseline.last_quote_date == "2024-12-31"
    report = baseline_calibration_report(ROOT)
    assert report["calibration"] == baseline.to_dict()
    assert report["raw_sha256"] == "b76b443cee6db1bcaa6ffcb515490aafa8e8c2ef339fa411a328cf287af74969"


def test_synthetic_irregular_series_approximately_recovers_known_parameters() -> None:
    rng = np.random.default_rng(137)
    gaps = rng.integers(1, 5, size=10_000)
    alpha, sigma = 0.04, 0.17
    dt = gaps / 365
    increments = alpha * dt + sigma * np.sqrt(dt) * rng.standard_normal(gaps.size)
    rates = (100 * np.exp(np.concatenate(([0.0], np.cumsum(increments))))).tolist()
    frame = make_returns(rates, gaps.tolist(), date(2020, 1, 1))
    fit = fit_calendar_gbm(frame, window_start="2020-01-01", window_end="2200-12-31")
    assert abs(fit.alpha_per_year - alpha) < 0.06
    assert abs(fit.sigma_per_sqrt_year - sigma) < 0.007


@pytest.mark.parametrize("change,match", [
    ({"calendar_gap_days": 0}, "Calendar gaps"),
    ({"calendar_gap_days": 2}, "Calendar gaps"),
    ({"log_return": 1.0}, "Log returns"),
    ({"source": "FRED DEXSLUS"}, "CBSL USD TT selling-rate"),
    ({"usd_lkr_rate": -5.0}, "Rates must be positive"),
])
def test_fit_rejects_inconsistent_or_wrong_source_rows(change: dict, match: str) -> None:
    frame = make_returns([100.0, 101.0, 99.0], [1, 1], date(2020, 1, 1))
    for column, value in change.items():
        frame.loc[1, column] = value
    with pytest.raises(ValueError, match=match):
        fit_fixture(frame)


def test_exact_step_zero_volatility_and_seed_reproducibility() -> None:
    initial, alpha, horizon = 297.0162, 0.08, 90
    deterministic = initial * exp(alpha * horizon / 365)
    assert exact_gbm_step(initial, alpha, 0.0, horizon / 365, 42.0) == pytest.approx(deterministic)
    zero_vol = simulate_terminal_rates(initial, alpha, 0.0, horizon_days=horizon, n_paths=50, seed=42)
    assert np.allclose(zero_vol, deterministic)
    assert np.all(simulate_terminal_rates(initial, 0.0, 0.0, horizon_days=horizon, n_paths=50, seed=42) == initial)
    first = simulate_terminal_rates(initial, alpha, 0.12, horizon_days=horizon, n_paths=100, seed=42)
    second = simulate_terminal_rates(initial, alpha, 0.12, horizon_days=horizon, n_paths=100, seed=42)
    other = simulate_terminal_rates(initial, alpha, 0.12, horizon_days=horizon, n_paths=100, seed=43)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.isfinite(first).all() and (first > 0).all()


def test_path_partition_reaches_exact_calendar_horizon() -> None:
    steps = [1.0, 2.0, 3.0]
    shocks = np.random.default_rng(19).standard_normal((3, len(steps)))
    paths = simulate_paths(100.0, 0.02, 0.3, horizon_days=6, n_paths=3, seed=19, step_days=steps)
    expected_logs = np.cumsum(
        0.02 * np.array(steps) / 365 + 0.3 * np.sqrt(np.array(steps) / 365) * shocks,
        axis=1,
    )
    assert paths.shape == (3, 4)
    assert np.array_equal(paths[:, 0], np.full(3, 100.0))
    assert np.allclose(paths[:, 1:], 100.0 * np.exp(expected_logs), rtol=1e-13)
    assert np.isfinite(paths).all() and (paths > 0).all()
    with pytest.raises(ValueError, match="summing to horizon_days"):
        simulate_paths(100.0, 0.02, 0.3, horizon_days=6, n_paths=3, seed=19, step_days=[1.0, 2.0])


def test_terminal_mean_variance_and_quantiles_match_analytic_gbm() -> None:
    initial, alpha, sigma, horizon = 300.0, 0.05, 0.25, 180
    values = simulate_terminal_rates(initial, alpha, sigma, horizon_days=horizon, n_paths=80_000, seed=2026)
    mean, variance = theoretical_terminal_moments(initial, alpha, sigma, horizon_days=horizon)
    assert abs(float(values.mean()) - mean) <= 4 * sqrt(variance / len(values))
    assert abs(float(values.var(ddof=1)) - variance) / variance < 0.05
    for probability in (0.5, 0.95):
        expected = theoretical_terminal_quantile(
            initial, alpha, sigma, horizon_days=horizon, probability=probability
        )
        assert abs(float(np.quantile(values, probability)) - expected) / expected < 0.02


def test_invalid_simulation_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="initial_rate"):
        simulate_terminal_rates(0.0, 0.02, 0.2, horizon_days=90, n_paths=10, seed=1)
    with pytest.raises(ValueError, match="sigma_per_sqrt_year"):
        simulate_terminal_rates(100.0, 0.02, -0.2, horizon_days=90, n_paths=10, seed=1)
    with pytest.raises(ValueError, match="horizon_days"):
        simulate_terminal_rates(100.0, 0.02, 0.2, horizon_days=0, n_paths=10, seed=1)
    with pytest.raises(ValueError, match="probability"):
        theoretical_terminal_quantile(100.0, 0.02, 0.2, horizon_days=90, probability=1.0)
