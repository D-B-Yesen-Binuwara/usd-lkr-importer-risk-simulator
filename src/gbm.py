"""Exact calendar-time GBM transitions and analytical terminal benchmarks."""

from __future__ import annotations

from math import exp, expm1, isclose, sqrt
from statistics import NormalDist
from typing import Sequence

import numpy as np

from src.calibration import DAYS_PER_YEAR


def _parameters(initial_rate: float, alpha_per_year: float, sigma_per_sqrt_year: float) -> None:
    if not np.isfinite(initial_rate) or initial_rate <= 0:
        raise ValueError("initial_rate must be finite and strictly positive")
    if not np.isfinite(alpha_per_year):
        raise ValueError("alpha_per_year must be finite")
    if not np.isfinite(sigma_per_sqrt_year) or sigma_per_sqrt_year < 0:
        raise ValueError("sigma_per_sqrt_year must be finite and nonnegative")


def _run_size(n_paths: int, seed: int) -> None:
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _horizon(horizon_days: int) -> float:
    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, (int, np.integer))
        or horizon_days <= 0
    ):
        raise ValueError("horizon_days must be a positive integer")
    return float(horizon_days) / DAYS_PER_YEAR


def exact_gbm_step(
    rate: float | np.ndarray,
    alpha_per_year: float,
    sigma_per_sqrt_year: float,
    delta_years: float,
    shock: float | np.ndarray,
) -> float | np.ndarray:
    """Return S_next = S_now * exp(alpha*dt + sigma*sqrt(dt)*shock)."""
    rates = np.asarray(rate, dtype=float)
    shocks = np.asarray(shock, dtype=float)
    if rates.size == 0 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("rate must be finite and strictly positive")
    _parameters(float(rates.flat[0]), alpha_per_year, sigma_per_sqrt_year)
    if not np.isfinite(delta_years) or delta_years <= 0:
        raise ValueError("delta_years must be finite and strictly positive")
    if not np.isfinite(shocks).all():
        raise ValueError("shock values must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = rates * np.exp(alpha_per_year * delta_years + sigma_per_sqrt_year * sqrt(delta_years) * shocks)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise OverflowError("GBM rate left the finite positive range for these inputs")
    return float(result) if result.ndim == 0 else result


def simulate_terminal_rates(
    initial_rate: float,
    alpha_per_year: float,
    sigma_per_sqrt_year: float,
    *,
    horizon_days: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Draw independent terminal rates directly over H/365 calendar years."""
    _parameters(initial_rate, alpha_per_year, sigma_per_sqrt_year)
    _run_size(n_paths, seed)
    elapsed_years = _horizon(horizon_days)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_paths)
    return np.asarray(
        exact_gbm_step(initial_rate, alpha_per_year, sigma_per_sqrt_year, elapsed_years, shocks)
    )


def simulate_paths(
    initial_rate: float,
    alpha_per_year: float,
    sigma_per_sqrt_year: float,
    *,
    horizon_days: int,
    n_paths: int,
    seed: int,
    step_days: Sequence[float] | None = None,
) -> np.ndarray:
    """Draw GBM paths with exact increments whose calendar days sum to H.

    By default each step is one calendar day. A caller-supplied positive
    partition can be used for a chart or a verified quotation schedule; it
    must end exactly at the requested payment horizon.
    """
    _parameters(initial_rate, alpha_per_year, sigma_per_sqrt_year)
    _run_size(n_paths, seed)
    _horizon(horizon_days)
    intervals = (
        np.ones(horizon_days, dtype=float)
        if step_days is None
        else np.asarray(step_days, dtype=float)
    )
    if (
        intervals.ndim != 1
        or intervals.size == 0
        or not np.isfinite(intervals).all()
        or (intervals <= 0).any()
        or not isclose(float(np.sum(intervals)), float(horizon_days), rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("step_days must be positive finite intervals summing to horizon_days")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, intervals.size))
    paths = np.empty((n_paths, intervals.size + 1), dtype=float)
    paths[:, 0] = initial_rate
    for j, days in enumerate(intervals):
        paths[:, j + 1] = exact_gbm_step(
            paths[:, j], alpha_per_year, sigma_per_sqrt_year, days / DAYS_PER_YEAR, shocks[:, j]
        )
    return paths


def theoretical_terminal_moments(
    initial_rate: float,
    alpha_per_year: float,
    sigma_per_sqrt_year: float,
    *,
    horizon_days: int,
) -> tuple[float, float]:
    """Return the GBM terminal mean and variance in LKR/USD units."""
    _parameters(initial_rate, alpha_per_year, sigma_per_sqrt_year)
    elapsed_years = _horizon(horizon_days)
    mu = alpha_per_year + 0.5 * sigma_per_sqrt_year**2
    mean = initial_rate * exp(mu * elapsed_years)
    variance = initial_rate**2 * exp(2 * mu * elapsed_years) * expm1(sigma_per_sqrt_year**2 * elapsed_years)
    if not np.isfinite(mean) or not np.isfinite(variance):
        raise OverflowError("Analytical moments are non-finite for these inputs")
    return mean, variance


def theoretical_terminal_quantile(
    initial_rate: float,
    alpha_per_year: float,
    sigma_per_sqrt_year: float,
    *,
    horizon_days: int,
    probability: float,
) -> float:
    """Return an analytical GBM rate quantile for 0 < probability < 1."""
    _parameters(initial_rate, alpha_per_year, sigma_per_sqrt_year)
    elapsed_years = _horizon(horizon_days)
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("probability must lie strictly between zero and one")
    result = initial_rate * exp(
        alpha_per_year * elapsed_years
        + sigma_per_sqrt_year * sqrt(elapsed_years) * NormalDist().inv_cdf(probability)
    )
    if not np.isfinite(result) or result <= 0:
        raise OverflowError("Analytical quantile is non-finite for these inputs")
    return result
