"""Bounded quote-step Markov volatility model for CBSL USD TT selling returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from math import sqrt
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.calibration import (
    BASELINE_END,
    BASELINE_START,
    DAYS_PER_YEAR,
    baseline_calibration_report,
    fit_calendar_gbm,
)
from src.data_loader import SOURCE_LABEL

STATE_NAMES = ("stable", "volatile")
DEFAULT_THRESHOLD_QUANTILE = 0.70
MIN_STATE_OBSERVATIONS = 10


@dataclass(frozen=True)
class MarkovCalibration:
    """Training-only state rule, quote-step matrix and conditional volatilities."""

    window_start: str
    window_end: str
    first_quote_date: str
    last_quote_date: str
    n_returns: int
    threshold_quantile: float
    threshold_abs_residual_per_sqrt_year: float
    state_counts: tuple[int, int]
    transition_counts: tuple[tuple[int, int], tuple[int, int]]
    transition_probabilities: tuple[tuple[float, float], tuple[float, float]]
    state_residual_means_per_sqrt_year: tuple[float, float]
    state_volatilities_per_sqrt_year: tuple[float, float]
    alpha_per_year: float
    last_state: int
    last_rate_lkr_per_usd: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MarkovPaths:
    """Rates and states at reference, assumed quote dates and invoice maturity."""

    day_offsets: tuple[int, ...]
    rates: np.ndarray
    states: np.ndarray


def classify_residuals(residuals: np.ndarray, threshold: float) -> np.ndarray:
    """Return 0 for |e| <= threshold and 1 for |e| > threshold (ties stable)."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("residuals must be a nonempty finite one-dimensional array")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return (np.abs(values) > threshold).astype(np.int8)


def validate_transition_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Reject invalid probability rows instead of silently normalizing them."""
    matrix = np.asarray(probabilities, dtype=float)
    if (
        matrix.shape != (2, 2)
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or (matrix > 1).any()
        or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("transition matrix must have two valid probability rows summing to one")
    return matrix


def estimate_transition_matrix(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count successive published-quote state pairs and normalize each row."""
    labels = np.asarray(states)
    if labels.ndim != 1 or labels.size < 2 or not np.isin(labels, (0, 1)).all():
        raise ValueError("states must be a one-dimensional sequence of at least two 0/1 labels")
    counts = np.zeros((2, 2), dtype=int)
    np.add.at(counts, (labels[:-1].astype(int), labels[1:].astype(int)), 1)
    row_totals = counts.sum(axis=1)
    if (row_totals == 0).any():
        raise ValueError("each state needs an observed outgoing transition; no row is invented")
    probabilities = validate_transition_matrix(counts / row_totals[:, None])
    return counts, probabilities


def fit_two_state_markov(
    returns: pd.DataFrame,
    *,
    window_start: str = BASELINE_START,
    window_end: str = BASELINE_END,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
) -> MarkovCalibration:
    """Classify and fit the chain using only the specified training window.

    The GBM fit validates the source, rate arithmetic and calendar gaps. The
    Markov step additionally requires consecutive *observed* return pairs;
    skipping a prepared row cannot silently create a new state transition.
    """
    if not np.isfinite(threshold_quantile) or not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must lie strictly between zero and one")
    gbm = fit_calendar_gbm(returns, window_start=window_start, window_end=window_end)
    dates = pd.to_datetime(returns["date"], errors="raise")
    previous_dates = pd.to_datetime(returns["previous_date"], errors="raise")
    selected = returns.loc[
        dates.between(gbm.window_start, gbm.window_end)
        & previous_dates.between(gbm.window_start, gbm.window_end)
    ].copy()
    selected["date"] = dates.loc[selected.index]
    selected["previous_date"] = previous_dates.loc[selected.index]
    selected = selected.sort_values("date", kind="stable").reset_index(drop=True)
    if not np.array_equal(
        selected["previous_date"].iloc[1:].to_numpy(),
        selected["date"].iloc[:-1].to_numpy(),
    ):
        raise ValueError("training returns must link consecutive published quotes")
    if not np.allclose(
        selected["previous_usd_lkr_rate"].iloc[1:].to_numpy(dtype=float),
        selected["usd_lkr_rate"].iloc[:-1].to_numpy(dtype=float),
        rtol=1e-10, atol=1e-12,
    ):
        raise ValueError("training returns must link consecutive published quote rates")

    delta_years = selected["calendar_gap_days"].to_numpy(dtype=float) / DAYS_PER_YEAR
    residuals = (
        selected["log_return"].to_numpy(dtype=float) - gbm.alpha_per_year * delta_years
    ) / np.sqrt(delta_years)
    threshold = float(np.quantile(np.abs(residuals), threshold_quantile, method="linear"))
    states = classify_residuals(residuals, threshold)
    state_counts = np.bincount(states, minlength=2)
    if (state_counts < MIN_STATE_OBSERVATIONS).any():
        raise ValueError(
            f"each state needs at least {MIN_STATE_OBSERVATIONS} training returns"
        )
    transition_counts, transition_probabilities = estimate_transition_matrix(states)
    state_means = tuple(float(np.mean(residuals[states == k])) for k in (0, 1))
    state_rms = tuple(float(np.sqrt(np.mean(np.square(residuals[states == k])))) for k in (0, 1))
    if state_rms[1] <= state_rms[0]:
        raise ValueError("volatile-state RMS must exceed stable-state RMS")

    return MarkovCalibration(
        window_start=gbm.window_start,
        window_end=gbm.window_end,
        first_quote_date=gbm.first_quote_date,
        last_quote_date=gbm.last_quote_date,
        n_returns=gbm.n_returns,
        threshold_quantile=float(threshold_quantile),
        threshold_abs_residual_per_sqrt_year=threshold,
        state_counts=(int(state_counts[0]), int(state_counts[1])),
        transition_counts=tuple(tuple(int(v) for v in row) for row in transition_counts),
        transition_probabilities=tuple(
            tuple(float(v) for v in row) for row in transition_probabilities
        ),
        state_residual_means_per_sqrt_year=state_means,
        state_volatilities_per_sqrt_year=state_rms,
        alpha_per_year=gbm.alpha_per_year,
        last_state=int(states[-1]),
        last_rate_lkr_per_usd=gbm.last_rate_lkr_per_usd,
    )


def weekday_quote_offsets(reference_date: str, horizon_days: int) -> tuple[int, ...]:
    """Assume Monday-Friday quotations, ignoring unverified holidays."""
    start = date.fromisoformat(reference_date)
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")
    return tuple(
        day for day in range(1, horizon_days + 1)
        if (start + timedelta(days=day)).weekday() < 5
    )


def _quote_offsets(offsets: Sequence[int], horizon_days: int) -> tuple[int, ...]:
    checked = tuple(offsets)
    if any(
        isinstance(day, bool) or not isinstance(day, (int, np.integer))
        or day <= 0 or day > horizon_days
        for day in checked
    ) or any(a >= b for a, b in zip(checked, checked[1:])):
        raise ValueError("quote_day_offsets must be increasing whole days within the horizon")
    return tuple(int(day) for day in checked)


def simulate_markov_paths(
    initial_rate: float,
    alpha_per_year: float,
    state_volatilities_per_sqrt_year: Sequence[float],
    transition_probabilities: Sequence[Sequence[float]],
    initial_state: int,
    *,
    horizon_days: int,
    quote_day_offsets: Sequence[int],
    n_paths: int,
    seed: int,
) -> MarkovPaths:
    """Sample next state, then its exact calendar-gap rate increment.

    One Markov transition occurs at each assumed published quote, never at
    every calendar day. Any final interval to the invoice date uses the last
    state without another transition.
    """
    if not np.isfinite(initial_rate) or initial_rate <= 0 or not np.isfinite(alpha_per_year):
        raise ValueError("initial rate must be positive and alpha finite")
    sigmas = np.asarray(state_volatilities_per_sqrt_year, dtype=float)
    if sigmas.shape != (2,) or not np.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("two finite nonnegative state volatilities are required")
    matrix = validate_transition_matrix(transition_probabilities)
    if isinstance(initial_state, bool) or not isinstance(initial_state, (int, np.integer)) or initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 or 1")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    offsets = _quote_offsets(quote_day_offsets, horizon_days)
    has_remainder = not offsets or offsets[-1] < horizon_days
    times = (0, *offsets, *((horizon_days,) if has_remainder else ()))
    rates = np.empty((n_paths, len(times)), dtype=float)
    states = np.empty((n_paths, len(times)), dtype=np.int8)
    rates[:, 0] = initial_rate
    states[:, 0] = initial_state
    rng = np.random.default_rng(seed)

    for j, current_day in enumerate(offsets, start=1):
        previous_day = times[j - 1]
        delta_years = (current_day - previous_day) / DAYS_PER_YEAR
        previous_states = states[:, j - 1]
        states[:, j] = (rng.random(n_paths) < matrix[previous_states, 1]).astype(np.int8)
        shocks = rng.standard_normal(n_paths)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rates[:, j] = rates[:, j - 1] * np.exp(
                alpha_per_year * delta_years
                + sigmas[states[:, j]] * sqrt(delta_years) * shocks
            )
        if not np.isfinite(rates[:, j]).all() or (rates[:, j] <= 0).any():
            raise OverflowError("Markov rate left the finite positive range")

    if has_remainder:
        j = len(times) - 1
        delta_years = (horizon_days - times[j - 1]) / DAYS_PER_YEAR
        states[:, j] = states[:, j - 1]
        shocks = rng.standard_normal(n_paths)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rates[:, j] = rates[:, j - 1] * np.exp(
                alpha_per_year * delta_years
                + sigmas[states[:, j]] * sqrt(delta_years) * shocks
            )
        if not np.isfinite(rates[:, j]).all() or (rates[:, j] <= 0).any():
            raise OverflowError("Markov rate left the finite positive range")
    return MarkovPaths(day_offsets=times, rates=rates, states=states)


def baseline_markov_report(project_root: Path) -> dict[str, object]:
    """Recreate the 2019-2024 training result after checking source hashes."""
    gbm_report = baseline_calibration_report(project_root)
    prepared = project_root / "data" / "processed" / "calibration_returns_2019_2024.csv"
    model = fit_two_state_markov(pd.read_csv(prepared))
    if not np.isclose(
        model.alpha_per_year, gbm_report["calibration"]["alpha_per_year"], rtol=0, atol=1e-12
    ):
        raise ValueError("Markov and baseline GBM log drift disagree")
    if model.n_returns != gbm_report["calibration"]["n_returns"]:
        raise ValueError("Markov and baseline GBM training counts disagree")
    return {
        "model": "two-state quote-increment Markov volatility extension",
        "source": SOURCE_LABEL,
        "return_clock": "elapsed calendar days / 365 years",
        "transition_clock": "one successive published-quote increment",
        "state_names": STATE_NAMES,
        "classification": "abs((log_return - alpha*delta_years)/sqrt(delta_years)); stable at or below training 70th percentile",
        "quantile_method": "NumPy linear",
        "state_minimum_training_returns": MIN_STATE_OBSERVATIONS,
        "raw_file": gbm_report["raw_file"],
        "raw_sha256": gbm_report["raw_sha256"],
        "prepared_file": gbm_report["prepared_file"],
        "prepared_sha256": gbm_report["prepared_sha256"],
        "calibration": model.to_dict(),
        "limitations": [
            "Realized-return state labels are a heuristic, not verified hidden economic regimes.",
            "The 2022 jumps remain and strongly affect the volatile-state RMS.",
            "P is per published quote increment, not per calendar day.",
            "Future quote schedules are assumptions; the weekday helper ignores holidays and settlement rules.",
            "The 2025-2026 evaluation snapshot is excluded from threshold, drift, RMS and matrix estimation.",
        ],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = baseline_markov_report(project_root)
    path = project_root / "data" / "processed" / "markov_calibration_2019_2024.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
