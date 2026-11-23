"""Independent data, probability and time-order checks for the bounded chain."""

from __future__ import annotations

import json
from math import exp, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import fit_calendar_gbm
from src.markov import (
    baseline_markov_report,
    classify_residuals,
    estimate_transition_matrix,
    fit_two_state_markov,
    simulate_markov_paths,
    validate_transition_matrix,
    weekday_quote_offsets,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data" / "processed" / "calibration_returns_2019_2024.csv"
ALL_RETURNS = ROOT / "data" / "processed" / "daily_log_returns.csv"


def test_actual_training_fit_matches_independent_state_calculation() -> None:
    frame = pd.read_csv(TRAINING)
    model = fit_two_state_markov(frame)
    alpha = fit_calendar_gbm(
        frame, window_start="2019-01-01", window_end="2024-12-31"
    ).alpha_per_year
    delta = frame["calendar_gap_days"].to_numpy(dtype=float) / 365.0
    residuals = (frame["log_return"].to_numpy(dtype=float) - alpha * delta) / np.sqrt(delta)
    threshold = float(np.quantile(np.abs(residuals), 0.7, method="linear"))
    labels = (np.abs(residuals) > threshold).astype(int)
    counts = np.array(
        [[np.sum((labels[:-1] == a) & (labels[1:] == b)) for b in (0, 1)] for a in (0, 1)]
    )
    rms = [sqrt(float(np.mean(residuals[labels == k] ** 2))) for k in (0, 1)]

    assert model.n_returns == 1444
    assert model.threshold_abs_residual_per_sqrt_year == pytest.approx(threshold)
    assert model.state_counts == (1011, 433)
    assert model.transition_counts == tuple(map(tuple, counts))
    assert np.allclose(model.transition_probabilities, counts / counts.sum(axis=1)[:, None])
    assert model.state_volatilities_per_sqrt_year == pytest.approx(rms)
    assert model.last_state == int(labels[-1]) == 0
    assert np.allclose(np.sum(model.transition_probabilities, axis=1), 1.0, atol=1e-12)
    assert sum(map(sum, model.transition_counts)) == model.n_returns - 1


def test_holdout_rows_cannot_change_training_states_or_matrix() -> None:
    baseline = fit_two_state_markov(pd.read_csv(TRAINING))
    full_history = fit_two_state_markov(pd.read_csv(ALL_RETURNS))
    assert baseline == full_history
    assert baseline.last_quote_date == "2024-12-31"


def test_missing_prepared_quote_link_is_rejected() -> None:
    frame = pd.read_csv(TRAINING).drop(index=10)
    with pytest.raises(ValueError, match="consecutive published quotes"):
        fit_two_state_markov(frame)


def test_inconsistent_adjacent_quote_rate_link_is_rejected() -> None:
    frame = pd.read_csv(TRAINING)
    frame.loc[10, "previous_usd_lkr_rate"] *= 1.01
    frame.loc[10, "log_return"] = np.log(
        frame.loc[10, "usd_lkr_rate"] / frame.loc[10, "previous_usd_lkr_rate"]
    )
    with pytest.raises(ValueError, match="consecutive published quote rates"):
        fit_two_state_markov(frame)


def test_training_class_guard_rejects_tiny_state_samples() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        fit_two_state_markov(pd.read_csv(TRAINING).head(20))


def test_ties_at_threshold_are_stable() -> None:
    labels = classify_residuals(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), threshold=1.0)
    assert labels.tolist() == [1, 0, 0, 0, 1]


def test_manual_transition_counts_and_probability_rows() -> None:
    counts, matrix = estimate_transition_matrix(np.array([0, 0, 1, 1, 0, 1]))
    assert counts.tolist() == [[1, 2], [1, 1]]
    np.testing.assert_allclose(matrix, [[1 / 3, 2 / 3], [1 / 2, 1 / 2]])
    np.testing.assert_allclose(validate_transition_matrix(matrix), matrix)


def test_unobserved_outgoing_row_is_not_invented() -> None:
    with pytest.raises(ValueError, match="outgoing transition"):
        estimate_transition_matrix(np.array([0, 0, 0]))


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.8, 0.3], [0.2, 0.8]],
        [[1.1, -0.1], [0.2, 0.8]],
        [[float("nan"), 1.0], [0.2, 0.8]],
    ],
)
def test_invalid_transition_matrix_is_rejected(matrix: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="probability rows"):
        validate_transition_matrix(matrix)


def test_weekday_quote_proxy_excludes_weekend_but_not_holiday() -> None:
    # 2024-12-31 was Tuesday. Jan 1 is included as an unverified weekday proxy.
    assert weekday_quote_offsets("2024-12-31", 5) == (1, 2, 3)


def test_new_state_governs_quote_increment_and_remainder_keeps_state() -> None:
    seed = 17
    result = simulate_markov_paths(
        100.0, 0.08, (0.01, 0.30), ((0.0, 1.0), (1.0, 0.0)), 0,
        horizon_days=3, quote_day_offsets=(1,), n_paths=1, seed=seed,
    )
    rng = np.random.default_rng(seed)
    rng.random(1)  # transition draw at the first and only assumed quote
    quote_shock = float(rng.standard_normal(1)[0])
    remainder_shock = float(rng.standard_normal(1)[0])
    expected_quote = 100.0 * exp(0.08 / 365 + 0.30 * sqrt(1 / 365) * quote_shock)
    expected_due = expected_quote * exp(0.08 * 2 / 365 + 0.30 * sqrt(2 / 365) * remainder_shock)
    assert result.day_offsets == (0, 1, 3)
    assert result.states.tolist() == [[0, 1, 1]]
    assert result.rates[0, 1] == pytest.approx(expected_quote)
    assert result.rates[0, 2] == pytest.approx(expected_due)


def test_seed_reproducibility_and_positive_finite_rate_paths() -> None:
    kwargs = dict(
        initial_rate=297.0162,
        alpha_per_year=0.07919696011210609,
        state_volatilities_per_sqrt_year=(0.0105, 0.22),
        transition_probabilities=((0.8, 0.2), (0.4, 0.6)),
        initial_state=0,
        horizon_days=7,
        quote_day_offsets=(1, 2, 3, 6, 7),
        n_paths=100,
    )
    first = simulate_markov_paths(**kwargs, seed=41)
    second = simulate_markov_paths(**kwargs, seed=41)
    other = simulate_markov_paths(**kwargs, seed=42)
    assert np.array_equal(first.rates, second.rates)
    assert np.array_equal(first.states, second.states)
    assert not np.array_equal(first.rates, other.rates)
    assert np.isfinite(first.rates).all() and (first.rates > 0).all()


def test_no_assumed_quote_keeps_initial_state_through_due_date() -> None:
    result = simulate_markov_paths(
        100.0, 0.0, (0.0, 0.0), ((0.0, 1.0), (1.0, 0.0)), 0,
        horizon_days=2, quote_day_offsets=(), n_paths=3, seed=7,
    )
    assert result.day_offsets == (0, 2)
    assert result.states.tolist() == [[0, 0]] * 3
    assert np.all(result.rates == 100.0)


def test_long_synthetic_chain_reproduces_declared_transition_rows() -> None:
    matrix = np.array([[0.85, 0.15], [0.30, 0.70]])
    result = simulate_markov_paths(
        100.0, 0.0, (0.0, 0.0), matrix, 0,
        horizon_days=100, quote_day_offsets=tuple(range(1, 101)),
        n_paths=3000, seed=123,
    )
    before, after = result.states[:, :-1], result.states[:, 1:]
    empirical = np.array([
        [np.mean(after[before == k] == j) for j in (0, 1)] for k in (0, 1)
    ])
    assert np.allclose(empirical, matrix, atol=0.01)
    assert np.all(result.rates == 100.0)


@pytest.mark.parametrize("offsets", [(0,), (2, 1), (3, 3), (6,), (1.0,)])
def test_invalid_quotation_schedules_are_rejected(offsets: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="quote_day_offsets"):
        simulate_markov_paths(
            100.0, 0.0, (0.1, 0.2), ((0.9, 0.1), (0.2, 0.8)), 0,
            horizon_days=5, quote_day_offsets=offsets, n_paths=2, seed=1,
        )


def test_saved_baseline_report_matches_current_calibration() -> None:
    report = baseline_markov_report(ROOT)
    saved = json.loads((ROOT / "data" / "processed" / "markov_calibration_2019_2024.json").read_text())
    model = fit_two_state_markov(pd.read_csv(TRAINING))
    assert report["calibration"] == model.to_dict()
    assert saved == json.loads(json.dumps(report))
    assert report["raw_sha256"] == "b76b443cee6db1bcaa6ffcb515490aafa8e8c2ef339fa411a328cf287af74969"

