"""Importer cost, upper-tail risk and reproducible run checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.gbm import simulate_terminal_rates
from src.risk import between_run_variability, empirical_upper_tail_risk, payment_risk_metrics
from src.run_risk import baseline_risk_report

ROOT = Path(__file__).resolve().parents[1]


def test_payment_metrics_match_hand_calculated_invoice() -> None:
    result = payment_risk_metrics(
        np.array([90.0, 100.0, 110.0, 120.0]),
        invoice_usd=100.0, reference_rate_lkr_per_usd=100.0,
        budget_lkr=10_500.0, level=0.75,
    )
    assert result.reference_cost_lkr == 10_000.0
    assert result.mean_cost_lkr == 10_500.0
    assert result.median_cost_lkr == 10_500.0
    assert result.p05_cost_lkr == pytest.approx(9_150.0)
    assert result.p95_cost_lkr == pytest.approx(11_850.0)
    assert result.mean_extra_cost_lkr == 500.0
    assert result.budget_exceedance_probability == 0.5
    assert result.var_lkr == 1_000.0
    assert result.es_lkr == 2_000.0
    assert result.mean_cost_mc_ci95_lkr[0] < result.mean_cost_lkr < result.mean_cost_mc_ci95_lkr[1]


def test_fractional_tail_and_ties_use_quantile_integral_es() -> None:
    var, es = empirical_upper_tail_risk(np.array([0.0, 10.0, 20.0, 30.0]), level=0.625)
    assert var == 20.0
    assert es == pytest.approx((30.0 + 0.5 * 20.0) / 1.5)
    tied_var, tied_es = empirical_upper_tail_risk(np.array([0.0, 0.0, 0.0, 10.0]), level=0.5)
    assert tied_var == 0.0
    assert tied_es == 5.0  # Mean(L >= VaR) would incorrectly be 2.5.


def test_signed_savings_are_not_clamped_to_zero() -> None:
    result = payment_risk_metrics(
        np.array([80.0, 90.0, 95.0, 99.0]), invoice_usd=100.0,
        reference_rate_lkr_per_usd=100.0, budget_lkr=10_000.0,
    )
    assert result.mean_extra_cost_lkr < 0
    assert result.var_lkr < 0 and result.es_lkr < 0
    assert result.budget_exceedance_probability == 0.0


def test_fixed_rate_paths_scale_cost_and_tail_metrics_with_invoice_size() -> None:
    rates = np.array([95.0, 100.0, 105.0, 130.0, 140.0])
    first = payment_risk_metrics(rates, invoice_usd=100.0, reference_rate_lkr_per_usd=100.0, budget_lkr=11_000.0)
    double = payment_risk_metrics(rates, invoice_usd=200.0, reference_rate_lkr_per_usd=100.0, budget_lkr=22_000.0)
    for name in (
        "reference_cost_lkr", "mean_cost_lkr", "median_cost_lkr", "p05_cost_lkr",
        "p95_cost_lkr", "mean_extra_cost_lkr", "var_lkr", "es_lkr",
        "mean_cost_mc_se_lkr",
    ):
        assert getattr(double, name) == pytest.approx(2 * getattr(first, name))
    assert double.mean_cost_mc_ci95_lkr == pytest.approx(tuple(2 * x for x in first.mean_cost_mc_ci95_lkr))
    assert double.budget_exceedance_probability == first.budget_exceedance_probability


def test_fixed_lkr_budget_probability_can_change_with_invoice_size() -> None:
    rates = np.array([95.0, 100.0, 105.0, 130.0])
    first = payment_risk_metrics(rates, invoice_usd=100.0, reference_rate_lkr_per_usd=100.0, budget_lkr=12_000.0)
    double = payment_risk_metrics(rates, invoice_usd=200.0, reference_rate_lkr_per_usd=100.0, budget_lkr=12_000.0)
    assert first.budget_exceedance_probability == 0.25
    assert double.budget_exceedance_probability == 1.0


def test_zero_volatility_has_deterministic_cost_and_zero_mc_error() -> None:
    rates = simulate_terminal_rates(100.0, 0.0, 0.0, horizon_days=90, n_paths=100, seed=7)
    result = payment_risk_metrics(rates, invoice_usd=100.0, reference_rate_lkr_per_usd=100.0, budget_lkr=11_000.0)
    assert result.reference_cost_lkr == result.mean_cost_lkr == 10_000.0
    assert result.var_lkr == result.es_lkr == result.mean_extra_cost_lkr == 0.0
    assert result.mean_cost_mc_ci95_lkr == (10_000.0, 10_000.0)


@pytest.mark.parametrize(
    "rates,invoice,reference,budget",
    [
        ([100.0], 100.0, 100.0, 10_000.0),
        ([100.0, 0.0], 100.0, 100.0, 10_000.0),
        ([100.0, float("nan")], 100.0, 100.0, 10_000.0),
        ([100.0, 110.0], 0.0, 100.0, 10_000.0),
        ([100.0, 110.0], 100.0, -1.0, 10_000.0),
        ([100.0, 110.0], 100.0, 100.0, 0.0),
    ],
)
def test_invalid_payment_inputs_fail(rates, invoice, reference, budget) -> None:
    with pytest.raises(ValueError):
        payment_risk_metrics(
            np.array(rates), invoice_usd=invoice,
            reference_rate_lkr_per_usd=reference, budget_lkr=budget,
        )


def test_replicate_spread_uses_independent_run_estimates() -> None:
    a = payment_risk_metrics(np.array([90.0, 110.0]), invoice_usd=100.0, reference_rate_lkr_per_usd=100.0, budget_lkr=10_000.0)
    b = payment_risk_metrics(np.array([100.0, 120.0]), invoice_usd=100.0, reference_rate_lkr_per_usd=100.0, budget_lkr=10_000.0)
    spread = between_run_variability([a, b])
    assert spread["mean_cost_lkr"]["mean"] == 10_500.0
    assert spread["mean_cost_lkr"]["sample_sd"] == pytest.approx(1_000.0 / np.sqrt(2))
    assert spread["var_lkr"]["min"] <= spread["var_lkr"]["max"]


def test_small_baseline_run_is_reproducible_and_has_frozen_reference() -> None:
    kwargs = {"n_paths_per_run": 200, "seeds": (7, 8)}
    first = baseline_risk_report(ROOT, **kwargs)
    second = baseline_risk_report(ROOT, **kwargs)
    assert first == second
    scenario = first["scenario"]
    assert scenario["reference_date"] == "2024-12-31"
    assert scenario["due_date"] == "2025-03-31"
    assert scenario["reference_cost_lkr"] == pytest.approx(29_701_620.0)
    assert scenario["budget_lkr"] == pytest.approx(31_186_701.0)
    assert scenario["total_paths_per_model"] == 400
    assert len(scenario["markov_quote_day_offsets"]) == 64
    for model in first["models"].values():
        assert model["pooled_metrics"]["n_paths"] == 400
        assert len(model["runs"]) == 2
        assert model["pooled_metrics"]["es_lkr"] >= model["pooled_metrics"]["var_lkr"]


def test_saved_default_report_matches_fresh_seeded_run_and_analytic_mean() -> None:
    saved = json.loads((ROOT / "data" / "processed" / "importer_risk_baseline_2019_2024.json").read_text())
    fresh = baseline_risk_report(ROOT)
    assert saved == json.loads(json.dumps(fresh))
    mean = fresh["models"]["gbm"]["pooled_metrics"]["mean_cost_lkr"]
    se = fresh["models"]["gbm"]["pooled_metrics"]["mean_cost_mc_se_lkr"]
    assert abs(mean - fresh["gbm_analytic_expected_cost_lkr"]) < 5 * se

