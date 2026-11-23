"""Importer payment costs and upper-tail risk from simulated USD/LKR rates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite, sqrt
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class PaymentRiskMetrics:
    """LKR results for a fixed positive USD invoice and one simulation sample."""

    n_paths: int
    reference_cost_lkr: float
    mean_cost_lkr: float
    median_cost_lkr: float
    p05_cost_lkr: float
    p95_cost_lkr: float
    mean_extra_cost_lkr: float
    budget_exceedance_probability: float
    var_lkr: float
    es_lkr: float
    mean_cost_mc_se_lkr: float
    mean_cost_mc_ci95_lkr: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def empirical_upper_tail_risk(extra_costs_lkr: np.ndarray, *, level: float = 0.95) -> tuple[float, float]:
    """Empirical left quantile and quantile-integral ES of signed extra cost.

    ES weights the worst (1-level)*N observations, including a fractional
    boundary observation when that count is not an integer. It remains
    consistent when observations tie at VaR.
    """
    values = np.asarray(extra_costs_lkr, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("extra costs must be a nonempty finite one-dimensional array")
    if not isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie strictly between zero and one")
    ordered = np.sort(values)
    n = ordered.size
    value_at_risk = float(ordered[ceil(level * n) - 1])
    tail_count = (1 - level) * n
    whole = floor(tail_count)
    fractional = tail_count - whole
    tail_sum = float(np.sum(ordered[-whole:])) if whole else 0.0
    if fractional:
        tail_sum += fractional * float(ordered[-whole - 1])
    expected_shortfall = float(tail_sum / tail_count)
    return value_at_risk, expected_shortfall


def payment_risk_metrics(
    terminal_rates: np.ndarray,
    *,
    invoice_usd: float,
    reference_rate_lkr_per_usd: float,
    budget_lkr: float,
    level: float = 0.95,
) -> PaymentRiskMetrics:
    """Convert independent terminal rate paths into importer LKR risk."""
    rates = np.asarray(terminal_rates, dtype=float)
    if rates.ndim != 1 or rates.size < 2 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("terminal rates must be at least two finite positive values")
    if not isfinite(invoice_usd) or invoice_usd <= 0:
        raise ValueError("invoice_usd must be finite and positive")
    if not isfinite(reference_rate_lkr_per_usd) or reference_rate_lkr_per_usd <= 0:
        raise ValueError("reference rate must be finite and positive")
    if not isfinite(budget_lkr) or budget_lkr <= 0:
        raise ValueError("budget_lkr must be finite and positive")
    reference_cost = float(invoice_usd * reference_rate_lkr_per_usd)
    costs = invoice_usd * rates
    extra_costs = costs - reference_cost
    if not np.isfinite(costs).all() or not isfinite(reference_cost):
        raise OverflowError("payment costs exceed the finite numeric range")
    var, es = empirical_upper_tail_risk(extra_costs, level=level)
    mean_cost = float(np.mean(costs))
    standard_error = float(np.std(costs, ddof=1) / sqrt(rates.size))
    half_width = NormalDist().inv_cdf(0.975) * standard_error
    p05, p95 = np.quantile(costs, (0.05, 0.95), method="linear")
    return PaymentRiskMetrics(
        n_paths=int(rates.size),
        reference_cost_lkr=reference_cost,
        mean_cost_lkr=mean_cost,
        median_cost_lkr=float(np.median(costs)),
        p05_cost_lkr=float(p05),
        p95_cost_lkr=float(p95),
        mean_extra_cost_lkr=float(np.mean(extra_costs)),
        budget_exceedance_probability=float(np.mean(costs > budget_lkr)),
        var_lkr=var,
        es_lkr=es,
        mean_cost_mc_se_lkr=standard_error,
        mean_cost_mc_ci95_lkr=(mean_cost - half_width, mean_cost + half_width),
    )


def between_run_variability(runs: list[PaymentRiskMetrics]) -> dict[str, dict[str, float]]:
    """Observed spread of estimates from independent seeded replicate batches."""
    if len(runs) < 2:
        raise ValueError("at least two independent runs are required")
    result = {}
    for field in ("mean_cost_lkr", "budget_exceedance_probability", "var_lkr", "es_lkr"):
        values = np.array([getattr(run, field) for run in runs], dtype=float)
        result[field] = {
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result
