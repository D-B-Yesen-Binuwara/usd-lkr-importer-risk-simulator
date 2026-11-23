"""Reproduce the hypothetical 90-day importer-risk scenario for both models."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from src.calibration import baseline_calibration_report
from src.gbm import simulate_terminal_rates, theoretical_terminal_moments
from src.markov import baseline_markov_report, simulate_markov_paths, weekday_quote_offsets
from src.risk import between_run_variability, payment_risk_metrics

INVOICE_USD = 100_000.0
HORIZON_CALENDAR_DAYS = 90
BUDGET_MARGIN = 0.05
RISK_LEVEL = 0.95
N_PATHS_PER_RUN = 10_000
REPLICATE_SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)


def baseline_risk_report(
    project_root: Path,
    *,
    invoice_usd: float = INVOICE_USD,
    horizon_days: int = HORIZON_CALENDAR_DAYS,
    budget_margin: float = BUDGET_MARGIN,
    level: float = RISK_LEVEL,
    n_paths_per_run: int = N_PATHS_PER_RUN,
    seeds: tuple[int, ...] = REPLICATE_SEEDS,
) -> dict[str, object]:
    """Run independent seeded batches with one frozen training reference."""
    if not np.isfinite(invoice_usd) or invoice_usd <= 0:
        raise ValueError("invoice_usd must be finite and positive")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")
    if not np.isfinite(budget_margin) or budget_margin < 0:
        raise ValueError("budget_margin must be finite and nonnegative")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must lie strictly between zero and one")
    if isinstance(n_paths_per_run, bool) or not isinstance(n_paths_per_run, int) or n_paths_per_run < 2:
        raise ValueError("n_paths_per_run must be an integer of at least two")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct replicate seeds are required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("replicate seeds must be distinct nonnegative integers")

    gbm = baseline_calibration_report(project_root)
    markov = baseline_markov_report(project_root)
    c = gbm["calibration"]
    m = markov["calibration"]
    if (
        c["last_quote_date"] != m["last_quote_date"]
        or c["last_rate_lkr_per_usd"] != m["last_rate_lkr_per_usd"]
        or gbm["raw_sha256"] != markov["raw_sha256"]
        or gbm["prepared_sha256"] != markov["prepared_sha256"]
    ):
        raise ValueError("GBM and Markov reference data disagree")
    reference_date = date.fromisoformat(c["last_quote_date"])
    due_date = reference_date + timedelta(days=horizon_days)
    initial_rate = c["last_rate_lkr_per_usd"]
    reference_cost = invoice_usd * initial_rate
    budget = reference_cost * (1 + budget_margin)
    quote_offsets = weekday_quote_offsets(reference_date.isoformat(), horizon_days)

    model_results = {}
    for name in ("gbm", "markov"):
        batches = []
        run_metrics = []
        run_summaries = []
        for seed in seeds:
            if name == "gbm":
                terminal_rates = simulate_terminal_rates(
                    initial_rate, c["alpha_per_year"], c["sigma_per_sqrt_year"],
                    horizon_days=horizon_days, n_paths=n_paths_per_run, seed=seed,
                )
            else:
                terminal_rates = simulate_markov_paths(
                    initial_rate, m["alpha_per_year"], m["state_volatilities_per_sqrt_year"],
                    m["transition_probabilities"], m["last_state"],
                    horizon_days=horizon_days, quote_day_offsets=quote_offsets,
                    n_paths=n_paths_per_run, seed=seed,
                ).rates[:, -1]
            metrics = payment_risk_metrics(
                terminal_rates, invoice_usd=invoice_usd,
                reference_rate_lkr_per_usd=initial_rate,
                budget_lkr=budget, level=level,
            )
            batches.append(terminal_rates)
            run_metrics.append(metrics)
            run_summaries.append({"seed": seed, "metrics": metrics.to_dict()})

        pooled_rates = np.concatenate(batches)
        pooled = payment_risk_metrics(
            pooled_rates, invoice_usd=invoice_usd,
            reference_rate_lkr_per_usd=initial_rate,
            budget_lkr=budget, level=level,
        )
        model_results[name] = {
            "pooled_metrics": pooled.to_dict(),
            "between_run_variability": between_run_variability(run_metrics),
            "runs": run_summaries,
        }

    analytic_mean_rate, _ = theoretical_terminal_moments(
        initial_rate, c["alpha_per_year"], c["sigma_per_sqrt_year"],
        horizon_days=horizon_days,
    )
    return {
        "scenario": {
            "label": "hypothetical end-2024 importer invoice, not a real payment or forecast",
            "invoice_usd": invoice_usd,
            "reference_date": reference_date.isoformat(),
            "due_date": due_date.isoformat(),
            "horizon_calendar_days": horizon_days,
            "reference_rate_lkr_per_usd": initial_rate,
            "reference_cost_lkr": reference_cost,
            "budget_margin": budget_margin,
            "budget_lkr": budget,
            "risk_level": level,
            "n_paths_per_run": n_paths_per_run,
            "replicate_seeds": seeds,
            "total_paths_per_model": n_paths_per_run * len(seeds),
            "markov_quote_schedule": "Monday-Friday proxy, holidays and settlement unverified",
            "markov_quote_day_offsets": quote_offsets,
        },
        "input_hashes": {
            "raw_sha256": gbm["raw_sha256"],
            "prepared_sha256": gbm["prepared_sha256"],
        },
        "methods": {
            "loss": "signed extra cost L=Q*(S_T-S_0); no zero clamp",
            "var": "empirical left quantile of L at risk_level",
            "es": "quantile-integral upper-tail mean with fractional boundary weighting",
            "cost_percentiles": "NumPy linear interpolation",
            "budget_exceedance": "strictly greater than the fixed LKR budget",
            "mean_cost_ci": "approximate normal 95% Monte Carlo interval using pooled independent paths and sample SD",
            "tail_uncertainty": "sample SD and min/max of VaR and ES across independent seeded replicate batches; not a confidence interval",
            "model_comparison": "same invoice, reference, due date, budget, path count and seed list; RNG draws are not paired across algorithms",
        },
        "gbm_analytic_expected_cost_lkr": invoice_usd * analytic_mean_rate,
        "models": model_results,
        "limitations": [
            "Monte Carlo uncertainty is conditional on historical fitted parameters; it excludes calibration and structural uncertainty.",
            "The Markov Monday-Friday quote schedule ignores actual holidays and settlement conventions.",
            "The 2022 exchange-rate movements strongly affect fitted volatilities.",
            "The 2025-2026 evaluation snapshot is not used for calibration or scenario choice.",
            "Trading fees and importer-specific bank spreads are excluded.",
        ],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = baseline_risk_report(project_root)
    path = project_root / "data" / "processed" / "importer_risk_baseline_2019_2024.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, model in report["models"].items():
        metrics = model["pooled_metrics"]
        print(
            f"{name}: mean LKR {metrics['mean_cost_lkr']:.2f}; "
            f"VaR95 extra LKR {metrics['var_lkr']:.2f}; "
            f"ES95 extra LKR {metrics['es_lkr']:.2f}"
        )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
