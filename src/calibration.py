"""Fit an annual, elapsed-calendar-time GBM to prepared TT selling returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import SOURCE_LABEL

DAYS_PER_YEAR = 365.0
BASELINE_START = "2019-01-01"
BASELINE_END = "2024-12-31"


@dataclass(frozen=True)
class GBMCalibration:
    """Annual GBM coefficients and the training observations that produced them."""

    window_start: str
    window_end: str
    first_quote_date: str
    last_quote_date: str
    n_returns: int
    total_elapsed_days: int
    total_elapsed_years: float
    alpha_per_year: float
    sigma_per_sqrt_year: float
    mu_per_year: float
    last_rate_lkr_per_usd: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def fit_calendar_gbm(
    returns: pd.DataFrame, *, window_start: str, window_end: str
) -> GBMCalibration:
    """Fit r_i ~ N(alpha*d_i/365, sigma^2*d_i/365) by maximum likelihood.

    Only rows with *both* quote dates inside the inclusive window enter the fit.
    The variance divisor is n, not the sample-SD divisor n-1. Other periods can
    be present in ``returns`` but never contribute to the estimates.
    """
    required = {
        "date", "previous_date", "calendar_gap_days", "log_return",
        "usd_lkr_rate", "previous_usd_lkr_rate", "source",
    }
    missing = required.difference(returns.columns)
    if missing:
        raise ValueError(f"Returns are missing required columns: {sorted(missing)}")

    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    if start > end:
        raise ValueError("window_start must not be after window_end")
    dates = pd.to_datetime(returns["date"], errors="raise")
    previous_dates = pd.to_datetime(returns["previous_date"], errors="raise")
    in_window = dates.between(start, end) & previous_dates.between(start, end)
    selected = returns.loc[in_window].copy()
    if len(selected) < 2:
        raise ValueError("At least two within-window returns are required")
    selected["date"] = dates.loc[in_window]
    selected["previous_date"] = previous_dates.loc[in_window]
    selected = selected.sort_values("date", kind="stable").reset_index(drop=True)

    if selected["date"].duplicated().any():
        raise ValueError("Within-window return dates must be unique")
    actual_gaps = (selected["date"] - selected["previous_date"]).dt.days.to_numpy()
    gaps = pd.to_numeric(selected["calendar_gap_days"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(gaps).all()
        or (gaps <= 0).any()
        or not np.equal(gaps, np.floor(gaps)).all()
        or not np.array_equal(gaps, actual_gaps)
    ):
        raise ValueError("Calendar gaps must be positive whole days matching quote dates")
    if not selected["source"].eq(SOURCE_LABEL).all():
        raise ValueError("Calibration requires the CBSL USD TT selling-rate source")

    rates = pd.to_numeric(selected["usd_lkr_rate"], errors="raise").to_numpy(dtype=float)
    previous_rates = pd.to_numeric(
        selected["previous_usd_lkr_rate"], errors="raise"
    ).to_numpy(dtype=float)
    log_returns = pd.to_numeric(selected["log_return"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(rates).all()
        or not np.isfinite(previous_rates).all()
        or (rates <= 0).any()
        or (previous_rates <= 0).any()
        or not np.isfinite(log_returns).all()
    ):
        raise ValueError("Rates must be positive and all calibration values finite")
    if not np.allclose(log_returns, np.log(rates / previous_rates), rtol=1e-10, atol=1e-12):
        raise ValueError("Log returns do not match the supplied selling-rate pairs")

    delta_years = gaps / DAYS_PER_YEAR
    alpha = float(np.sum(log_returns) / np.sum(delta_years))
    residuals = log_returns - alpha * delta_years
    sigma_squared = float(np.mean(np.square(residuals) / delta_years))
    sigma = float(np.sqrt(max(sigma_squared, 0.0)))
    mu = float(alpha + 0.5 * sigma_squared)
    return GBMCalibration(
        window_start=start.date().isoformat(),
        window_end=end.date().isoformat(),
        first_quote_date=selected.loc[0, "previous_date"].date().isoformat(),
        last_quote_date=selected.loc[len(selected) - 1, "date"].date().isoformat(),
        n_returns=len(selected),
        total_elapsed_days=int(np.sum(gaps)),
        total_elapsed_years=float(np.sum(delta_years)),
        alpha_per_year=alpha,
        sigma_per_sqrt_year=sigma,
        mu_per_year=mu,
        last_rate_lkr_per_usd=float(rates[-1]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_calibration_report(project_root: Path) -> dict[str, object]:
    """Build a reproducible 2019–2024 report from the prepared CBSL file."""
    raw_path = project_root / "data" / "raw" / "CBSL_USD_TT_Selling.csv"
    prepared_path = project_root / "data" / "processed" / "calibration_returns_2019_2024.csv"
    quality_path = project_root / "data" / "processed" / "data_quality_report.json"
    raw_hash = _sha256(raw_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality["source_sha256"] != raw_hash:
        raise ValueError("Raw CSV hash does not match the data-quality report")
    calibration = fit_calendar_gbm(
        pd.read_csv(prepared_path),
        window_start=BASELINE_START,
        window_end=BASELINE_END,
    )
    if calibration.n_returns != quality["calibration"]["return_count_both_dates_in_period"]:
        raise ValueError("Calibration return count does not match the data-quality report")
    if calibration.sigma_per_sqrt_year <= 0:
        raise ValueError("Fitted volatility must be positive for stochastic GBM use")
    return {
        "model": "constant-coefficient geometric Brownian motion",
        "source": SOURCE_LABEL,
        "clock": "elapsed calendar days / 365 years",
        "variance_estimator": "Gaussian maximum likelihood; residual divisor n",
        "raw_file": "data/raw/CBSL_USD_TT_Selling.csv",
        "raw_sha256": raw_hash,
        "prepared_file": "data/processed/calibration_returns_2019_2024.csv",
        "prepared_sha256": _sha256(prepared_path),
        "calibration": calibration.to_dict(),
        "limitations": [
            "2022 movements are retained; constant GBM coefficients may be unstable across regimes.",
            "Calendar-time variance scaling across weekends and holidays is an assumption.",
            "The 2025–2026 evaluation snapshot is excluded from this fit.",
        ],
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = baseline_calibration_report(project_root)
    output_path = project_root / "data" / "processed" / "gbm_calibration_2019_2024.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
