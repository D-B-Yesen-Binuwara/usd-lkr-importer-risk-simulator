"""Return construction and time-ordered period preparation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CALIBRATION_START = pd.Timestamp("2019-01-01")
CALIBRATION_END = pd.Timestamp("2024-12-31")
EVALUATION_START = pd.Timestamp("2025-01-01")
EVALUATION_END = pd.Timestamp("2026-12-31")


def calculate_log_returns(quotes: pd.DataFrame) -> pd.DataFrame:
    """Calculate log changes between adjacent observed quotes.

    Missing calendar dates are not filled or interpolated. ``calendar_gap_days``
    records the number of days spanned by each quote-to-quote return so that
    weekends, holidays, and longer data gaps remain visible.
    """
    required = {"date", "usd_lkr_rate", "source"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Quotes are missing required columns: {sorted(missing)}")

    ordered = quotes.loc[:, ["date", "usd_lkr_rate", "source"]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered["usd_lkr_rate"] = pd.to_numeric(
        ordered["usd_lkr_rate"], errors="raise"
    ).astype(float)
    if ordered["date"].duplicated().any():
        raise ValueError("Quotes must have unique dates before return calculation.")
    if ordered["usd_lkr_rate"].isna().any() or (ordered["usd_lkr_rate"] <= 0).any():
        raise ValueError("USD/LKR rates must be present and strictly positive.")

    ordered = ordered.sort_values("date", kind="stable").reset_index(drop=True)
    previous_date = ordered["date"].shift(1)
    previous_rate = ordered["usd_lkr_rate"].shift(1)
    result = pd.DataFrame(
        {
            "date": ordered["date"],
            "previous_date": previous_date,
            "calendar_gap_days": (ordered["date"] - previous_date).dt.days,
            "usd_lkr_rate": ordered["usd_lkr_rate"],
            "previous_usd_lkr_rate": previous_rate,
            "log_return": np.log(ordered["usd_lkr_rate"] / previous_rate),
            "source": ordered["source"],
        }
    )
    result = result.iloc[1:].reset_index(drop=True)
    result["calendar_gap_days"] = result["calendar_gap_days"].astype(int)
    result["period"] = np.select(
        [
            result["date"].between(CALIBRATION_START, CALIBRATION_END),
            result["date"].between(EVALUATION_START, EVALUATION_END),
        ],
        ["calibration", "evaluation"],
        default="outside_requested_periods",
    )
    return result[
        [
            "date",
            "previous_date",
            "calendar_gap_days",
            "usd_lkr_rate",
            "previous_usd_lkr_rate",
            "log_return",
            "source",
            "period",
        ]
    ]


def split_periods(
    quotes: pd.DataFrame, returns: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Return quote and return frames for calibration and evaluation windows.

    Calibration returns require both observations to fall in 2019–2024.
    Evaluation returns are selected by the current quote date in 2025–2026, so
    the first evaluation return can use the final available 2024 quote as its
    prior observation. This is an explicit one-step boundary calculation, not
    a model fit using evaluation data.
    """
    quote_dates = pd.to_datetime(quotes["date"], errors="raise")
    return_dates = pd.to_datetime(returns["date"], errors="raise")
    previous_dates = pd.to_datetime(returns["previous_date"], errors="raise")

    calibration_quotes = quotes.loc[
        quote_dates.between(CALIBRATION_START, CALIBRATION_END)
    ].copy()
    evaluation_quotes = quotes.loc[
        quote_dates.between(EVALUATION_START, EVALUATION_END)
    ].copy()
    calibration_returns = returns.loc[
        return_dates.between(CALIBRATION_START, CALIBRATION_END)
        & previous_dates.between(CALIBRATION_START, CALIBRATION_END)
    ].copy()
    evaluation_returns = returns.loc[
        return_dates.between(EVALUATION_START, EVALUATION_END)
    ].copy()

    if calibration_quotes.empty:
        raise ValueError("No quote observations fall in the 2019–2024 calibration period.")
    if evaluation_quotes.empty:
        raise ValueError("No quote observations fall in the 2025–2026 evaluation period.")
    if calibration_returns.empty or evaluation_returns.empty:
        raise ValueError("At least one requested period has no quote-to-quote returns.")

    return {
        "calibration_quotes": calibration_quotes.reset_index(drop=True),
        "evaluation_quotes": evaluation_quotes.reset_index(drop=True),
        "calibration_returns": calibration_returns.reset_index(drop=True),
        "evaluation_returns": evaluation_returns.reset_index(drop=True),
    }


def gap_summary(quotes: pd.DataFrame) -> dict[str, Any]:
    """Summarize calendar gaps between available source observations."""
    dates = pd.to_datetime(quotes["date"], errors="raise").sort_values()
    gaps = dates.diff().dt.days.dropna()
    return {
        "gap_count": int(len(gaps)),
        "maximum_calendar_gap_days": int(gaps.max()) if len(gaps) else None,
        "gaps_longer_than_three_calendar_days": int((gaps > 3).sum()),
    }


def return_summary(returns: pd.DataFrame) -> dict[str, Any]:
    """Summarize a return frame without fitting a risk model."""
    values = pd.to_numeric(returns["log_return"], errors="raise")
    return {
        "return_count": int(values.count()),
        "mean_log_return_per_observed_quote_step": float(values.mean()),
        "sample_std_log_return_per_observed_quote_step": float(values.std(ddof=1)),
        "minimum_log_return": float(values.min()),
        "maximum_log_return": float(values.max()),
    }
