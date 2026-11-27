"""Build the cleaned Stage 2 datasets and their auditable summaries.

Run from the project root with ``python -m src.prepare_data``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_cbsl_selling_rates
from src.preprocess import gap_summary, calculate_log_returns, return_summary, split_periods


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("date", "previous_date"):
        if column in output.columns:
            output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d")
    return output


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _make_rate_figure(quotes: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.plot(quotes["date"], quotes["usd_lkr_rate"], color="#185a8d", linewidth=1.25)
    ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31"),
               color="#dcebd9", alpha=0.45, label="Calibration: 2019–2024")
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"),
               color="#f5e5c8", alpha=0.45, label="Evaluation: 2025–2026")
    ax.set_title("CBSL USD TT Selling Rate")
    ax.set_xlabel("Date")
    ax.set_ylabel("LKR per USD")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare_data(
    input_path: str | Path,
    processed_dir: str | Path,
    figure_path: str | Path,
) -> dict[str, Any]:
    """Prepare canonical quotes, adjacent-quote returns, splits, and reports."""
    input_path = Path(input_path)
    processed_dir = Path(processed_dir)
    figure_path = Path(figure_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    source_hash_before = sha256_file(input_path)
    quotes, audit = load_cbsl_selling_rates(input_path)
    returns = calculate_log_returns(quotes)
    periods = split_periods(quotes, returns)

    _csv_ready(quotes).to_csv(processed_dir / "cleaned_usd_lkr.csv", index=False)
    _csv_ready(returns).to_csv(processed_dir / "daily_log_returns.csv", index=False)
    _csv_ready(periods["calibration_quotes"]).to_csv(
        processed_dir / "calibration_quotes_2019_2024.csv", index=False
    )
    _csv_ready(periods["evaluation_quotes"]).to_csv(
        processed_dir / "evaluation_quotes_2025_2026.csv", index=False
    )
    _csv_ready(periods["calibration_returns"]).to_csv(
        processed_dir / "calibration_returns_2019_2024.csv", index=False
    )
    _csv_ready(periods["evaluation_returns"]).to_csv(
        processed_dir / "evaluation_returns_2025_2026.csv", index=False
    )

    _make_rate_figure(quotes, figure_path)
    source_hash_after = sha256_file(input_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("The raw source CSV hash changed during data preparation.")

    calibration = periods["calibration_quotes"]
    evaluation = periods["evaluation_quotes"]
    summary: dict[str, Any] = {
        "source_file": input_path.name,
        "source_sha256": source_hash_after,
        "raw_source_unchanged_during_processing": source_hash_before == source_hash_after,
        "canonical_schema": ["date", "usd_lkr_rate", "source"],
        "source_order_sorted_ascending": bool(quotes["date"].is_monotonic_increasing),
        "unique_quote_dates": bool(quotes["date"].is_unique),
        "all_rates_positive": bool((quotes["usd_lkr_rate"] > 0).all()),
        "date_start": quotes["date"].min().strftime("%Y-%m-%d"),
        "date_end": quotes["date"].max().strftime("%Y-%m-%d"),
        "observation_count": int(len(quotes)),
        "selling_rate_min": float(quotes["usd_lkr_rate"].min()),
        "selling_rate_max": float(quotes["usd_lkr_rate"].max()),
        "overall_calendar_gaps": gap_summary(quotes),
        "calibration": {
            "start": "2019-01-01",
            "end": "2024-12-31",
            "quote_count": int(len(calibration)),
            "return_count_both_dates_in_period": int(len(periods["calibration_returns"])),
            "calendar_gaps": gap_summary(calibration),
            **return_summary(periods["calibration_returns"]),
        },
        "evaluation": {
            "start": "2025-01-01",
            "end": "2026-12-31 (available data through source end date)",
            "quote_count": int(len(evaluation)),
            "return_count_current_quote_in_period": int(len(periods["evaluation_returns"])),
            "first_evaluation_return_uses_prior_quote_date": (
                periods["evaluation_returns"].iloc[0]["previous_date"].strftime("%Y-%m-%d")
            ),
            "calendar_gaps": gap_summary(evaluation),
            **return_summary(periods["evaluation_returns"]),
        },
        "audit": audit,
        "processing": {
            "return_definition": "ln(current observed USD TT selling rate / previous observed USD TT selling rate)",
            "missing_calendar_dates_filled": False,
            "returns_divided_by_calendar_gap_days": False,
            "calibration_boundary_returns_require_both_dates_in_calibration": True,
            "evaluation_boundary_return_uses_previous_observed_quote": True,
        },
        "outputs": {
            "cleaned_quotes": "cleaned_usd_lkr.csv",
            "all_returns": "daily_log_returns.csv",
            "calibration_quotes": "calibration_quotes_2019_2024.csv",
            "evaluation_quotes": "evaluation_quotes_2025_2026.csv",
            "calibration_returns": "calibration_returns_2019_2024.csv",
            "evaluation_returns": "evaluation_returns_2025_2026.csv",
            "rate_chart": str(figure_path),
        },
    }
    _write_json(processed_dir / "data_quality_report.json", summary)
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    summary = prepare_data(
        input_path=project_root / "data" / "raw" / "CBSL_USD_TT_Selling.csv",
        processed_dir=project_root / "data" / "processed",
        figure_path=project_root / "figures" / "usd_lkr_selling_rate.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
