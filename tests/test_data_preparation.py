"""Automated validation for Stage 2 data ingestion and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import SOURCE_LABEL, load_cbsl_selling_rates
from src.prepare_data import prepare_data
from src.preprocess import calculate_log_returns, split_periods

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CBSL = PROJECT_ROOT / "Data" / "raw" / "CBSL_USD_TT_Selling.csv"
RAW_COLUMNS = ["Currency", "Date", "Buy Rate", "Sell Rate"]


def write_csv(path: Path, records: list[dict[str, object]]) -> Path:
    pd.DataFrame(records, columns=RAW_COLUMNS).to_csv(path, index=False)
    return path


def test_real_cbsl_file_has_expected_canonical_data_and_audit() -> None:
    quotes, audit = load_cbsl_selling_rates(RAW_CBSL)

    assert list(quotes.columns) == ["date", "usd_lkr_rate", "source"]
    assert len(quotes) == 3381
    assert quotes["date"].is_monotonic_increasing
    assert quotes["date"].is_unique
    assert (quotes["usd_lkr_rate"] > 0).all()
    assert quotes["source"].eq(SOURCE_LABEL).all()
    assert audit["blank_rows_removed"] == 1
    assert audit["duplicate_date_rows_removed"] == 0
    assert audit["date_start"] == "2010-05-07"
    assert audit["date_end"] == "2026-11-19"
    assert len(audit["buy_rate_greater_than_sell_records"]) == 2


def test_uses_sell_rate_and_audits_blank_missing_and_non_usd_rows(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "sample.csv",
        [
            {"Currency": "USD", "Date": "02/01/2020", "Buy Rate": "90", "Sell Rate": "91"},
            {"Currency": "EUR", "Date": "02/01/2020", "Buy Rate": "100", "Sell Rate": "110"},
            {"Currency": "USD", "Date": "03/01/2020", "Buy Rate": "91", "Sell Rate": None},
            {"Currency": None, "Date": None, "Buy Rate": None, "Sell Rate": None},
        ],
    )

    quotes, audit = load_cbsl_selling_rates(path)

    assert quotes["usd_lkr_rate"].tolist() == [91.0]
    assert audit["non_usd_rows_excluded"] == 1
    assert audit["blank_rows_removed"] == 1
    assert audit["usd_rows_missing_sell_rate_removed"] == 1


def test_missing_date_is_audited_and_removed(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "missing_date.csv",
        [
            {"Currency": "USD", "Date": None, "Buy Rate": "90", "Sell Rate": "91"},
            {"Currency": "USD", "Date": "02/01/2020", "Buy Rate": "90", "Sell Rate": "91"},
        ],
    )

    quotes, audit = load_cbsl_selling_rates(path)

    assert len(quotes) == 1
    assert audit["usd_rows_missing_date_removed"] == 1


def test_equal_rate_duplicate_date_is_removed_and_counted(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "duplicate.csv",
        [
            {"Currency": "USD", "Date": "02/01/2020", "Buy Rate": "90", "Sell Rate": "91"},
            {"Currency": "USD", "Date": "02/01/2020", "Buy Rate": "89", "Sell Rate": "91"},
        ],
    )

    quotes, audit = load_cbsl_selling_rates(path)

    assert len(quotes) == 1
    assert audit["duplicate_date_rows_removed"] == 1
    assert audit["duplicate_date_rows"] == ["2020-01-02"]


def test_conflicting_sell_rates_for_same_date_are_rejected(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "conflict.csv",
        [
            {"Currency": "USD", "Date": "02/01/2020", "Buy Rate": "90", "Sell Rate": "91"},
            {"Currency": "USD", "Date": "02/01/2020", "Buy Rate": "91", "Sell Rate": "92"},
        ],
    )

    with pytest.raises(ValueError, match="Conflicting USD selling rates"):
        load_cbsl_selling_rates(path)


def test_invalid_nonblank_rate_is_rejected(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "invalid_rate.csv",
        [{"Currency": "USD", "Date": "02/01/2020", "Buy Rate": "90", "Sell Rate": "not-a-rate"}],
    )

    with pytest.raises(ValueError, match="Invalid nonblank USD selling rates"):
        load_cbsl_selling_rates(path)


def test_returns_use_adjacent_observations_and_preserve_calendar_gaps() -> None:
    quotes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-10"]),
            "usd_lkr_rate": [100.0, 110.0, 121.0],
            "source": [SOURCE_LABEL] * 3,
        }
    )

    returns = calculate_log_returns(quotes)

    assert len(returns) == 2
    assert returns["calendar_gap_days"].tolist() == [2, 7]
    assert np.allclose(returns["log_return"], [np.log(1.1), np.log(1.1)])
    assert returns["previous_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2020-01-03",
    ]


def test_evaluation_first_return_uses_final_calibration_quote() -> None:
    quotes = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-05"]
            ),
            "usd_lkr_rate": [300.0, 303.0, 333.3, 366.63],
            "source": [SOURCE_LABEL] * 4,
        }
    )
    returns = calculate_log_returns(quotes)

    periods = split_periods(quotes, returns)

    assert len(periods["calibration_quotes"]) == 2
    assert len(periods["calibration_returns"]) == 1
    assert len(periods["evaluation_quotes"]) == 2
    assert len(periods["evaluation_returns"]) == 2
    assert periods["evaluation_returns"].iloc[0]["previous_date"] == pd.Timestamp(
        "2024-12-31"
    )


def test_preparation_writes_csvs_report_and_rate_chart(tmp_path: Path) -> None:
    raw = write_csv(
        tmp_path / "stage2.csv",
        [
            {"Currency": "USD", "Date": "30/12/2024", "Buy Rate": "299", "Sell Rate": "300"},
            {"Currency": "USD", "Date": "31/12/2024", "Buy Rate": "302", "Sell Rate": "303"},
            {"Currency": "USD", "Date": "02/01/2025", "Buy Rate": "332", "Sell Rate": "333.3"},
            {"Currency": "USD", "Date": "05/01/2025", "Buy Rate": "365", "Sell Rate": "366.63"},
        ],
    )
    processed = tmp_path / "processed"
    figure = tmp_path / "figures" / "rates.png"

    summary = prepare_data(raw, processed, figure)

    expected_files = {
        "cleaned_usd_lkr.csv",
        "daily_log_returns.csv",
        "calibration_quotes_2019_2024.csv",
        "evaluation_quotes_2025_2026.csv",
        "calibration_returns_2019_2024.csv",
        "evaluation_returns_2025_2026.csv",
        "data_quality_report.json",
    }
    assert expected_files.issubset({path.name for path in processed.iterdir()})
    assert figure.is_file() and figure.stat().st_size > 0
    assert summary["raw_source_unchanged_during_processing"] is True
    assert json.loads((processed / "data_quality_report.json").read_text(encoding="utf-8"))[
        "evaluation"
    ]["first_evaluation_return_uses_prior_quote_date"] == "2024-12-31"
