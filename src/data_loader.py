"""Load and validate the Central Bank of Sri Lanka TT selling-rate CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_LABEL = "CBSL USD TT Selling Rate"
REQUIRED_COLUMNS = ("Currency", "Date", "Buy Rate", "Sell Rate")


def _as_text(series: pd.Series) -> pd.Series:
    """Normalize CSV text fields while preserving missing values."""
    return series.astype("string").str.strip()


def load_cbsl_selling_rates(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read CBSL rates and return sorted canonical quotes plus an audit record.

    The modelling series uses only the USD ``Sell Rate``. Completely blank
    spacer rows, rows for other currencies, missing dates, missing selling
    rates, and non-conflicting duplicate dates are counted in the audit. A
    malformed nonblank date/rate or a date with conflicting selling rates is
    rejected rather than silently repaired.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CBSL source CSV was not found: {csv_path}")

    raw = pd.read_csv(csv_path, dtype="string", skip_blank_lines=False)
    raw.columns = [str(column).strip() for column in raw.columns]
    if tuple(raw.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            "Unexpected CBSL CSV columns. Expected "
            f"{list(REQUIRED_COLUMNS)!r}; found {list(raw.columns)!r}."
        )

    blank_rows = raw.isna().all(axis=1)
    nonblank = raw.loc[~blank_rows].copy()
    nonblank["_source_row"] = nonblank.index + 2  # account for the header row
    currency = _as_text(nonblank["Currency"]).str.upper()

    unclassified = currency.isna() | currency.eq("")
    if unclassified.any():
        rows = nonblank.loc[unclassified, "_source_row"].astype(int).tolist()
        raise ValueError(f"Nonblank rows have no currency code at CSV rows {rows}.")

    usd_mask = currency.eq("USD")
    usd = nonblank.loc[usd_mask].copy()
    usd["_currency"] = currency.loc[usd_mask]
    other_currency_counts = (
        currency.loc[~usd_mask].value_counts(dropna=False).astype(int).to_dict()
    )

    audit: dict[str, Any] = {
        "input_file": csv_path.name,
        "raw_rows_including_blank_rows": int(len(raw)),
        "blank_rows_removed": int(blank_rows.sum()),
        "nonblank_rows": int(len(nonblank)),
        "usd_rows_before_validation": int(len(usd)),
        "non_usd_rows_excluded": int((~usd_mask).sum()),
        "non_usd_currency_counts": other_currency_counts,
        "usd_rows_missing_date_removed": 0,
        "usd_rows_missing_sell_rate_removed": 0,
        "duplicate_date_rows_removed": 0,
        "buy_rate_missing_or_non_numeric": 0,
        "buy_rate_nonpositive": 0,
        "buy_rate_greater_than_sell_records": [],
    }

    if usd.empty:
        raise ValueError("The source CSV contains no USD observations.")

    date_text = _as_text(usd["Date"])
    missing_date = date_text.isna() | date_text.eq("")
    audit["usd_rows_missing_date_removed"] = int(missing_date.sum())
    usd = usd.loc[~missing_date].copy()
    date_text = date_text.loc[~missing_date]
    dates = pd.to_datetime(date_text, format="%d/%m/%Y", errors="coerce")
    invalid_date = dates.isna()
    if invalid_date.any():
        rows = usd.loc[invalid_date, "_source_row"].astype(int).tolist()
        values = date_text.loc[invalid_date].tolist()
        raise ValueError(
            f"Invalid nonblank date values at CSV rows {rows}: {values!r}."
        )

    sell_text = _as_text(usd["Sell Rate"])
    missing_sell = sell_text.isna() | sell_text.eq("")
    audit["usd_rows_missing_sell_rate_removed"] = int(missing_sell.sum())
    keep = ~missing_sell
    usd = usd.loc[keep].copy()
    dates = dates.loc[keep]
    sell_text = sell_text.loc[keep]
    sell_rate = pd.to_numeric(sell_text, errors="coerce")
    invalid_sell = sell_rate.isna()
    if invalid_sell.any():
        rows = usd.loc[invalid_sell, "_source_row"].astype(int).tolist()
        values = sell_text.loc[invalid_sell].tolist()
        raise ValueError(
            f"Invalid nonblank USD selling rates at CSV rows {rows}: {values!r}."
        )
    nonpositive_sell = sell_rate <= 0
    if nonpositive_sell.any():
        rows = usd.loc[nonpositive_sell, "_source_row"].astype(int).tolist()
        raise ValueError(f"USD selling rates must be positive; bad CSV rows: {rows}.")

    buy_text = _as_text(usd["Buy Rate"])
    missing_buy = buy_text.isna() | buy_text.eq("")
    buy_rate = pd.to_numeric(buy_text, errors="coerce")
    invalid_buy = missing_buy | buy_rate.isna()
    audit["buy_rate_missing_or_non_numeric"] = int(invalid_buy.sum())
    audit["buy_rate_nonpositive"] = int((buy_rate.notna() & (buy_rate <= 0)).sum())

    working = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "usd_lkr_rate": sell_rate.astype(float).to_numpy(),
            "buy_rate": buy_rate.astype(float).to_numpy(),
            "_source_row": usd["_source_row"].astype(int).to_numpy(),
        }
    )

    conflicting = working.groupby("date", sort=False)["usd_lkr_rate"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        detail = {
            str(date.date()): working.loc[
                working["date"].eq(date), ["usd_lkr_rate", "_source_row"]
            ].to_dict(orient="records")
            for date in conflicting.index
        }
        raise ValueError(
            "Conflicting USD selling rates exist for duplicate dates: "
            f"{detail!r}. Resolve against the source before modelling."
        )

    duplicated = working["date"].duplicated(keep="first")
    audit["duplicate_date_rows_removed"] = int(duplicated.sum())
    audit["duplicate_date_rows"] = (
        working.loc[working["date"].duplicated(keep=False), "date"]
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .tolist()
    )

    spread_inversions = working.loc[
        working["buy_rate"].notna()
        & (working["buy_rate"] > working["usd_lkr_rate"]),
        ["date", "buy_rate", "usd_lkr_rate", "_source_row"],
    ]
    audit["buy_rate_greater_than_sell_records"] = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "buy_rate": float(row["buy_rate"]),
            "sell_rate": float(row["usd_lkr_rate"]),
            "csv_row": int(row["_source_row"]),
        }
        for _, row in spread_inversions.iterrows()
    ]

    # Keep the first source row for equal-selling-rate duplicate dates; the
    # modelling value is identical, and the removal is explicitly audited.
    quotes = working.loc[~duplicated, ["date", "usd_lkr_rate"]].copy()
    quotes = quotes.sort_values("date", kind="stable").reset_index(drop=True)
    quotes["source"] = SOURCE_LABEL
    quotes["date"] = pd.to_datetime(quotes["date"])

    audit["observation_count"] = int(len(quotes))
    audit["date_start"] = quotes["date"].min().strftime("%Y-%m-%d")
    audit["date_end"] = quotes["date"].max().strftime("%Y-%m-%d")
    audit["selling_rate_min"] = float(quotes["usd_lkr_rate"].min())
    audit["selling_rate_max"] = float(quotes["usd_lkr_rate"].max())
    audit["weekend_quote_count"] = int((quotes["date"].dt.day_of_week >= 5).sum())

    return quotes[["date", "usd_lkr_rate", "source"]], audit
