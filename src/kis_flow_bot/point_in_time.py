"""Conservative date-only availability and receipt joins shared by live/replay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .storage import normalize_identifiers

PIT_VERSION = "receipt-pit-v3"


def local_dates(values: pd.Series) -> pd.Series:
    # Naive calendar dates and YYYYMMDD are accepted; aware timestamps are
    # converted to the Korean trading calendar before their date is taken.
    parsed = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
    return parsed.dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.normalize()


def receipt_calendar(filings: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "rcept_no", "available"]
    if filings.empty or not {"symbol", "rcept_no", "rcept_dt"}.issubset(filings):
        return pd.DataFrame(columns=columns)
    events = normalize_identifiers(filings)[["symbol", "rcept_no", "rcept_dt"]].copy()
    events["available"] = local_dates(events["rcept_dt"])
    events = events.dropna(subset=["symbol", "rcept_no", "available"])
    events = events[events["rcept_no"].ne("")]
    # Inconsistent duplicates must never move a receipt earlier.
    return events.groupby(["symbol", "rcept_no"], as_index=False)["available"].max()


def bind_receipts(rows: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    result = normalize_identifiers(rows).drop(columns=["available", "_pit_verified"], errors="ignore")
    if result.empty or not {"symbol", "rcept_no"}.issubset(result):
        result["available"] = pd.NaT
        result["_pit_verified"] = False
        return result
    calendar = receipt_calendar(filings)
    result = result.merge(calendar, on=["symbol", "rcept_no"], how="left", validate="many_to_one")
    result["available"] = pd.to_datetime(result["available"], errors="coerce")
    result["_pit_verified"] = result["available"].notna()
    return result


def backward_asof(base: pd.DataFrame, events: pd.DataFrame, *, date_column="as_of",
                  available_column="available", by="symbol") -> pd.DataFrame:
    """Preserve rows/index; do not use date-only events on their receipt day.

    Without an intraday filing timestamp we cannot know whether publication
    preceded the report/close. Strict backward matching makes the first
    observed signal date AFTER publication the first eligible date.
    """
    left = normalize_identifiers(base).copy()
    left["_pit_row"] = np.arange(len(left))
    left["_pit_date"] = local_dates(left[date_column])
    right = normalize_identifiers(events).copy()
    payload = [column for column in right if column != by]
    if available_column not in right:
        right[available_column] = pd.NaT
        payload.append(available_column)
    right[available_column] = local_dates(right[available_column])
    right = right.dropna(subset=[available_column, by]).sort_values(available_column, kind="stable")
    valid = left["_pit_date"].notna() & left[by].notna()
    if right.empty or not valid.any():
        joined = left.copy()
        for column in payload:
            joined[column] = pd.NaT if column == available_column else np.nan
    else:
        joined = pd.merge_asof(left.loc[valid].sort_values("_pit_date", kind="stable"), right,
                               left_on="_pit_date", right_on=available_column, by=by,
                               direction="backward", allow_exact_matches=False)
        invalid = left.loc[~valid].copy()
        for column in payload:
            invalid[column] = pd.NaT if column == available_column else np.nan
        if not invalid.empty:
            joined = pd.concat([joined, invalid], ignore_index=True)
    joined = joined.sort_values("_pit_row").drop(columns=["_pit_row", "_pit_date"])
    joined.index = base.index
    return joined


def observable_answers(rows: pd.DataFrame, answers: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Answers are available after BOTH filing receipt and answer capture.

    Date-only historical rows are deliberately delayed until the following
    signal date, including an answer captured late at night in Korea.
    """
    base = rows.copy()
    if "captured_at" not in answers or answers.empty:
        for column in value_columns:
            base[column] = np.nan
        base["answer_available"] = pd.NaT
        return base
    events = answers[["state_hash", "captured_at", *value_columns]].copy()
    events["answer_available"] = local_dates(events.pop("captured_at"))
    events = events.dropna(subset=["answer_available"])
    # Keep the first recorded answer, not a later re-run with hindsight.
    events = events.sort_values("answer_available").drop_duplicates("state_hash", keep="first")
    result = backward_asof(base, events, by="state_hash", available_column="answer_available")
    return result
