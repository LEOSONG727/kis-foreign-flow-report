"""Point-in-time financials from DART periodic reports.

The universe master's financials are today's numbers; replaying them into
the past leaks the future (the audit found the strongest-looking signal was
exactly that). Here each report is usable only from its receipt date, which
DART encodes in the first eight digits of ``rcept_no``.

Signals (all scaled by market value on the signal date, so they are
comparable across companies):
- ``op_yoy_mcap``: latest report's operating profit minus the same period a
  year earlier. An earnings-change (surprise) proxy; the drift after it is
  one of the most persistent effects documented across markets.
- ``op_yield``: latest annual operating profit / market value.
- ``book_to_market``: latest reported equity / market value.
- ``revenue_yoy``: latest report's revenue growth versus a year earlier.
- ``report_age``: calendar days since that report became public.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .dart import DartClient, DartError, load_corp_codes
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)

REPORT_CODES = {"11013": "1분기", "11012": "반기", "11014": "3분기", "11011": "사업보고서"}
FIN_COLUMNS = [
    "symbol",
    "bsns_year",
    "reprt_code",
    "rcept_no",
    "available",
    "fs_div",
    "revenue",
    "revenue_prev",
    "op",
    "op_prev",
    "ni",
    "equity",
]
FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "op_yoy_mcap",
    "op_yield",
    "book_to_market",
    "revenue_yoy",
    "report_age",
)


def _amount(value: Any) -> float:
    text = str(value or "").replace(",", "").strip()
    if text in {"", "-"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _account(name: str) -> str | None:
    name = str(name or "").replace(" ", "")
    if name.startswith("영업이익"):
        return "op"
    if name.startswith("당기순이익"):
        return "ni"
    if name.startswith("자본총계"):
        return "equity"
    if name in {"매출액", "수익(매출액)", "영업수익"} or name.startswith("매출액"):
        return "revenue"
    return None


def parse_multi_accounts(rows: Iterable[dict[str, Any]], code_to_symbol: dict[str, str]) -> pd.DataFrame:
    """One row per (company, report), consolidated statements preferred."""

    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        kind = _account(row.get("account_nm", ""))
        if kind is None:
            continue
        symbol = str(row.get("stock_code") or "").strip()
        if not symbol:
            symbol = code_to_symbol.get(str(row.get("corp_code", "")).zfill(8), "")
        if not symbol:
            continue
        rcept_no = str(row.get("rcept_no", ""))
        key = (symbol.zfill(6), str(row.get("bsns_year", "")), str(row.get("reprt_code", "")), str(row.get("fs_div", "")))
        record = records.setdefault(
            key,
            {
                "symbol": key[0],
                "bsns_year": key[1],
                "reprt_code": key[2],
                "fs_div": key[3],
                "rcept_no": rcept_no,
                "available": rcept_no[:8],
            },
        )
        record[kind] = _amount(row.get("thstrm_amount"))
        if kind in {"op", "revenue"}:
            record[f"{kind}_prev"] = _amount(row.get("frmtrm_amount"))
    frame = pd.DataFrame(records.values())
    if frame.empty:
        return pd.DataFrame(columns=FIN_COLUMNS)
    for column in FIN_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    # Consolidated (CFS) first, separate (OFS) only when no CFS exists.
    frame["_order"] = frame["fs_div"].map({"CFS": 0, "OFS": 1}).fillna(2)
    frame = frame.sort_values("_order").drop_duplicates(["symbol", "bsns_year", "reprt_code"], keep="first")
    return frame[FIN_COLUMNS].reset_index(drop=True)


class FinancialsClient:
    """Thin wrapper around DartClient for the multi-company key accounts API."""

    def __init__(self, dart: DartClient):
        self.dart = dart

    def multi_accounts(self, corp_codes: list[str], year: int, reprt_code: str) -> list[dict[str, Any]]:
        response = self.dart._get(
            "fnlttMultiAcnt.json",
            {"corp_code": ",".join(corp_codes), "bsns_year": str(year), "reprt_code": reprt_code},
        )
        payload = response.json()
        status = str(payload.get("status", ""))
        if status == "013":
            return []
        if status != "000":
            raise DartError(f"fnlttMultiAcnt {status}: {payload.get('message')}")
        return payload.get("list") or []


def update_financials(
    dart: DartClient | None,
    data_dir: Path,
    symbols: Iterable[str],
    years: Iterable[int],
    refresh_years: Iterable[int] = (),
) -> pd.DataFrame:
    """Cache key accounts for every (year, report) and the given symbols."""

    path = data_dir / "dart_financials.csv"
    cached = read_csv(path)
    if cached.empty:
        cached = pd.DataFrame(columns=FIN_COLUMNS)
    else:
        cached["symbol"] = cached["symbol"].astype(str).str.zfill(6)
        for column in ("bsns_year", "reprt_code", "rcept_no", "available"):
            cached[column] = cached[column].astype(str).str.replace(r"\.0$", "", regex=True)
    if dart is None:
        return cached
    codes = load_corp_codes(dart, data_dir)
    wanted = codes[codes["symbol"].isin({str(s).zfill(6) for s in symbols})]
    code_to_symbol = dict(zip(wanted["corp_code"].astype(str).str.zfill(8), wanted["symbol"]))
    corp_codes = sorted(code_to_symbol)
    fetched_keys = set(zip(cached["bsns_year"].astype(str), cached["reprt_code"].astype(str)))
    refresh = {str(year) for year in refresh_years}
    frames = [cached]
    client = FinancialsClient(dart)
    for year in years:
        for reprt_code in REPORT_CODES:
            if (str(year), reprt_code) in fetched_keys and str(year) not in refresh:
                continue
            rows: list[dict[str, Any]] = []
            for start in range(0, len(corp_codes), 100):
                try:
                    rows.extend(client.multi_accounts(corp_codes[start : start + 100], year, reprt_code))
                except (DartError, requests.RequestException, ValueError) as exc:
                    LOGGER.warning("DART financials %s/%s failed: %s", year, reprt_code, exc)
            parsed = parse_multi_accounts(rows, code_to_symbol)
            LOGGER.info("DART financials %s %s: %d companies", year, REPORT_CODES[reprt_code], len(parsed))
            frames.append(parsed)
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if combined.empty:
        combined = pd.DataFrame(columns=FIN_COLUMNS)
    combined = combined.drop_duplicates(["symbol", "bsns_year", "reprt_code"], keep="last")
    combined = combined.sort_values(["symbol", "available"]).reset_index(drop=True)
    write_csv(combined[FIN_COLUMNS], path)
    return combined[FIN_COLUMNS]


def attach_fundamentals(frame: pd.DataFrame, financials: pd.DataFrame, date_column: str = "as_of") -> pd.DataFrame:
    """Add point-in-time fundamental features using only reports public by each date."""

    result = frame.copy()
    for column in FUNDAMENTAL_FEATURES:
        result[column] = np.nan
    if result.empty or financials is None or financials.empty or "pit_size" not in result:
        return result
    fin = financials.copy()
    fin["symbol"] = fin["symbol"].astype(str).str.zfill(6)
    fin["available"] = pd.to_datetime(fin["available"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    for column in ("revenue", "revenue_prev", "op", "op_prev", "ni", "equity"):
        fin[column] = pd.to_numeric(fin[column], errors="coerce")
    fin = fin.dropna(subset=["available"]).sort_values("available")
    annual = fin[fin["reprt_code"].astype(str) == "11011"][["symbol", "available", "op"]].rename(
        columns={"op": "op_annual", "available": "annual_available"}
    )

    base = result[["symbol", date_column, "pit_size"]].copy()
    base["_row"] = np.arange(len(base))
    base["_date"] = pd.to_datetime(base[date_column]).astype("datetime64[ns]")
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    base = base.sort_values("_date")
    fin = fin.assign(available=fin["available"].astype("datetime64[ns]"))
    latest = pd.merge_asof(
        base,
        fin[["symbol", "available", "revenue", "revenue_prev", "op", "op_prev", "equity"]],
        left_on="_date",
        right_on="available",
        by="symbol",
        direction="backward",
    )
    annual = annual.assign(annual_available=annual["annual_available"].astype("datetime64[ns]")).sort_values("annual_available")
    latest = pd.merge_asof(
        latest.sort_values("_date"),
        annual,
        left_on="_date",
        right_on="annual_available",
        by="symbol",
        direction="backward",
    )
    mcap = np.exp(pd.to_numeric(latest["pit_size"], errors="coerce"))
    latest["op_yoy_mcap"] = (latest["op"] - latest["op_prev"]) / mcap
    latest["op_yield"] = latest["op_annual"] / mcap
    latest["book_to_market"] = latest["equity"] / mcap
    prev = latest["revenue_prev"].where(latest["revenue_prev"].abs() > 0)
    latest["revenue_yoy"] = (latest["revenue"] / prev - 1.0).clip(-1.0, 5.0)
    latest["report_age"] = (latest["_date"] - latest["available"]).dt.days
    latest = latest.sort_values("_row")
    for column in FUNDAMENTAL_FEATURES:
        result[column] = latest[column].to_numpy()
    return result


def load_financials(data_dir: Path) -> pd.DataFrame:
    return read_csv(data_dir / "dart_financials.csv")


def refresh_financials_from_env(settings, symbols: Iterable[str], history_start: date) -> pd.DataFrame:
    """Fetch every year the stored history touches (plus one prior for YoY)."""

    key = os.getenv("DART_API_KEY", "")
    dart = DartClient(key) if key else None
    today = date.today()
    years = range(history_start.year - 1, today.year + 1)
    return update_financials(dart, settings.data_dir, symbols, years, refresh_years=(today.year - 1, today.year))
