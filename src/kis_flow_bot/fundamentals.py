"""Point-in-time financials from DART periodic reports.

The universe master's financials are today's numbers; replaying them into
the past leaks the future (the audit found the strongest-looking signal was
exactly that). Here each report is usable only from its receipt date, which
``dart_filings.csv`` records as ``rcept_dt``. Date-only disclosures become
eligible on the next signal date, never on the receipt day.

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

from .dart import DartClient, DartError, DartUnavailable, load_corp_codes, update_filings
from .point_in_time import PIT_VERSION, bind_receipts, backward_asof, local_dates
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
    "pit_source_version",
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

    records: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
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
        if not rcept_no:
            continue
        key = (symbol.zfill(6), str(row.get("bsns_year", "")), str(row.get("reprt_code", "")), str(row.get("fs_div", "")), rcept_no)
        record = records.setdefault(
            key,
            {
                "symbol": key[0],
                "bsns_year": key[1],
                "reprt_code": key[2],
                "fs_div": key[3],
                "rcept_no": rcept_no,
                "available": None,  # Only actual filings may establish availability.
                "pit_source_version": PIT_VERSION,
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
    frame = frame.sort_values("_order").drop_duplicates(["symbol", "bsns_year", "reprt_code", "rcept_no"], keep="first")
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
        if status == "020":
            self.dart.unavailable = True
            raise DartUnavailable("DART daily request budget exhausted; remaining collection deferred")
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
    safe_cache = cached[cached.get("pit_source_version", pd.Series("", index=cached.index)).eq(PIT_VERSION)]
    refresh = {str(year) for year in refresh_years}
    frames = [cached]
    client = FinancialsClient(dart)
    stopped = False
    for year in years:
        for reprt_code in REPORT_CODES:
            # A partial successful batch must not mark an entire report complete.
            have = set(safe_cache.loc[
                safe_cache["bsns_year"].astype(str).eq(str(year))
                & safe_cache["reprt_code"].astype(str).eq(reprt_code), "symbol"
            ]) if str(year) not in refresh else set()
            pending = [code for code in corp_codes if code_to_symbol[code] not in have]
            rows: list[dict[str, Any]] = []
            for start in range(0, len(pending), 100):
                try:
                    rows.extend(client.multi_accounts(pending[start : start + 100], year, reprt_code))
                except DartUnavailable as exc:
                    LOGGER.warning("%s; retaining partial financial cache", exc)
                    stopped = True
                    break
                except (DartError, requests.RequestException, ValueError) as exc:
                    LOGGER.warning("DART financials %s/%s failed: %s", year, reprt_code, exc)
                    if "020" in str(exc):
                        stopped = True
                        break
            parsed = parse_multi_accounts(rows, code_to_symbol)
            LOGGER.info("DART financials %s %s: %d companies", year, REPORT_CODES[reprt_code], len(parsed))
            frames.append(parsed)
            if stopped:
                break
        if stopped:
            break
    nonempty = [frame for frame in frames if not frame.empty]
    combined = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=FIN_COLUMNS)
    if combined.empty:
        combined = pd.DataFrame(columns=FIN_COLUMNS)
    # Preserve each correction separately, and never rewrite an existing
    # receipt's numbers with a later response under the same receipt number.
    for column in FIN_COLUMNS:
        if column not in combined:
            combined[column] = pd.NA
    combined["_safe"] = combined["pit_source_version"].eq(PIT_VERSION).fillna(False)
    combined = combined.sort_values("_safe", ascending=False, kind="stable").drop_duplicates(["symbol", "bsns_year", "reprt_code", "rcept_no"], keep="first")
    combined = bind_receipts(combined, read_csv(data_dir / "dart_filings.csv"))
    combined = combined.sort_values(["symbol", "available"]).reset_index(drop=True)
    write_csv(combined[FIN_COLUMNS], path)
    return combined[FIN_COLUMNS]


def attach_fundamentals(frame: pd.DataFrame, financials: pd.DataFrame, date_column: str = "as_of",
                        *, filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add point-in-time fundamental features using only reports public by each date."""

    result = frame.copy()
    output_columns = (*FUNDAMENTAL_FEATURES, "operating_profit", "net_income", "sales", "roe")
    for column in output_columns:
        result[column] = np.nan
    result["financial_rcept_no"] = pd.NA
    result["financial_available"] = pd.NaT
    if result.empty or financials is None or financials.empty or "pit_size" not in result:
        return result
    fin = bind_receipts(financials, filings) if filings is not None else financials.copy()
    if "_pit_verified" not in fin or "pit_source_version" not in fin:
        return result  # Never trust an unbound legacy 'available' value.
    fin = fin[fin["_pit_verified"].fillna(False).eq(True) & fin["pit_source_version"].eq(PIT_VERSION)].copy()
    if fin.empty:
        return result
    fin["symbol"] = fin["symbol"].astype(str).str.zfill(6)
    fin["available"] = pd.to_datetime(fin["available"], format="mixed", errors="coerce")
    for column in ("revenue", "revenue_prev", "op", "op_prev", "ni", "equity"):
        fin[column] = pd.to_numeric(fin.get(column, pd.Series(np.nan, index=fin.index)), errors="coerce")
    fin = fin.dropna(subset=["available"]).sort_values("available")
    annual = fin[fin["reprt_code"].astype(str) == "11011"][["symbol", "available", "op", "ni", "equity"]].rename(
        columns={"op": "op_annual", "ni": "ni_annual", "equity": "equity_annual", "available": "annual_available"}
    )

    base = result[["symbol", date_column, "pit_size"]].copy()
    base["_row"] = np.arange(len(base))
    base["_date"] = local_dates(base[date_column])
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    base = base.sort_values("_date")
    fin = fin.assign(available=fin["available"].astype("datetime64[ns]"))
    latest = backward_asof(base, fin[["symbol", "available", "rcept_no", "revenue", "revenue_prev", "op", "op_prev", "equity"]], date_column=date_column)
    annual = annual.assign(annual_available=annual["annual_available"].astype("datetime64[ns]")).sort_values("annual_available")
    latest = backward_asof(latest, annual, date_column=date_column, available_column="annual_available")
    mcap = np.exp(pd.to_numeric(latest["pit_size"], errors="coerce"))
    latest["op_yoy_mcap"] = (latest["op"] - latest["op_prev"]) / mcap
    latest["op_yield"] = latest["op_annual"] / mcap
    latest["book_to_market"] = latest["equity"] / mcap
    prev = latest["revenue_prev"].where(latest["revenue_prev"].abs() > 0)
    latest["revenue_yoy"] = (latest["revenue"] / prev - 1.0).clip(-1.0, 5.0)
    latest["report_age"] = (latest["_date"] - latest["available"]).dt.days
    latest["operating_profit"] = latest["op"] / 100_000_000.0
    latest["net_income"] = latest["ni_annual"] / 100_000_000.0
    latest["sales"] = latest["revenue"] / 100_000_000.0
    latest["roe"] = latest["ni_annual"] / latest["equity_annual"].where(latest["equity_annual"] > 0) * 100.0
    latest = latest.sort_values("_row")
    for column in output_columns:
        result[column] = latest[column].to_numpy()
    result["financial_rcept_no"] = latest["rcept_no"].to_numpy()
    result["financial_available"] = latest["available"].to_numpy()
    return result


def load_financials(data_dir: Path) -> pd.DataFrame:
    return bind_receipts(read_csv(data_dir / "dart_financials.csv"), read_csv(data_dir / "dart_filings.csv"))


def refresh_financials_from_env(settings, symbols: Iterable[str], history_start: date) -> pd.DataFrame:
    """Fetch every year the stored history touches (plus one prior for YoY)."""

    key = os.getenv("DART_API_KEY", "")
    dart = DartClient(key) if key else None
    today = date.today()
    years = range(history_start.year - 1, today.year + 1)
    symbols = list(symbols)
    financials = update_financials(dart, settings.data_dir, symbols, years, refresh_years=(today.year - 1, today.year))
    # Fetch actual receipts; prefix dates may narrow the download range but are
    # never a fallback availability date. Missing evidence stays unavailable.
    if dart is not None and not dart.unavailable and not financials.empty:
        update_filings(dart, settings.data_dir,
                       [(str(symbol), date(history_start.year - 1, 1, 1), today) for symbol in symbols],
                       max_calls=5000)
    return load_financials(settings.data_dir)
