"""OpenDART filing titles with a point-in-time cache.

Only the filing list is used: official report names, receipt date, filer and
remark flags. Receipt dates make it possible to rebuild what was public on
any past signal date. Results are cached so replays and daily runs never
re-download the same range.
"""

from __future__ import annotations

import html
import io
import json
import re
import logging
import os
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable
from xml.etree import ElementTree

import pandas as pd
import requests

from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://opendart.fss.or.kr/api"
FILING_COLUMNS = [
    "symbol",
    "corp_code",
    "corp_name",
    "rcept_no",
    "rcept_dt",
    "report_nm",
    "flr_nm",
    "rm",
]


class DartError(RuntimeError):
    pass


class DartUnavailable(DartError):
    """Stop collection while preserving successfully cached data."""


def xml_to_text(markup: str) -> str:
    """Strip DART XML/HTML tags and collapse whitespace."""

    text = re.sub(r"<[^>]+>", " ", markup)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class DartClient:
    def __init__(
        self,
        api_key: str,
        session: Any | None = None,
        pause: Callable[[float], None] = time.sleep,
        min_interval: float = 0.12,
        max_attempts: int = 3,
        failure_limit: int = 3,
    ):
        if not api_key:
            raise ValueError("DART_API_KEY is required")
        self.api_key = api_key.strip()
        self.session = session or requests.Session()
        self.pause = pause
        self.min_interval = min_interval
        self.calls = 0
        self.max_attempts = max(1, max_attempts)
        self.failure_limit = max(1, failure_limit)
        self.consecutive_failures = 0
        self.unavailable = False

    @classmethod
    def from_env(cls) -> "DartClient":
        return cls(os.getenv("DART_API_KEY", ""))

    def _get(self, path: str, params: dict[str, Any]) -> requests.Response:
        if self.unavailable:
            raise DartUnavailable("DART collection paused after repeated service failures")
        error = ""
        for attempt in range(self.max_attempts):
            self.calls += 1
            try:
                response = self.session.get(
                    f"{BASE_URL}/{path}",
                    params={"crtfc_key": self.api_key, **params},
                    timeout=30,
                )
            except requests.RequestException as exc:
                # Request exception URLs can contain the API key.
                error = f"DART {path} transport failure ({type(exc).__name__})"
            else:
                if response.status_code in (401, 403, 429):
                    self.unavailable = True
                    raise DartUnavailable(f"DART {path} HTTP {response.status_code}; collection deferred")
                if response.status_code not in (500, 502, 503, 504):
                    self.consecutive_failures = 0
                    if self.min_interval:
                        self.pause(self.min_interval)
                    return response
                error = f"DART {path} HTTP {response.status_code}"
            if attempt + 1 < self.max_attempts:
                self.pause(2.0 * (attempt + 1))
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_limit:
            self.unavailable = True
            raise DartUnavailable(f"{error}; repeated failures, remaining collection deferred")
        raise DartError(error)

    def document_text(self, rcept_no: str, max_chars: int = 6000) -> str:
        """Plain text of one filing (document.xml returns a zip of DART XML)."""

        response = self._get("document.xml", {"rcept_no": rcept_no})
        if response.status_code != 200:
            raise DartError(f"document HTTP {response.status_code}")
        try:
            archive = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile as exc:
            raise DartError(f"document error: {response.content[:200]!r}") from exc
        parts = []
        for name in sorted(archive.namelist()):
            raw = archive.read(name)
            for encoding in ("utf-8", "cp949"):
                try:
                    parts.append(raw.decode(encoding))
                    break
                except UnicodeDecodeError:
                    continue
        return xml_to_text(" ".join(parts))[:max_chars]

    def corp_codes(self) -> pd.DataFrame:
        """Listed companies: stock code → DART corp code."""

        response = self._get("corpCode.xml", {})
        if response.status_code != 200:
            raise DartError(f"corpCode HTTP {response.status_code}")
        try:
            archive = zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile as exc:
            # Errors come back as JSON/XML text instead of a zip.
            raise DartError(f"corpCode error: {response.content[:200]!r}") from exc
        root = ElementTree.fromstring(archive.read(archive.namelist()[0]))
        rows = []
        for item in root.iter("list"):
            stock = (item.findtext("stock_code") or "").strip()
            if not stock:
                continue
            rows.append(
                {
                    "symbol": stock.zfill(6),
                    "corp_code": (item.findtext("corp_code") or "").strip(),
                    "corp_name": (item.findtext("corp_name") or "").strip(),
                }
            )
        return pd.DataFrame(rows, columns=["symbol", "corp_code", "corp_name"])

    def filings(self, corp_code: str, start: date, end: date) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            response = self._get(
                "list.json",
                {
                    "corp_code": corp_code,
                    "bgn_de": start.strftime("%Y%m%d"),
                    "end_de": end.strftime("%Y%m%d"),
                    "page_no": page,
                    "page_count": 100,
                },
            )
            payload = response.json()
            status = str(payload.get("status", ""))
            if status == "013":  # no data
                return rows
            if status == "020":
                self.unavailable = True
                raise DartUnavailable("DART daily request budget exhausted; remaining collection deferred")
            if status != "000":
                raise DartError(f"list.json {status}: {payload.get('message')}")
            rows.extend(payload.get("list") or [])
            if page >= int(payload.get("total_page") or 1):
                return rows
            page += 1


def _coverage_path(data_dir: Path) -> Path:
    return data_dir / "dart_coverage.json"


def load_corp_codes(client: DartClient | None, data_dir: Path) -> pd.DataFrame:
    path = data_dir / "dart_corp_codes.csv"
    cached = read_csv(path)
    if not cached.empty:
        cached["symbol"] = cached["symbol"].astype(str).str.zfill(6)
        return cached
    if client is None:
        return pd.DataFrame(columns=["symbol", "corp_code", "corp_name"])
    codes = client.corp_codes()
    write_csv(codes, path)
    return codes


def update_filings(
    client: DartClient | None,
    data_dir: Path,
    needs: Iterable[tuple[str, date, date]],
    max_calls: int = 5000,
) -> pd.DataFrame:
    """Make sure each (symbol, start, end) range is cached; return all filings."""

    path = data_dir / "dart_filings.csv"
    filings = read_csv(path)
    if filings.empty:
        filings = pd.DataFrame(columns=FILING_COLUMNS)
    try:
        coverage: dict[str, list[str]] = json.loads(_coverage_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        coverage = {}
    if client is None:
        return _normalize(filings)
    codes = load_corp_codes(client, data_dir).set_index("symbol")
    new_rows: list[dict[str, Any]] = []
    fetched = 0
    for symbol, start, end in needs:
        symbol = str(symbol).zfill(6)
        have = coverage.get(symbol)
        if have and have[0] <= start.isoformat() and have[1] >= end.isoformat():
            continue
        if symbol not in codes.index:
            coverage[symbol] = [start.isoformat(), end.isoformat()]
            continue
        if client.calls >= max_calls:
            LOGGER.warning("DART call budget reached; remaining symbols deferred")
            break
        # Fetch the union of the old and new range so coverage stays contiguous.
        lo = min(start, date.fromisoformat(have[0])) if have else start
        hi = max(end, date.fromisoformat(have[1])) if have else end
        corp = codes.loc[symbol]
        try:
            rows = client.filings(str(corp["corp_code"]).zfill(8), lo, hi)
        except DartUnavailable as exc:
            LOGGER.warning("%s; keeping filing cache and incomplete coverage", exc)
            break
        except (DartError, requests.RequestException, ValueError) as exc:
            LOGGER.warning("DART filings failed for %s: %s", symbol, exc)
            if "020" in str(exc):  # daily request limit
                break
            continue
        for row in rows:
            new_rows.append({**{key: row.get(key, "") for key in FILING_COLUMNS}, "symbol": symbol})
        coverage[symbol] = [lo.isoformat(), hi.isoformat()]
        fetched += 1
        if fetched % 50 == 0:
            LOGGER.info("DART filings fetched for %d symbols (%d calls)", fetched, client.calls)
    if new_rows:
        filings = pd.concat([filings, pd.DataFrame(new_rows, columns=FILING_COLUMNS)], ignore_index=True)
    filings = _normalize(filings)
    write_csv(filings, path)
    _coverage_path(data_dir).write_text(json.dumps(coverage, sort_keys=True) + "\n", encoding="utf-8")
    return filings


def _normalize(filings: pd.DataFrame) -> pd.DataFrame:
    if filings.empty:
        return pd.DataFrame(columns=FILING_COLUMNS)
    frame = filings.copy()
    for column in FILING_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["rcept_no"] = frame["rcept_no"].astype(str)
    frame["rcept_dt"] = pd.to_datetime(frame["rcept_dt"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["rcept_dt"]).drop_duplicates(["symbol", "rcept_no"], keep="last")
    for column in ("corp_name", "report_nm", "flr_nm", "rm"):
        frame[column] = frame[column].fillna("").astype(str)
    frame = frame.sort_values(["symbol", "rcept_dt", "rcept_no"]).reset_index(drop=True)
    frame["rcept_dt"] = frame["rcept_dt"].dt.strftime("%Y%m%d")
    return frame[FILING_COLUMNS]


def window_needs(rows: pd.DataFrame, lookback_days: int) -> list[tuple[str, date, date]]:
    """One contiguous fetch range per symbol covering every requested window."""

    if rows.empty:
        return []
    frame = rows[["symbol", "as_of"]].copy()
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    grouped = frame.groupby(frame["symbol"].astype(str).str.zfill(6))["as_of"].agg(["min", "max"])
    return [
        (symbol, (row["min"] - pd.Timedelta(days=lookback_days)).date(), row["max"].date())
        for symbol, row in grouped.iterrows()
    ]
