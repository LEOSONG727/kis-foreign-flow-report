from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from typing import Iterable

import pandas as pd
import requests

from .config import ScreeningConfig

LOGGER = logging.getLogger(__name__)

MASTER_URLS = {
    "KOSPI": "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip",
    "KOSDAQ": "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip",
}

KOSPI_WIDTHS = [
    2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 5, 5, 1, 1, 1, 2, 1, 1,
    1, 2, 2, 2, 3, 1, 3, 12, 12, 8, 15, 21, 2, 7, 1, 1, 1, 1, 1, 9,
    9, 9, 5, 9, 8, 9, 3, 1, 1, 1,
]

KOSDAQ_WIDTHS = [
    2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3,
    1, 3, 12, 12, 8, 15, 21, 2, 7, 1, 1, 1, 1, 9, 9, 9, 5, 9, 8,
    9, 3, 1, 1, 1,
]


def _split_fixed(raw: bytes, widths: Iterable[int]) -> list[str]:
    fields: list[str] = []
    cursor = 0
    for width in widths:
        fields.append(raw[cursor : cursor + width].decode("cp949", errors="ignore").strip())
        cursor += width
    return fields


def _number(value: str) -> float:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _normalize_code(raw_code: bytes) -> str:
    text = raw_code.decode("ascii", errors="ignore").strip()
    digits = "".join(re.findall(r"\d", text))
    return digits[-6:] if len(digits) >= 6 else ""


def _row_from_fields(
    raw: bytes, market: str, suffix_len: int, widths: list[int]
) -> dict | None:
    raw = raw.rstrip(b"\r\n")
    if len(raw) < 21 + suffix_len:
        return None
    code = _normalize_code(raw[:9])
    if len(code) != 6:
        return None
    name = raw[21:-suffix_len].decode("cp949", errors="ignore").strip()
    fields = _split_fixed(raw[-suffix_len:], widths)

    if market == "KOSPI":
        indexes = {
            "group_code": 0,
            "sector_code": 2,
            "low_liquidity": 6,
            "spac": 19,
            "halted": 34,
            "liquidation": 35,
            "managed": 36,
            "warning": 37,
            "listed_shares": 50,
            "preferred": 54,
            "sales": 59,
            "operating_profit": 60,
            "net_income": 62,
            "roe": 63,
            "financial_base_ym": 64,
            "market_cap_eok": 65,
        }
    else:
        indexes = {
            "group_code": 0,
            "sector_code": 2,
            "low_liquidity": 6,
            "spac": 14,
            "halted": 29,
            "liquidation": 30,
            "managed": 31,
            "warning": 32,
            "listed_shares": 45,
            "preferred": 49,
            "sales": 53,
            "operating_profit": 54,
            "net_income": 56,
            "roe": 57,
            "financial_base_ym": 58,
            "market_cap_eok": 59,
        }

    row = {"symbol": code, "name": name, "market": market}
    for key, index in indexes.items():
        row[key] = fields[index] if index < len(fields) else ""
    for key in (
        "listed_shares",
        "sales",
        "operating_profit",
        "net_income",
        "roe",
        "market_cap_eok",
    ):
        row[key] = _number(row[key])
    return row


def parse_master(content: bytes, market: str) -> list[dict]:
    if market == "KOSPI":
        # 공식 예제의 228에는 줄바꿈 1바이트가 포함된다.
        suffix_len, widths = sum(KOSPI_WIDTHS), KOSPI_WIDTHS
    elif market == "KOSDAQ":
        suffix_len, widths = sum(KOSDAQ_WIDTHS), KOSDAQ_WIDTHS
    else:
        raise ValueError(f"Unsupported market: {market}")
    rows = []
    for raw in content.splitlines():
        row = _row_from_fields(raw, market, suffix_len, widths)
        if row:
            rows.append(row)
    return rows


def _download_zip(session: requests.Session, url: str, timeout: int = 60) -> bytes:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        names = archive.namelist()
        if not names:
            raise RuntimeError(f"Empty KIS master archive: {url}")
        return archive.read(names[0])


def _flagged(value: object) -> bool:
    return str(value or "").strip().upper() not in {"", "0", "00", "N"}


PREFERRED_NAME = re.compile(r"(?:우|우B|우C|우선주|[0-9]+우)$", re.IGNORECASE)


def is_investable(row: pd.Series, min_market_cap_eok: float) -> bool:
    if row.get("group_code") != "ST":
        return False
    if not str(row.get("symbol", "")).isdigit():
        return False
    if _flagged(row.get("preferred")) or PREFERRED_NAME.search(str(row.get("name", ""))):
        return False
    if _flagged(row.get("spac")) or "스팩" in str(row.get("name", "")):
        return False
    if any(_flagged(row.get(key)) for key in ("halted", "liquidation", "managed")):
        return False
    market_cap = pd.to_numeric(row.get("market_cap_eok"), errors="coerce")
    return pd.notna(market_cap) and float(market_cap) >= min_market_cap_eok


def load_universe(
    config: ScreeningConfig,
    watchlist: tuple[str, ...],
    session: requests.Session | None = None,
) -> pd.DataFrame:
    session = session or requests.Session()
    all_rows: list[dict] = []
    for market, url in MASTER_URLS.items():
        LOGGER.info("Downloading KIS %s master file", market)
        all_rows.extend(parse_master(_download_zip(session, url), market))

    full = pd.DataFrame(all_rows).drop_duplicates("symbol", keep="first")
    if full.empty:
        raise RuntimeError("KIS master files returned no symbols")

    mask = full.apply(is_investable, axis=1, min_market_cap_eok=config.min_market_cap_eok)
    selected = full.loc[mask].copy()
    selected = selected.sort_values("market_cap_eok", ascending=False).head(config.max_universe)

    # 사용자가 지정한 관찰 종목은 시가총액 필터와 무관하게 반드시 포함한다.
    watched = full[full["symbol"].isin(watchlist)]
    selected = (
        pd.concat([selected, watched], ignore_index=True)
        .drop_duplicates("symbol", keep="first")
        .sort_values(["market", "market_cap_eok"], ascending=[True, False])
        .reset_index(drop=True)
    )
    if len(selected) < 100:
        raise RuntimeError(
            f"Only {len(selected)} symbols survived master parsing; refusing to run with a likely schema error"
        )
    LOGGER.info("Universe ready: %d symbols", len(selected))
    return selected
