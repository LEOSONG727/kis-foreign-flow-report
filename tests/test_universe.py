from __future__ import annotations

import pandas as pd

from kis_flow_bot.universe import is_investable


def row(**changes):
    base = {
        "symbol": "005930",
        "name": "삼성전자",
        "group_code": "ST",
        "preferred": "0",
        "spac": "N",
        "halted": "N",
        "liquidation": "N",
        "managed": "N",
        "market_cap_eok": 1_000_000,
    }
    base.update(changes)
    return pd.Series(base)


def test_common_stock_is_investable():
    assert is_investable(row(), 1_000)


def test_preferred_and_spac_are_excluded():
    assert not is_investable(row(symbol="005935", name="삼성전자우"), 1_000)
    assert not is_investable(row(name="테스트스팩", spac="Y"), 1_000)


def test_market_cap_floor_is_applied():
    assert not is_investable(row(market_cap_eok=999), 1_000)

