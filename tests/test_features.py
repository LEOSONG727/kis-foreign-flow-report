from __future__ import annotations

import pandas as pd

from kis_flow_bot.features import (
    add_cross_sectional_residuals,
    build_base_features,
    normalize_investor_rows,
)


def test_kis_thousand_won_amount_is_normalized_to_won():
    rows = [
        {
            "stck_bsop_date": "20260814",
            "stck_oprc": "100000",
            "stck_hgpr": "101000",
            "stck_lwpr": "99000",
            "stck_clpr": "100000",
            "acml_vol": "1000000",
            "acml_tr_pbmn": "100000000000",
            "frgn_ntby_qty": "1000",
            "frgn_ntby_tr_pbmn": "100000",
            "orgn_ntby_qty": "-500",
            "orgn_ntby_tr_pbmn": "-50000",
        }
    ]
    frame = normalize_investor_rows("005930", "삼성전자", "KOSPI", rows)
    assert frame.iloc[0]["foreign_net_amount"] == 100_000_000
    assert frame.iloc[0]["institution_net_amount"] == -50_000_000


def _history(symbol: str, name: str, foreign_daily: float, start: float, end: float):
    dates = pd.bdate_range("2026-02-02", periods=120)
    closes = pd.Series(
        [start + (end - start) * index / (len(dates) - 1) for index in range(len(dates))]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "market": "KOSPI",
            "name": name,
            "open": closes * 1.002,
            "high": closes * 1.015,
            "low": closes * 0.985,
            "close": closes,
            "volume": 100_000,
            "turnover": 10_000_000_000,
            "foreign_net_qty": foreign_daily / closes,
            "foreign_net_amount": foreign_daily,
            "institution_net_qty": 0,
            "institution_net_amount": 0,
        }
    )


def synthetic_features():
    history = pd.concat(
        [
            _history("000010", "매집후보", 750_000_000, 100_000, 94_000),
            _history("000020", "시장대조", 0, 100_000, 106_000),
        ],
        ignore_index=True,
    )
    universe = pd.DataFrame(
        [
            {
                "symbol": "000010", "name": "매집후보", "market": "KOSPI",
                "sector_code": "0010", "market_cap_eok": 50000, "sales": 1000,
                "operating_profit": 150, "net_income": 100, "roe": 15,
                "financial_base_ym": "202512",
            },
            {
                "symbol": "000020", "name": "시장대조", "market": "KOSPI",
                "sector_code": "0020", "market_cap_eok": 50000, "sales": 1000,
                "operating_profit": 100, "net_income": 80, "roe": 10,
                "financial_base_ym": "202512",
            },
        ]
    )
    return add_cross_sectional_residuals(build_base_features(history, universe))


def test_price_decline_with_persistent_foreign_buying_is_detected():
    features = synthetic_features()
    candidate = features[features["symbol"] == "000010"].iloc[0]
    assert candidate["price_return20"] < 0
    assert candidate["foreign_20_adtv"] > 1.0
    assert candidate["tolerant_buy_ratio20"] >= 0.9
    assert candidate["flow_residual20"] > 0

