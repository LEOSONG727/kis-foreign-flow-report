from __future__ import annotations

import pandas as pd

from kis_flow_bot.features import (
    _cumulative_flow_path,
    _opinion_consensus,
    add_cross_sectional_residuals,
    build_base_features,
    normalize_investor_rows,
    validate_investor_amounts,
)
from kis_flow_bot.scoring import is_steady_accumulation


def test_opinion_consensus_ignores_stale_targets_and_reports_dispersion():
    opinions = pd.DataFrame(
        [
            {"symbol": "000010", "date": "2026-03-01", "target_price": 300_000},
            {"symbol": "000010", "date": "2026-07-20", "target_price": 120_000},
            {"symbol": "000010", "date": "2026-08-10", "target_price": 140_000},
        ]
    )
    target, upside, coverage, recency, dispersion = _opinion_consensus(
        "000010", opinions, 100_000, pd.Timestamp("2026-08-14")
    )
    assert target == 130_000
    assert round(upside, 6) == 0.3
    assert coverage == 2
    assert recency == 4
    assert dispersion > 0


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


def test_kis_million_won_amount_is_normalized_to_won():
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
            "frgn_ntby_tr_pbmn": "100",
            "orgn_ntby_qty": "-500",
            "orgn_ntby_tr_pbmn": "-50",
        }
    ]
    frame = normalize_investor_rows("005930", "삼성전자", "KOSPI", rows)
    assert frame.iloc[0]["foreign_net_amount"] == 100_000_000
    assert frame.iloc[0]["institution_net_amount"] == -50_000_000


def test_bad_amount_scale_is_rejected():
    frame = pd.DataFrame(
        {
            "close": [100_000.0] * 5,
            "foreign_net_qty": [1_000.0] * 5,
            "foreign_net_amount": [100.0] * 5,
        }
    )
    try:
        validate_investor_amounts(frame)
    except ValueError as exc:
        assert "amount unit validation failed" in str(exc)
    else:
        raise AssertionError("mis-scaled foreign-flow amount was accepted")


def test_missing_amounts_are_rejected_when_quantities_exist():
    frame = pd.DataFrame(
        {
            "close": [100_000.0] * 5,
            "foreign_net_qty": [1_000.0] * 5,
            "foreign_net_amount": [0.0] * 5,
        }
    )
    try:
        validate_investor_amounts(frame)
    except ValueError as exc:
        assert "amount coverage validation failed" in str(exc)
    else:
        raise AssertionError("missing foreign-flow amounts were accepted")


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


def test_short_price_path_and_low_rebound_features_are_calculated():
    history = _history("000010", "V자반등", 500_000_000, 100_000, 100_000)
    path = [
        100_000, 95_000, 90_000, 85_000, 80_000, 82_000, 84_000,
        86_000, 88_000, 90_000, 100_000, 100_000, 100_000, 100_000,
        100_000, 100_000, 100_000, 100_000, 100_000, 100_000, 100_000,
    ]
    indexes = history.tail(len(path)).index
    history.loc[indexes, "close"] = path
    history.loc[indexes, "open"] = history.loc[indexes, "close"] * 0.998
    history.loc[indexes, "high"] = history.loc[indexes, "close"] * 1.01
    history.loc[indexes, "low"] = history.loc[indexes, "close"] * 0.99
    history.loc[indexes[:10], "volume"] = 100_000
    history.loc[indexes[10:], "volume"] = 200_000
    universe = pd.DataFrame(
        [{
            "symbol": "000010", "name": "V자반등", "market": "KOSPI",
            "sector_code": "0010", "market_cap_eok": 50_000,
            "sales": 1_000, "operating_profit": 100, "net_income": 80,
            "roe": 12, "financial_base_ym": "202512",
        }]
    )

    row = build_base_features(history, universe).iloc[0]

    assert abs(row["price_return20"]) < 0.001
    assert abs(row["price_return10"]) < 0.001
    assert row["rebound_from_low20"] > 0.20
    assert row["rebound_from_low10"] < 0.03
    assert row["rebound_atr_distance20"] > 0
    assert row["rebound_volume_ratio20"] > 1.0
    assert row["distance_high20"] <= 0
    assert {
        "price_return1", "price_return3", "price_return5", "price_acceleration3",
        "ma5_slope", "above_ma5", "above_ma10", "up_down_volume_ratio10",
        "foreign_acceleration3", "touched_ma20", "near_ma20", "ma20_support",
    }.issubset(row.index)


def test_today_candle_touching_ma20_is_detected():
    history = _history("000010", "이평접촉", 500_000_000, 100_000, 100_000)
    history["close"] = 100_000
    history["open"] = 100_000
    history["high"] = 101_000
    history["low"] = 99_000
    last = history.index[-1]
    history.loc[last, "open"] = 101_500
    history.loc[last, "high"] = 102_000
    history.loc[last, "low"] = 99_500
    history.loc[last, "close"] = 100_200
    universe = pd.DataFrame(
        [{
            "symbol": "000010", "name": "이평접촉", "market": "KOSPI",
            "sector_code": "0010", "market_cap_eok": 50_000,
            "sales": 1_000, "operating_profit": 100, "net_income": 80,
            "roe": 12, "financial_base_ym": "202512",
        }]
    )
    row = build_base_features(history, universe).iloc[0]
    assert bool(row["touched_ma20"])
    assert bool(row["ma20_support"])
    assert abs(row["ma20_distance"]) < 0.02


def _flow_history(daily_amounts: list[float], symbol: str = "000010", name: str = "누적주"):
    dates = pd.bdate_range("2026-02-02", periods=len(daily_amounts))
    close = 100_000.0
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "market": "KOSPI",
            "name": name,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 100_000,
            "turnover": 10_000_000_000,
            "foreign_net_qty": [amount / close for amount in daily_amounts],
            "foreign_net_amount": daily_amounts,
            "institution_net_qty": 0,
            "institution_net_amount": 0,
        }
    )


def test_flow_path_separates_daily_buy_days_from_rolling_inventory_changes():
    staircase = pd.Series([500_000_000.0] * 40)
    burst = pd.Series([0.0] * 37 + [6_666_666_667.0] * 3)
    stair = _cumulative_flow_path(staircase)
    lump = _cumulative_flow_path(burst)
    assert stair["foreign_buy_days10"] == 10
    assert stair["foreign_bought_today"] is True
    assert stair["foreign_buy_streak"] >= 10
    assert stair["foreign_inventory_up_days10"] == 0
    assert lump["foreign_buy_days10"] <= 3
    assert lump["foreign_buy_streak"] <= 3
    assert lump["foreign_inventory_up_days10"] <= 3


def test_steady_accumulation_accepts_staircase_and_rejects_a_same_size_burst():
    universe = pd.DataFrame(
        [{
            "symbol": "000010", "name": "계단", "market": "KOSPI",
            "sector_code": "0010", "market_cap_eok": 50_000,
            "sales": 1_000, "operating_profit": 100, "net_income": 80,
            "roe": 12, "financial_base_ym": "202512",
        }]
    )
    stair_row = build_base_features(_flow_history([800_000_000.0] * 40, name="계단"), universe).iloc[0]
    burst_amounts = [0.0] * 37 + [5_333_333_333.0] * 3
    burst_row = build_base_features(_flow_history(burst_amounts, name="몰빵"), universe).iloc[0]
    assert stair_row["foreign_net20"] >= 10_000_000_000
    assert burst_row["foreign_net20"] >= 10_000_000_000
    assert is_steady_accumulation(stair_row)
    assert not is_steady_accumulation(burst_row)
    assert stair_row["foreign_buy_days10"] >= 8
    assert burst_row["foreign_buy_days10"] < 8


def test_extended_close_above_ma20_is_not_a_touch():
    history = _history("000010", "이평위", 500_000_000, 100_000, 100_000)
    history["close"] = 100_000
    history["open"] = 100_000
    history["high"] = 101_000
    history["low"] = 99_000
    last = history.index[-1]
    history.loc[last, "open"] = 108_000
    history.loc[last, "high"] = 110_000
    history.loc[last, "low"] = 107_000
    history.loc[last, "close"] = 109_000
    universe = pd.DataFrame(
        [{
            "symbol": "000010", "name": "이평위", "market": "KOSPI",
            "sector_code": "0010", "market_cap_eok": 50_000,
            "sales": 1_000, "operating_profit": 100, "net_income": 80,
            "roe": 12, "financial_base_ym": "202512",
        }]
    )
    row = build_base_features(history, universe).iloc[0]
    assert not bool(row["touched_ma20"])
    assert not bool(row["ma20_support"])
    assert row["ma20_distance"] > 0.04


def _one_name(symbol: str, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "symbol": symbol, "name": name, "market": "KOSPI",
            "sector_code": "0010", "market_cap_eok": 50_000,
            "sales": 1_000, "operating_profit": 100, "net_income": 80,
            "roe": 12, "financial_base_ym": "202512",
        }]
    )


def test_rising_ma20_is_tagged_uptrend_not_falling():
    row = build_base_features(
        _history("000010", "우상향", 500_000_000, 80_000, 110_000),
        _one_name("000010", "우상향"),
    ).iloc[0]
    assert row["ma20_slope"] > 0.005
    assert bool(row["ma20_rising"])
    assert not bool(row["ma20_falling"])
    assert bool(row["ma_aligned_up"])
    assert not bool(row["ma_compressed"])
    assert row["ema20"] > 0


def test_falling_ma20_without_buy_flow_is_not_bullish_divergence():
    row = build_base_features(
        _history("000010", "우하향", 0, 110_000, 80_000),
        _one_name("000010", "우하향"),
    ).iloc[0]
    assert row["ma20_slope"] < -0.005
    assert bool(row["ma20_falling"])
    assert not bool(row["ma20_rising"])
    assert row["flow_price_divergence"] != "bullish"


def test_flat_tape_is_compressed_and_not_a_hard_entry_block():
    row = build_base_features(
        _flow_history([500_000_000.0] * 40, name="횡보"),
        _one_name("000010", "횡보"),
    ).iloc[0]
    assert bool(row["ma_compressed"])
    assert abs(row["ma20_slope"]) < 0.001
    assert not bool(row["ma20_falling"])
    assert not bool(row["ma20_rising"])
    assert row["flow_price_divergence"] == ""


def test_price_down_foreign_up_is_bullish_flow_divergence():
    history = _flow_history([800_000_000.0] * 40, name="다이버전스")
    tail = history.index[-5:]
    history.loc[tail, "close"] = [99_000, 98_200, 97_400, 96_600, 95_800]
    history.loc[tail, "open"] = history.loc[tail, "close"]
    history.loc[tail, "high"] = history.loc[tail, "close"] * 1.005
    history.loc[tail, "low"] = history.loc[tail, "close"] * 0.995
    row = build_base_features(history, _one_name("000010", "다이버전스")).iloc[0]
    assert row["price_return5"] < -0.01
    assert row["foreign_net5"] > 0
    assert row["foreign_net20"] > 0
    assert row["flow_price_divergence"] == "bullish"
