from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_HISTORY_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "foreign_net_qty",
    "foreign_net_amount",
    "institution_net_qty",
    "institution_net_amount",
]


def _number(value: Any) -> float:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _infer_amount_scale(rows: list[dict], amount_key: str, qty_key: str) -> float:
    ratios: list[float] = []
    for row in rows:
        amount = abs(_number(row.get(amount_key)))
        qty = abs(_number(row.get(qty_key)))
        close = abs(_number(row.get("stck_clpr")))
        if amount > 0 and qty > 0 and close > 0:
            ratios.append((qty * close) / amount)
    if not ratios:
        # The live investor endpoint currently reports these fields in KRW millions.
        # Keep this as a fallback only; normal responses are inferred from qty × price.
        return 1_000_000.0
    median = float(np.median(ratios))
    # KIS field units can differ by endpoint/version. Select the unit whose
    # converted amount is closest to net quantity × closing price.
    candidates = (1.0, 1_000.0, 1_000_000.0)
    return min(candidates, key=lambda scale: abs(np.log10(max(median, 1e-12) / scale)))


def validate_investor_amounts(frame: pd.DataFrame) -> None:
    """Reject obviously mis-scaled flow amounts before a report can look successful."""
    if frame.empty:
        return
    required = {"close", "foreign_net_qty", "foreign_net_amount"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Investor data is missing columns: {sorted(required - set(frame.columns))}")
    quantity_rows = frame[
        (frame["close"] > 0)
        & (frame["foreign_net_qty"].abs() > 0)
    ]
    if len(quantity_rows) >= 5:
        amount_coverage = float((quantity_rows["foreign_net_amount"].abs() > 0).mean())
        if amount_coverage < 0.20:
            raise ValueError(
                "Foreign-flow amount coverage validation failed: "
                f"only {amount_coverage:.1%} of non-zero quantity rows have an amount. "
                "Refusing to publish a misleading report."
            )
    comparable = quantity_rows[quantity_rows["foreign_net_amount"].abs() > 0]
    if len(comparable) < 5:
        return
    ratio = (
        comparable["foreign_net_amount"].abs()
        / (comparable["foreign_net_qty"].abs() * comparable["close"])
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return
    median = float(ratio.median())
    if not 0.05 <= median <= 20.0:
        raise ValueError(
            "Foreign-flow amount unit validation failed: "
            f"median(amount/(qty*close))={median:.6g}. "
            "Refusing to publish a misleading report."
        )


def normalize_investor_rows(
    symbol: str,
    name: str,
    market: str,
    rows: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    raw = [row for row in rows if isinstance(row, dict)]
    if not raw:
        return pd.DataFrame()
    foreign_scale = _infer_amount_scale(raw, "frgn_ntby_tr_pbmn", "frgn_ntby_qty")
    institution_scale = _infer_amount_scale(raw, "orgn_ntby_tr_pbmn", "orgn_ntby_qty")
    normalized = []
    for row in raw:
        dt = pd.to_datetime(str(row.get("stck_bsop_date", "")), format="%Y%m%d", errors="coerce")
        close = _number(row.get("stck_clpr"))
        turnover = _number(row.get("acml_tr_pbmn"))
        if pd.isna(dt) or close <= 0 or turnover <= 0:
            continue
        normalized.append(
            {
                "date": dt,
                "symbol": str(symbol).zfill(6),
                "market": market,
                "name": name,
                "open": _number(row.get("stck_oprc")),
                "high": _number(row.get("stck_hgpr")),
                "low": _number(row.get("stck_lwpr")),
                "close": close,
                "volume": _number(row.get("acml_vol")),
                "turnover": turnover,
                "foreign_net_qty": _number(row.get("frgn_ntby_qty")),
                "foreign_net_amount": _number(row.get("frgn_ntby_tr_pbmn")) * foreign_scale,
                "institution_net_qty": _number(row.get("orgn_ntby_qty")),
                "institution_net_amount": _number(row.get("orgn_ntby_tr_pbmn")) * institution_scale,
            }
        )
    frame = pd.DataFrame(normalized)
    if frame.empty:
        return frame
    return frame.drop_duplicates(["date", "symbol"], keep="last").sort_values("date")


def normalize_price_snapshot(symbol: str, as_of: pd.Timestamp, row: dict[str, Any]) -> dict:
    return {
        "date": as_of,
        "symbol": str(symbol).zfill(6),
        "current_price": _number(row.get("stck_prpr")),
        "foreign_holding_qty": _number(row.get("frgn_hldn_qty")),
        "foreign_holding_ratio": _number(row.get("hts_frgn_ehrt")),
        "per": _number(row.get("per")),
        "pbr": _number(row.get("pbr")),
        "market_cap_eok": _number(row.get("hts_avls")),
        "investment_caution": str(row.get("invt_caful_yn", "N")),
        "market_warning": str(row.get("mrkt_warn_cls_code", "00")),
        "managed": str(row.get("mang_issu_cls_code", "00")),
    }


def normalize_opinions(symbol: str, rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dt = pd.to_datetime(str(row.get("stck_bsop_date", "")), format="%Y%m%d", errors="coerce")
        target = _number(row.get("hts_goal_prc"))
        if pd.isna(dt) or target <= 0:
            continue
        result.append(
            {
                "date": dt,
                "symbol": str(symbol).zfill(6),
                "opinion": str(row.get("invt_opnn", "")),
                "opinion_code": str(row.get("invt_opnn_cls_code", "")),
                "target_price": target,
                "reference_price": _number(row.get("stck_prdy_clpr")),
            }
        )
    return pd.DataFrame(result)


def _window(group: pd.DataFrame, n: int) -> pd.DataFrame:
    return group.tail(min(n, len(group))).copy()


def _float_or(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else default


def _return(group: pd.DataFrame, n: int) -> float:
    view = group.tail(min(n + 1, len(group)))
    if len(view) < 2 or view["close"].iloc[0] <= 0:
        return 0.0
    return float(view["close"].iloc[-1] / view["close"].iloc[0] - 1.0)


def _lagged_return(group: pd.DataFrame, n: int, end_offset: int) -> float:
    """Return over n sessions ending end_offset sessions before the latest row."""
    end = len(group) - end_offset
    start = end - n - 1
    if start < 0 or end <= start + 1:
        return 0.0
    first = float(group["close"].iloc[start])
    last = float(group["close"].iloc[end - 1])
    return last / first - 1.0 if first > 0 else 0.0


def _ma20_touches(group: pd.DataFrame, lookback: int = 3) -> tuple[bool, bool, int]:
    """Whether the candle wick crossed the 20-day close average.

    Today and the last few sessions are checked separately so a first
    pullback to the 20-day line can be distinguished from a close that
    is merely nearby.
    """
    if group.empty or "close" not in group or "low" not in group or "high" not in group:
        return False, False, 0
    ma = pd.to_numeric(group["close"], errors="coerce").rolling(20, min_periods=20).mean()
    recent = group.tail(min(lookback, len(group)))
    last_idx = group.index[-1]
    today = False
    count = 0
    for idx in recent.index:
        level = ma.loc[idx] if idx in ma.index else np.nan
        if pd.isna(level) or float(level) <= 0:
            continue
        low = float(recent.loc[idx, "low"])
        high = float(recent.loc[idx, "high"])
        if low <= float(level) <= high:
            count += 1
            if idx == last_idx:
                today = True
    return today, count > 0, count


MA20_SLOPE_LOOKBACK = 5
MA20_RISING_SLOPE = 0.002
MA20_FALLING_SLOPE = -0.005
MA_COMPRESSED_SPREAD = 0.012
MA_FAN_SPREAD = 0.030
DIV_BULLISH_PRICE5 = -0.01
DIV_BEARISH_PRICE5 = 0.025


def _series_slope(series: pd.Series, lookback: int = MA20_SLOPE_LOOKBACK) -> float:
    """Fractional change of a level over `lookback` sessions."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= lookback:
        return 0.0
    now = float(clean.iloc[-1])
    prev = float(clean.iloc[-(lookback + 1)])
    if prev <= 0 or now <= 0:
        return 0.0
    return now / prev - 1.0


def _ma_regime(ma5: float, ma10: float, ma20: float, ma20_slope: float) -> dict[str, Any]:
    """Trend shape of the 5/10/20-day averages. Compression is descriptive, not a hard gate."""
    levels = [float(ma5), float(ma10), float(ma20)]
    spread = (max(levels) - min(levels)) / ma20 if ma20 > 0 else 0.0
    aligned_up = ma5 >= ma10 >= ma20 > 0
    aligned_down = 0 < ma5 <= ma10 <= ma20
    return {
        "ma_spread_pct": float(spread),
        "ma_aligned_up": bool(aligned_up),
        "ma_aligned_down": bool(aligned_down),
        "ma_compressed": bool(spread < MA_COMPRESSED_SPREAD),
        "ma_fan_up": bool(aligned_up and spread > MA_FAN_SPREAD),
        "ma20_rising": bool(ma20_slope >= MA20_RISING_SLOPE),
        "ma20_falling": bool(ma20_slope <= MA20_FALLING_SLOPE),
    }


def _flow_price_divergence(price5: float, foreign_net5: float, foreign_net20: float) -> str:
    """Multi-day price vs foreign-flow split. Empty string means no readable divergence."""
    if price5 <= DIV_BULLISH_PRICE5 and foreign_net5 > 0 and foreign_net20 > 0:
        return "bullish"
    if price5 >= DIV_BEARISH_PRICE5 and foreign_net5 < 0:
        return "bearish"
    return ""


def _streak(values: pd.Series, threshold: float = 0.001) -> int:
    count = 0
    for value in reversed(values.tolist()):
        if value > threshold:
            count += 1
        else:
            break
    return count


def _cumulative_flow_path(
    amounts: pd.Series,
    window: int = 20,
    lookback: int = 10,
) -> dict[str, float | bool | int]:
    """Describe daily foreign buying and the actual rolling-20 inventory path.

    ``foreign_buy_days10`` counts positive *daily* net buys. It is the main
    persistence signal because it separates steady demand from a two-day
    burst. ``foreign_inventory_up_days10`` is intentionally separate: it
    counts increases in the rolling 20-session sum and must not be described
    as daily foreign buying.

    The legacy ``foreign_roll20_*`` aliases are kept for one migration cycle
    so stored CSVs and downstream readers do not fail abruptly.
    """
    empty = {
        "foreign_buy_days10": 0,
        "foreign_sell_days10": 0,
        "foreign_buy_days20": 0,
        "foreign_buy_streak": 0,
        "foreign_bought_today": False,
        "foreign_inventory_up_days10": 0,
        "foreign_inventory_down_days10": 0,
    }
    empty.update(
        {
            "foreign_roll20_up_days10": 0,
            "foreign_roll20_down_days10": 0,
            "foreign_roll20_up_days20": 0,
            "foreign_roll20_up_streak": 0,
            "foreign_roll20_today_up": False,
        }
    )
    if amounts.empty:
        return empty
    daily = pd.to_numeric(amounts, errors="coerce").fillna(0.0)
    roll = daily.rolling(window, min_periods=window).sum()
    current = float(roll.iloc[-1]) if not roll.empty and pd.notna(roll.iloc[-1]) else float(daily.tail(window).sum())
    epsilon = max(10_000_000.0, 0.005 * abs(current) if current else 10_000_000.0)
    last10 = daily.tail(lookback)
    last20 = daily.tail(20)
    today = float(daily.iloc[-1]) if pd.notna(daily.iloc[-1]) else 0.0
    inventory_change = roll.diff().dropna().tail(lookback)
    streak = 0
    for value in reversed(daily.tolist()):
        if value > epsilon:
            streak += 1
        else:
            break
    result = {
        "foreign_buy_days10": int((last10 > epsilon).sum()),
        "foreign_sell_days10": int((last10 < -epsilon).sum()),
        "foreign_buy_days20": int((last20 > epsilon).sum()),
        "foreign_buy_streak": streak,
        "foreign_bought_today": today > epsilon,
        "foreign_inventory_up_days10": int((inventory_change > epsilon).sum()),
        "foreign_inventory_down_days10": int((inventory_change < -epsilon).sum()),
    }
    result.update(
        {
            "foreign_roll20_up_days10": result["foreign_buy_days10"],
            "foreign_roll20_down_days10": result["foreign_sell_days10"],
            "foreign_roll20_up_days20": result["foreign_buy_days20"],
            "foreign_roll20_up_streak": result["foreign_buy_streak"],
            "foreign_roll20_today_up": result["foreign_bought_today"],
        }
    )
    return result


def _holding_change(symbol: str, snapshots: pd.DataFrame) -> float:
    if snapshots.empty or "foreign_holding_qty" not in snapshots:
        return float("nan")
    group = snapshots[snapshots["symbol"].astype(str).str.zfill(6) == symbol].copy()
    group = group.sort_values("date").dropna(subset=["foreign_holding_qty"])
    if len(group) < 2:
        return float("nan")
    latest_date = pd.to_datetime(group["date"].iloc[-1])
    eligible = group[pd.to_datetime(group["date"]) <= latest_date - pd.Timedelta(days=14)]
    if eligible.empty:
        return float("nan")
    return float(group["foreign_holding_qty"].iloc[-1] - eligible["foreign_holding_qty"].iloc[-1])


def _opinion_consensus(
    symbol: str,
    opinions: pd.DataFrame,
    close: float,
    as_of: pd.Timestamp,
) -> tuple[float, float, int, float, float]:
    if opinions.empty or close <= 0:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    group = opinions[opinions["symbol"].astype(str).str.zfill(6) == symbol].copy()
    group["date"] = pd.to_datetime(group["date"], errors="coerce")
    as_of = pd.to_datetime(as_of)
    group = group[
        (group["date"] <= as_of)
        & (group["date"] >= as_of - pd.Timedelta(days=120))
    ]
    group = group.dropna(subset=["date", "target_price"]).sort_values("date").tail(20)
    if group.empty:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    targets = pd.to_numeric(group["target_price"], errors="coerce").dropna()
    target = float(targets.median())
    dispersion = float((targets.quantile(0.75) - targets.quantile(0.25)) / target) if target > 0 else float("nan")
    recency_days = float((as_of - group["date"].max()).days)
    return target, target / close - 1.0, len(group), recency_days, dispersion


def build_base_features(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    snapshots: pd.DataFrame | None = None,
    opinions: pd.DataFrame | None = None,
    *, point_in_time: bool = False,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    snapshots = snapshots if snapshots is not None else pd.DataFrame()
    opinions = opinions if opinions is not None else pd.DataFrame()
    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype("string").str.zfill(6)
    for column in NUMERIC_HISTORY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame.sort_values(["symbol", "date"])

    meta = universe.copy()
    meta["symbol"] = meta["symbol"].astype("string").str.zfill(6)
    meta = meta.set_index("symbol", drop=False)
    rows: list[dict] = []
    for symbol, group in frame.groupby("symbol", sort=False):
        symbol = str(symbol).zfill(6)
        group = group.sort_values("date").drop_duplicates("date", keep="last")
        if len(group) < 5 or symbol not in meta.index:
            continue
        m = meta.loc[symbol]
        latest = group.iloc[-1]
        w1, w3, w5, w10, w20, w60, w120 = (
            _window(group, n) for n in (1, 3, 5, 10, 20, 60, 120)
        )
        for window in (w1, w3, w5, w10, w20, w60, w120):
            window["flow_ratio"] = np.where(
                window["turnover"] > 0,
                window["foreign_net_amount"] / window["turnover"],
                0.0,
            )
        adtv20 = float(w20["turnover"].median()) if not w20.empty else 0.0
        foreign_pos20 = float(w20.loc[w20["foreign_net_amount"] > 0, "foreign_net_amount"].sum())
        foreign_neg20 = abs(float(w20.loc[w20["foreign_net_amount"] < 0, "foreign_net_amount"].sum()))
        flow_ratio20 = w20["flow_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        positive = flow_ratio20 > 0.001
        negative = flow_ratio20 < -0.001
        neutral = ~(positive | negative)
        tolerant_buy_ratio = float((positive.sum() + 0.5 * neutral.sum()) / max(len(w20), 1))
        sellback = foreign_neg20 / foreign_pos20 if foreign_pos20 > 0 else 9.0
        prior15 = w20.iloc[:-5] if len(w20) > 5 else w20.iloc[:0]
        foreign_net3 = float(w3["foreign_net_amount"].sum())
        foreign_net5 = float(w5["foreign_net_amount"].sum())
        foreign_net20 = float(w20["foreign_net_amount"].sum())
        foreign_prior15 = float(prior15["foreign_net_amount"].sum())
        institution_net5 = float(w5["institution_net_amount"].sum())
        institution_net20 = float(w20["institution_net_amount"].sum())
        acceleration5 = (
            (foreign_net5 / 5.0 - foreign_prior15 / max(len(prior15), 1)) / adtv20
            if adtv20 else 0.0
        )
        prior17 = w20.iloc[:-3] if len(w20) > 3 else w20.iloc[:0]
        acceleration3 = (
            (foreign_net3 / 3.0 - float(prior17["foreign_net_amount"].sum()) / max(len(prior17), 1))
            / adtv20
            if adtv20 else 0.0
        )
        absolute_flow20 = float(w20["foreign_net_amount"].abs().sum())
        flow_concentration20 = (
            float(w20["foreign_net_amount"].abs().max() / absolute_flow20)
            if absolute_flow20 > 0 else 0.0
        )
        flow_path = _cumulative_flow_path(group["foreign_net_amount"])

        close_series = group["close"].replace(0, np.nan).ffill()
        prev_close = close_series.shift(1)
        true_range = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - prev_close).abs(),
                (group["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr20 = float(true_range.tail(20).mean() / latest["close"]) if latest["close"] > 0 else 0.0
        day_return = group["close"].pct_change().fillna(0.0)
        w5_returns = day_return.loc[w5.index]
        w10_returns = day_return.loc[w10.index]
        w20_returns = day_return.loc[w20.index]
        max_abs_daily_return20 = float(w20_returns.abs().max()) if not w20_returns.empty else 0.0
        absorption_mask = (w20_returns < 0) & (w20["foreign_net_amount"] > 0)
        absorption_amount = float(w20.loc[absorption_mask, "foreign_net_amount"].sum())
        absorption_share = absorption_amount / foreign_pos20 if foreign_pos20 > 0 else 0.0

        volume20 = w20["volume"].replace(0, np.nan)
        volume_ratio5_20 = float(w5["volume"].median() / volume20.median()) if volume20.median() > 0 else 1.0
        rolling_high20 = float(w20["high"].max()) if not w20.empty else float(latest["high"])
        rolling_low20 = float(w20["low"].min()) if not w20.empty else float(latest["low"])
        rolling_low10 = float(w10["low"].min()) if not w10.empty else float(latest["low"])
        rolling_high60 = float(w60["high"].max()) if not w60.empty else float(latest["high"])
        rolling_low60 = float(w60["low"].min()) if not w60.empty else float(latest["low"])
        ma20 = float(w20["close"].mean())
        ma60 = float(w60["close"].mean())
        ma5 = float(w5["close"].mean())
        ma10 = float(w10["close"].mean())
        prior_ma5 = float(group.iloc[-8:-3]["close"].mean()) if len(group) >= 8 else ma5
        ma5_slope = ma5 / prior_ma5 - 1.0 if prior_ma5 > 0 else 0.0
        close_numeric = pd.to_numeric(group["close"], errors="coerce")
        sma20_series = close_numeric.rolling(20, min_periods=20).mean()
        ema20_series = close_numeric.ewm(span=20, adjust=False, min_periods=10).mean()
        ema20_value = ema20_series.iloc[-1]
        ema20 = float(ema20_value) if pd.notna(ema20_value) and float(ema20_value) > 0 else ma20
        ma20_slope = _series_slope(sma20_series, MA20_SLOPE_LOOKBACK)
        ema20_slope = _series_slope(ema20_series, MA20_SLOPE_LOOKBACK)
        regime = _ma_regime(ma5, ma10, ma20, ma20_slope)
        price_return5 = _return(group, 5)
        flow_div = _flow_price_divergence(price_return5, foreign_net5, foreign_net20)
        ma20_distance = float(latest["close"] / ma20 - 1.0) if ma20 > 0 else 0.0
        ma20_atr_distance = ma20_distance / atr20 if atr20 > 0 else 0.0
        touched_ma20, touched_ma20_3, _ = _ma20_touches(group)
        near_band = max(0.01, 0.5 * atr20) if atr20 > 0 else 0.01
        near_ma20 = abs(ma20_distance) <= near_band
        above_ma20 = float(latest["close"]) >= ma20
        # A support tag requires the line to be tested without closing
        # meaningfully through it. A close well below is a breakdown.
        ma20_support = (touched_ma20 or near_ma20) and ma20_distance >= -near_band
        rebound_from_low20 = float(latest["close"] / rolling_low20 - 1.0) if rolling_low20 else 0.0
        rebound_atr_distance20 = rebound_from_low20 / atr20 if atr20 > 0 else 0.0
        low_position20 = int(np.argmin(w20["low"].to_numpy()))
        base_window20 = w20.iloc[: low_position20 + 1]
        rebound_window20 = w20.iloc[low_position20 + 1 :]
        rebound_volume = rebound_window20["volume"].replace(0, np.nan).median()
        base_volume = base_window20["volume"].replace(0, np.nan).median()
        rebound_volume_ratio20 = (
            float(rebound_volume / base_volume)
            if pd.notna(rebound_volume) and pd.notna(base_volume) and base_volume > 0
            else 1.0
        )
        up_volume10 = w10.loc[w10_returns > 0, "volume"].replace(0, np.nan).median()
        down_volume10 = w10.loc[w10_returns < 0, "volume"].replace(0, np.nan).median()
        up_down_volume_ratio10 = (
            float(up_volume10 / down_volume10)
            if pd.notna(up_volume10) and pd.notna(down_volume10) and down_volume10 > 0
            else 1.0
        )
        price_return3 = _return(group, 3)
        price_return_prev3 = _lagged_return(group, 3, 3)

        previous_close = float(group["close"].iloc[-2]) if len(group) >= 2 else float(latest["close"])
        gap = float(latest["open"] / previous_close - 1.0) if previous_close > 0 else 0.0
        daily_range = max(float(latest["high"] - latest["low"]), 1.0)
        upper_wick = float((latest["high"] - max(latest["open"], latest["close"])) / daily_range)
        target, target_upside, coverage, opinion_recency_days, opinion_dispersion = _opinion_consensus(
            symbol, opinions, float(latest["close"]), latest["date"]
        )

        latest_snapshot = pd.DataFrame()
        symbol_snapshots = pd.DataFrame()
        if not snapshots.empty:
            symbol_snapshots = snapshots[
                snapshots["symbol"].astype(str).str.zfill(6) == symbol
            ].copy()
            if point_in_time:
                symbol_snapshots = symbol_snapshots[pd.to_datetime(symbol_snapshots["date"], errors="coerce") <= latest["date"]]
            latest_snapshot = symbol_snapshots.sort_values("date").tail(1)
        snapshot = latest_snapshot.iloc[0] if not latest_snapshot.empty else {}
        shares = _float_or(m.get("listed_shares"), np.nan)
        cap = _float_or(m.get("market_cap_eok"))
        if point_in_time:
            quote_cap = _float_or(snapshot.get("market_cap_eok"), np.nan)
            quote_price = _float_or(snapshot.get("current_price"), np.nan)
            shares = quote_cap * 100_000_000.0 / quote_price if quote_cap > 0 and quote_price > 0 else np.nan
            cap = shares * float(latest["close"]) / 100_000_000.0

        rows.append(
            {
                "symbol": symbol,
                "name": str(m.get("name", latest.get("name", symbol))),
                "market": str(m.get("market", latest.get("market", ""))),
                "sector_code": str(m.get("sector_code", "")),
                "as_of": latest["date"],
                "data_points": len(group),
                "close": float(latest["close"]),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "price_return1": _return(group, 1),
                "price_return3": price_return3,
                "price_return_prev3": price_return_prev3,
                "price_acceleration3": price_return3 - price_return_prev3,
                "price_return5": price_return5,
                "price_return10": _return(group, 10),
                "price_return20": _return(group, 20),
                "price_return60": _return(group, 60),
                "price_return120": _return(group, 120),
                "ma20": ma20,
                "ema20": ema20,
                "ma60": ma60,
                "ma5": ma5,
                "ma10": ma10,
                "above_ma5": bool(float(latest["close"]) >= ma5),
                "above_ma10": bool(float(latest["close"]) >= ma10),
                "above_ma20": above_ma20,
                "ma5_slope": ma5_slope,
                "ma20_slope": ma20_slope,
                "ema20_slope": ema20_slope,
                "ma_spread_pct": regime["ma_spread_pct"],
                "ma_aligned_up": regime["ma_aligned_up"],
                "ma_aligned_down": regime["ma_aligned_down"],
                "ma_compressed": regime["ma_compressed"],
                "ma_fan_up": regime["ma_fan_up"],
                "ma20_rising": regime["ma20_rising"],
                "ma20_falling": regime["ma20_falling"],
                "flow_price_divergence": flow_div,
                "ma20_distance": ma20_distance,
                "ma20_atr_distance": ma20_atr_distance,
                "touched_ma20": touched_ma20,
                "touched_ma20_3": touched_ma20_3,
                "near_ma20": near_ma20,
                "ma20_support": ma20_support,
                "rebound_from_low10": float(latest["close"] / rolling_low10 - 1.0) if rolling_low10 else 0.0,
                "rebound_from_low20": rebound_from_low20,
                "rebound_atr_distance20": rebound_atr_distance20,
                "distance_high20": float(latest["close"] / rolling_high20 - 1.0) if rolling_high20 else 0.0,
                "distance_high60": float(latest["close"] / rolling_high60 - 1.0) if rolling_high60 else 0.0,
                "distance_low60": float(latest["close"] / rolling_low60 - 1.0) if rolling_low60 else 0.0,
                "atr20_pct": atr20,
                "up_days5": int((w5_returns > 0).sum()),
                "up_days10": int((w10_returns > 0).sum()),
                "price_up_streak": _streak(w20_returns, threshold=0.0),
                "gap_pct": gap,
                "upper_wick": upper_wick,
                "max_abs_daily_return20": max_abs_daily_return20,
                "adtv20": adtv20,
                "volume_ratio5_20": volume_ratio5_20,
                "rebound_volume_ratio20": rebound_volume_ratio20,
                "up_down_volume_ratio10": up_down_volume_ratio10,
                "foreign_net3": foreign_net3,
                "foreign_net5": foreign_net5,
                "foreign_net1": float(w1["foreign_net_amount"].sum()),
                "foreign_net20": foreign_net20,
                "foreign_net60": float(w60["foreign_net_amount"].sum()),
                "foreign_net120": float(w120["foreign_net_amount"].sum()),
                "foreign_3_adtv": float(foreign_net3 / adtv20) if adtv20 else 0.0,
                "foreign_5_adtv": float(foreign_net5 / adtv20) if adtv20 else 0.0,
                "foreign_1_adtv": float(w1["foreign_net_amount"].sum() / adtv20) if adtv20 else 0.0,
                "foreign_20_adtv": float(foreign_net20 / adtv20) if adtv20 else 0.0,
                "foreign_60_adtv": float(w60["foreign_net_amount"].sum() / adtv20) if adtv20 else 0.0,
                "foreign_120_adtv": float(w120["foreign_net_amount"].sum() / adtv20) if adtv20 else 0.0,
                "foreign_prior15_adtv": float(foreign_prior15 / adtv20) if adtv20 else 0.0,
                "foreign_acceleration5": acceleration5,
                "foreign_acceleration3": acceleration3,
                "foreign_flow_concentration20": flow_concentration20,
                "foreign_buy_days10": flow_path["foreign_buy_days10"],
                "foreign_sell_days10": flow_path["foreign_sell_days10"],
                "foreign_buy_days20": flow_path["foreign_buy_days20"],
                "foreign_buy_streak": flow_path["foreign_buy_streak"],
                "foreign_bought_today": flow_path["foreign_bought_today"],
                "foreign_inventory_up_days10": flow_path["foreign_inventory_up_days10"],
                "foreign_inventory_down_days10": flow_path["foreign_inventory_down_days10"],
                # Backward-compatible aliases; use the canonical names above.
                "foreign_roll20_up_days10": flow_path["foreign_buy_days10"],
                "foreign_roll20_down_days10": flow_path["foreign_sell_days10"],
                "foreign_roll20_up_days20": flow_path["foreign_buy_days20"],
                "foreign_roll20_up_streak": flow_path["foreign_buy_streak"],
                "foreign_roll20_today_up": flow_path["foreign_bought_today"],
                "foreign_flow_share20": float(w20["foreign_net_amount"].sum() / w20["turnover"].sum()) if w20["turnover"].sum() else 0.0,
                "foreign_ewm20": float(w120["flow_ratio"].ewm(halflife=20, adjust=False).mean().iloc[-1]),
                "foreign_ewm120": float(w120["flow_ratio"].ewm(halflife=120, adjust=False).mean().iloc[-1]),
                "tolerant_buy_ratio20": tolerant_buy_ratio,
                "buy_days20": int(positive.sum()),
                "sell_days20": int(negative.sum()),
                "current_buy_streak": _streak(flow_ratio20),
                "sellback_ratio20": sellback,
                "absorption_share20": absorption_share,
                "institution_net5": institution_net5,
                "institution_net20": institution_net20,
                "institution_5_adtv": float(institution_net5 / adtv20) if adtv20 else 0.0,
                "institution_20_adtv": float(institution_net20 / adtv20) if adtv20 else 0.0,
                "market_cap_eok": cap,
                "listed_shares": shares,
                "sales": _float_or(m.get("sales")),
                "operating_profit": _float_or(m.get("operating_profit")),
                "net_income": _float_or(m.get("net_income")),
                "roe": _float_or(m.get("roe")),
                "financial_base_ym": str(m.get("financial_base_ym", "")),
                "foreign_holding_qty": _float_or(snapshot.get("foreign_holding_qty", np.nan), np.nan),
                "foreign_holding_change": _holding_change(symbol, symbol_snapshots),
                "foreign_holding_ratio": _float_or(snapshot.get("foreign_holding_ratio", np.nan), np.nan),
                "per": _float_or(snapshot.get("per", np.nan), np.nan),
                "pbr": _float_or(snapshot.get("pbr", np.nan), np.nan),
                "investment_caution": str(snapshot.get("investment_caution", "N")),
                "market_warning": str(snapshot.get("market_warning", "00")),
                "managed_status": str(snapshot.get("managed", "00")),
                "target_price": target,
                "target_upside": target_upside,
                "opinion_coverage": coverage,
                "opinion_recency_days": opinion_recency_days,
                "opinion_dispersion": opinion_dispersion,
                "history_confidence": "충분" if len(group) >= 120 else ("보통" if len(group) >= 60 else "초기"),
                "price_series": group.tail(60)["close"].astype(float).tolist(),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if point_in_time:
        # Today's master shares/profits/ROE must not enter historical scores.
        # Infer shares only from a quote snapshot observable on that date.
        for column in ("sales", "operating_profit", "net_income", "roe"):
            result[column] = np.nan  # A verified DART receipt supplies these later.
        result["financial_base_ym"] = ""
    # Size on the signal date. market_cap_eok is today's value from the
    # universe master and leaks the future when replayed.
    pit_value = result["listed_shares"] * result["close"]
    result["pit_size"] = np.log(pit_value.where(pit_value > 0))
    result["log_adtv20"] = np.log(result["adtv20"].clip(lower=1.0))
    holding_value = result["foreign_holding_qty"] * result["close"]
    result["foreign_ownership_days_adtv"] = np.where(
        (result["adtv20"] > 0) & holding_value.notna(),
        holding_value / result["adtv20"],
        np.nan,
    )
    holding_change_value = result["foreign_holding_change"] * result["close"]
    result["foreign_holding_change_adtv"] = np.where(
        (result["adtv20"] > 0) & holding_change_value.notna(),
        holding_change_value / result["adtv20"],
        np.nan,
    )
    return result


def add_cross_sectional_residuals(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    frame = features.copy()
    market_price = frame.groupby("market")["price_return20"].transform("median")
    market_flow = frame.groupby("market")["foreign_flow_share20"].transform("median")
    sector_key = frame["market"].astype(str) + ":" + frame["sector_code"].astype(str)
    sector_count = frame.groupby(sector_key)["symbol"].transform("count")
    sector_price = frame.groupby(sector_key)["price_return20"].transform("median")
    sector_flow = frame.groupby(sector_key)["foreign_flow_share20"].transform("median")
    sector_price = sector_price.where(sector_count >= 5, market_price)
    sector_flow = sector_flow.where(sector_count >= 5, market_flow)
    price_benchmark = 0.4 * market_price + 0.6 * sector_price
    flow_benchmark = 0.4 * market_flow + 0.6 * sector_flow
    frame["price_residual20"] = frame["price_return20"] - price_benchmark
    frame["flow_residual20"] = frame["foreign_flow_share20"] - flow_benchmark
    for horizon in (3, 5, 10):
        market_short = frame.groupby("market")[f"price_return{horizon}"].transform("median")
        sector_short = frame.groupby(sector_key)[f"price_return{horizon}"].transform("median")
        sector_short = sector_short.where(sector_count >= 5, market_short)
        frame[f"price_residual{horizon}"] = frame[f"price_return{horizon}"] - (
            0.4 * market_short + 0.6 * sector_short
        )
    frame["market_return20"] = market_price
    return frame
