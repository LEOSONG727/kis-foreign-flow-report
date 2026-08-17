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


def _streak(values: pd.Series, threshold: float = 0.001) -> int:
    count = 0
    for value in reversed(values.tolist()):
        if value > threshold:
            count += 1
        else:
            break
    return count


def _holding_change(symbol: str, snapshots: pd.DataFrame) -> float:
    if snapshots.empty:
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
        w1, w3, w5, w20, w60, w120 = (_window(group, n) for n in (1, 3, 5, 20, 60, 120))
        for window in (w1, w3, w5, w20, w60, w120):
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
        absolute_flow20 = float(w20["foreign_net_amount"].abs().sum())
        flow_concentration20 = (
            float(w20["foreign_net_amount"].abs().max() / absolute_flow20)
            if absolute_flow20 > 0 else 0.0
        )

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
        w20_returns = day_return.loc[w20.index]
        max_abs_daily_return20 = float(w20_returns.abs().max()) if not w20_returns.empty else 0.0
        absorption_mask = (w20_returns < 0) & (w20["foreign_net_amount"] > 0)
        absorption_amount = float(w20.loc[absorption_mask, "foreign_net_amount"].sum())
        absorption_share = absorption_amount / foreign_pos20 if foreign_pos20 > 0 else 0.0

        volume20 = w20["volume"].replace(0, np.nan)
        volume_ratio5_20 = float(w5["volume"].median() / volume20.median()) if volume20.median() > 0 else 1.0
        rolling_high60 = float(w60["high"].max()) if not w60.empty else float(latest["high"])
        rolling_low60 = float(w60["low"].min()) if not w60.empty else float(latest["low"])
        ma20 = float(w20["close"].mean())
        ma60 = float(w60["close"].mean())

        previous_close = float(group["close"].iloc[-2]) if len(group) >= 2 else float(latest["close"])
        gap = float(latest["open"] / previous_close - 1.0) if previous_close > 0 else 0.0
        daily_range = max(float(latest["high"] - latest["low"]), 1.0)
        upper_wick = float((latest["high"] - max(latest["open"], latest["close"])) / daily_range)
        target, target_upside, coverage, opinion_recency_days, opinion_dispersion = _opinion_consensus(
            symbol, opinions, float(latest["close"]), latest["date"]
        )

        latest_snapshot = pd.DataFrame()
        if not snapshots.empty:
            latest_snapshot = snapshots[
                snapshots["symbol"].astype(str).str.zfill(6) == symbol
            ].sort_values("date").tail(1)
        snapshot = latest_snapshot.iloc[0] if not latest_snapshot.empty else {}

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
                "price_return5": _return(group, 5),
                "price_return1": _return(group, 1),
                "price_return20": _return(group, 20),
                "price_return60": _return(group, 60),
                "price_return120": _return(group, 120),
                "ma20": ma20,
                "ma60": ma60,
                "distance_high60": float(latest["close"] / rolling_high60 - 1.0) if rolling_high60 else 0.0,
                "distance_low60": float(latest["close"] / rolling_low60 - 1.0) if rolling_low60 else 0.0,
                "atr20_pct": atr20,
                "gap_pct": gap,
                "upper_wick": upper_wick,
                "max_abs_daily_return20": max_abs_daily_return20,
                "adtv20": adtv20,
                "volume_ratio5_20": volume_ratio5_20,
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
                "foreign_flow_concentration20": flow_concentration20,
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
                "market_cap_eok": _float_or(m.get("market_cap_eok")),
                "sales": _float_or(m.get("sales")),
                "operating_profit": _float_or(m.get("operating_profit")),
                "net_income": _float_or(m.get("net_income")),
                "roe": _float_or(m.get("roe")),
                "financial_base_ym": str(m.get("financial_base_ym", "")),
                "foreign_holding_qty": _float_or(snapshot.get("foreign_holding_qty", np.nan), np.nan),
                "foreign_holding_change": _holding_change(symbol, snapshots),
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
    return pd.DataFrame(rows)


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
    return frame
