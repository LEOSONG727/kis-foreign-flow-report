from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelConfig


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def scale(value: float, low: float, high: float) -> float:
    if not math.isfinite(float(value)) or high <= low:
        return 0.0
    return clamp((float(value) - low) / (high - low), 0.0, 1.0)


def _calibration_active(calibration: dict[str, Any] | None) -> bool:
    """Reject legacy/sample-count-only calibrations after the V1 audit."""
    return bool(
        calibration
        and calibration.get("active")
        and calibration.get("quality_gate_version") == 2
    )


def _calibrated_probability(score: float, heuristic: float, calibration: dict[str, Any] | None) -> float:
    if not _calibration_active(calibration):
        return heuristic
    for band in calibration.get("bands", []):
        if float(band.get("low", 0.0)) <= score < float(band.get("high", 101.0)):
            value = band.get("calibrated_probability")
            if value is not None and int(band.get("samples", 0)) >= 30:
                return clamp(float(value), 0.05, 0.95)
    return heuristic


def _fundamental_score(row: pd.Series) -> tuple[float, float]:
    quality = 0.0
    if row.get("operating_profit", 0.0) > 0:
        quality += 4.0
    if row.get("net_income", 0.0) > 0:
        quality += 3.0
    quality += 5.0 * scale(row.get("roe", 0.0), 0.0, 18.0)

    upside = row.get("target_upside", np.nan)
    if pd.isna(upside) or row.get("opinion_coverage", 0) == 0:
        outlook = 4.0  # 리포트 미커버 종목을 자동으로 벌점 처리하지 않는다.
    else:
        raw_outlook = 8.0 * scale(upside, -0.10, 0.40)
        coverage_confidence = 0.2 + 0.8 * scale(row.get("opinion_coverage", 0), 1.0, 5.0)
        freshness_confidence = 1.0 - scale(row.get("opinion_recency_days", 120.0), 45.0, 120.0)
        dispersion_confidence = 1.0 - scale(row.get("opinion_dispersion", 1.0), 0.15, 0.60)
        opinion_confidence = coverage_confidence * freshness_confidence * dispersion_confidence
        outlook = 4.0 + opinion_confidence * (raw_outlook - 4.0)
    return quality, outlook


def _long_inventory20(row: pd.Series) -> float:
    """Convert 60/120-day foreign flow into a comparable 20-day inventory pace."""
    flow60 = row.get("foreign_60_adtv", 0.0) / 3.0
    flow120 = row.get("foreign_120_adtv", 0.0) / 6.0
    return 0.6 * flow60 + 0.4 * flow120


def _hard_price_heat_flags(row: pd.Series, model: ModelConfig) -> list[tuple[str, float]]:
    """Return price moves large enough to block a fresh entry on their own."""
    candidates = [
        (
            "20일 저점반등",
            row.get("rebound_from_low20", 0.0),
            model.rebound_chase_return20_pct / 100.0,
        ),
        (
            "10일 저점반등",
            row.get("rebound_from_low10", 0.0),
            model.rebound_chase_return10_pct / 100.0,
        ),
        ("10일 상승", row.get("price_return10", 0.0), model.recent_chase_return10_pct / 100.0),
        ("5일 상승", row.get("price_return5", 0.0), model.recent_chase_return5_pct / 100.0),
        ("3일 상승", row.get("price_return3", 0.0), model.recent_chase_return3_pct / 100.0),
        ("1일 상승", row.get("price_return1", 0.0), model.recent_chase_return1_pct / 100.0),
        ("20일 상승", row.get("price_return20", 0.0), model.absolute_chase_return20_pct / 100.0),
    ]
    return [(label, float(value)) for label, value, threshold in candidates if float(value) >= threshold]


def _price_heat_flags(row: pd.Series, model: ModelConfig) -> list[tuple[str, float]]:
    """Add ATR confirmations only after an absolute price-path gate is active."""
    flags = _hard_price_heat_flags(row, model)
    if not flags:
        return flags
    confirmations = [
        (
            "저점반등 ATR",
            row.get("rebound_atr_distance20", 0.0),
            model.rebound_atr_chase_multiple,
        ),
        (
            "20일선 ATR 이격",
            row.get("ma20_atr_distance", 0.0),
            model.ma20_atr_chase_multiple,
        ),
    ]
    flags.extend(
        (label, float(value))
        for label, value, threshold in confirmations
        if float(value) >= threshold
    )
    return flags


def _is_price_hot(row: pd.Series, model: ModelConfig) -> bool:
    return bool(_hard_price_heat_flags(row, model))


def _is_dip_absorption(row: pd.Series, model: ModelConfig) -> bool:
    """Price is pulling back while a live foreign campaign keeps buying the dip.

    This is the Celltrion-type tape: a strong 20/60-day accumulation, a down
    day (or 3-day pullback), and foreigners still adding. It is not a chase
    and not a falling knife, so discovery should keep the name visible.
    """
    price1 = float(row.get("price_return1", 0.0))
    price3 = float(row.get("price_return3", 0.0))
    price20 = float(row.get("price_return20", 0.0))
    foreign1 = float(row.get("foreign_1_adtv", 0.0))
    foreign3 = float(row.get("foreign_3_adtv", 0.0))
    net1 = float(row.get("foreign_net1", 0.0))
    net3 = float(row.get("foreign_net3", 0.0))
    current = float(row.get("foreign_20_adtv", 0.0))
    long_inventory = _long_inventory20(row)
    amount = float(row.get("foreign_net20", 0.0))
    buy_ratio = float(row.get("tolerant_buy_ratio20", 0.0))
    sellback = float(row.get("sellback_ratio20", 9.0))

    if price20 <= model.structural_break_return20_pct / 100.0:
        return False
    if float(row.get("max_abs_daily_return20", 0.0)) >= model.corporate_action_daily_return_pct / 100.0:
        return False
    # A 20-day collapse is still a knife even if today's tape is a buy.
    if price20 <= -0.15:
        return False
    campaign_alive = (
        current >= 0.35
        or long_inventory > 0.25
        or amount >= model.large_flow_threshold_krw
    )
    if not campaign_alive:
        return False
    if buy_ratio < 0.50 or sellback > 0.80:
        return False
    today_buy = net1 > 0 or foreign1 > 0.05
    three_buy = net3 > 0 or foreign3 > 0.10
    if not (today_buy and three_buy):
        return False
    return price1 <= -0.008 or (price3 <= -0.02 and price1 <= 0.0)


def is_steady_accumulation(row: pd.Series | dict, model: ModelConfig | None = None) -> bool:
    """The 20-day foreign pile is still being added to, almost every day.

    Looks at daily adds, not the latest 20-day sum. A name that bought on 8 of
    the last 10 sessions is the Celltrion tape. A two-day burst with the same
    20-day won amount is not.
    """
    model = model or ModelConfig()

    def _num(key: str, default: float) -> float:
        value = row.get(key, default)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return default if pd.isna(number) else number

    if _num("foreign_net20", 0.0) < model.steady_min_net20_krw:
        return False
    if _num("price_return20", 0.0) <= model.structural_break_return20_pct / 100.0:
        return False
    if _num("max_abs_daily_return20", 0.0) >= model.corporate_action_daily_return_pct / 100.0:
        return False
    buy_days10 = _num(
        "foreign_buy_days10",
        _num("foreign_roll20_up_days10", 0.0),
    )
    if buy_days10 < model.steady_min_up_days10:
        return False
    if _num("foreign_flow_concentration20", 0.0) >= model.steady_max_concentration:
        return False
    if _num("sellback_ratio20", 9.0) > model.steady_max_sellback:
        return False
    today_up = row.get("foreign_bought_today", row.get("foreign_roll20_today_up", False))
    if isinstance(today_up, str):
        today_up = today_up.strip().lower() in {"1", "true", "yes", "y", "t"}
    if not bool(today_up):
        return False
    return True


def _price_heat_reason(row: pd.Series, model: ModelConfig) -> str:
    reasons = []
    for label, value in _price_heat_flags(row, model)[:4]:
        if "ATR" in label:
            reasons.append(f"{label} {value:.1f}배")
        else:
            reasons.append(f"{label} {value:+.1%}")
    return " · ".join(reasons) if reasons else "과열 없음"


def _price_heat_penalty(row: pd.Series) -> float:
    endpoint_heat = max(
        8.0 * scale(row.get("price_return1", 0.0), 0.02, 0.08),
        10.0 * scale(row.get("price_return3", 0.0), 0.03, 0.10),
        12.0 * scale(row.get("price_return5", 0.0), 0.03, 0.12),
        14.0 * scale(row.get("price_return10", 0.0), 0.06, 0.18),
        16.0 * scale(row.get("price_return20", 0.0), 0.05, 0.20),
    )
    path_heat = max(
        14.0 * scale(row.get("rebound_from_low10", 0.0), 0.08, 0.20),
        18.0 * scale(row.get("rebound_from_low20", 0.0), 0.10, 0.25),
        10.0 * scale(row.get("rebound_atr_distance20", 0.0), 2.5, 7.0),
    )
    trend_heat = (
        4.0 * scale(row.get("ma20_atr_distance", 0.0), 1.5, 4.0)
        + 3.0 * scale(row.get("up_days10", 0.0), 6.0, 10.0)
        + 2.0 * scale(row.get("price_up_streak", 0.0), 3.0, 6.0)
    )
    volume_heat = (
        4.0
        * scale(row.get("rebound_volume_ratio20", 1.0), 1.2, 2.4)
        * scale(row.get("rebound_from_low20", 0.0), 0.10, 0.25)
    )
    near_high_heat = (
        2.0
        * scale(row.get("distance_high20", -1.0), -0.06, 0.0)
        * scale(row.get("rebound_from_low20", 0.0), 0.10, 0.20)
    )
    # Endpoint returns and the low-to-current path are correlated, so only the
    # larger core penalty is fully counted. Trend and volume confirm the move.
    return clamp(max(endpoint_heat, path_heat) + 0.5 * trend_heat + volume_heat + near_high_heat, 0.0, 32.0)


def _campaign_regime(row: pd.Series, model: ModelConfig) -> str:
    current = row.get("foreign_20_adtv", 0.0)
    recent = row.get("foreign_5_adtv", 0.0)
    recent3 = row.get("foreign_3_adtv", recent)
    long_inventory = _long_inventory20(row)
    institution = row.get("institution_20_adtv", 0.0)
    concentration = row.get("foreign_flow_concentration20", 0.0)
    buy_ratio = row.get("tolerant_buy_ratio20", 0.0)
    sellback = row.get("sellback_ratio20", 9.0)
    acceleration = row.get("foreign_acceleration5", 0.0)
    hot = _is_price_hot(row, model)

    if (
        row.get("max_abs_daily_return20", 0.0) >= model.corporate_action_daily_return_pct / 100.0
        or row.get("price_return20", 0.0) <= model.structural_break_return20_pct / 100.0
    ):
        return "이벤트·구조확인"
    if concentration >= 0.36 and buy_ratio < 0.50 and abs(current) < 0.35:
        return "이상대량·확인"
    if current < -0.35 and not (long_inventory > 0.25 and institution > 0.50 and recent3 > 0):
        return "분산·이탈"
    if long_inventory > 0.25 and current <= 0 and institution > 0.50 and recent3 > 0:
        return "손바뀜·재확인"
    if institution >= 0.30 and current < 0.20 and (recent <= 0.15 or sellback > 0.80):
        return "기관주도·외인확인"
    if current >= 0.35 and recent >= 0.20 and long_inventory < -0.10:
        return "신규재진입"
    # A live campaign buying a down day is still accumulation, not a chase.
    if _is_dip_absorption(row, model):
        if long_inventory > 0.25 and recent > 0.15 and acceleration > 0.02:
            return "장기매집·재가속"
        if current >= 0.45 and buy_ratio >= 0.55 and sellback <= 0.80:
            return "지속매집"
        if current > 0.20:
            return "초기매집"
    if hot and (current > 0.20 or long_inventory > 0.25):
        return "상승진행"
    if long_inventory > 0.25 and current > 0.20 and recent < 0 and sellback <= 0.80:
        return "장기매집·휴식"
    if long_inventory > 0.25 and recent > 0.15 and acceleration > 0.02:
        return "장기매집·재가속"
    if current >= 0.45 and buy_ratio >= 0.55 and sellback <= 0.80:
        return "지속매집"
    if current > 0.20:
        return "초기매집"
    if long_inventory > 0.25:
        return "장기재고·확인"
    return "중립"


def _campaign_strength(row: pd.Series, model: ModelConfig) -> tuple[float, bool]:
    current = row.get("foreign_20_adtv", 0.0)
    amount = row.get("foreign_net20", 0.0)
    long_inventory = _long_inventory20(row)
    concentration = row.get("foreign_flow_concentration20", 0.0)
    large_flow = amount >= model.large_flow_threshold_krw
    strength = (
        32.0 * scale(current, 0.0, 2.0)
        + 5.0 * scale(amount, 0.0, model.large_flow_threshold_krw)
        + 18.0 * scale(row.get("tolerant_buy_ratio20", 0.0), 0.45, 0.75)
        + 12.0 * (1.0 - scale(row.get("sellback_ratio20", 9.0), 0.25, 1.0))
        + 18.0 * scale(long_inventory, 0.0, 1.5)
        + 10.0 * scale(row.get("foreign_acceleration5", 0.0), -0.03, 0.12)
        + 5.0 * (1.0 - scale(concentration, 0.25, 0.55))
        + 8.0 * scale(
            row.get("foreign_buy_days10", row.get("foreign_roll20_up_days10", 0.0)),
            5.0,
            10.0,
        )
        + 4.0 * scale(
            row.get("foreign_buy_streak", row.get("foreign_roll20_up_streak", 0.0)),
            3.0,
            10.0,
        )
    )
    return clamp(strength, 0.0, 100.0), large_flow


def _discovery_score(
    row: pd.Series,
    campaign_strength: float,
    underreaction: float,
    fundamentals: float,
    stage: str,
    campaign_regime: str,
    distribution_score: float,
    dip_absorption: bool = False,
    steady_accumulation: bool = False,
) -> float:
    """Foreign accumulation quality before timing, without rewarding falling knives."""
    ownership_days = row.get("foreign_ownership_days_adtv", np.nan)
    holding_change = row.get("foreign_holding_change_adtv", np.nan)
    if pd.isna(ownership_days):
        ownership = 4.0
    else:
        ownership = 7.0 * scale(ownership_days, 1.0, 20.0)
    if pd.isna(holding_change):
        ownership += 1.5
    else:
        ownership += 3.0 * scale(holding_change, -0.5, 2.0)
    value = (
        0.65 * campaign_strength
        + 0.70 * underreaction
        + 0.45 * fundamentals
        + ownership
    )
    stage_adjustment = {
        "초기매집": 6.0,
        "매집·눌림": 3.0,
        "매집휴식": 3.0,
        "가격상승": -18.0,
        "수익실현": -25.0,
        "분산·이탈": -30.0,
        "이벤트확인": -30.0,
    }.get(stage, 0.0)
    regime_adjustment = {
        "지속매집": 2.0,
        "장기매집·재가속": 4.0,
        "장기매집·휴식": -8.0,
        "상승진행": -12.0,
        "분산·이탈": -20.0,
        "이벤트·구조확인": -25.0,
    }.get(campaign_regime, 0.0)
    # Price weakness is useful only while it remains controlled. The model
    # still rewards absorption, but a 20-day collapse beyond -15% no longer
    # creates an ever-higher "underreaction" score. A campaign that is buying
    # a 3-day pullback is not a knife.
    falling_knife_penalty = 14.0 * scale(-row.get("price_return20", 0.0), 0.15, 0.25)
    if not dip_absorption:
        falling_knife_penalty += 8.0 * scale(-row.get("price_return3", 0.0), 0.03, 0.08)
    ma20_bonus = 3.0 if dip_absorption and bool(row.get("ma20_support")) else 0.0
    steady_bonus = 0.0
    if steady_accumulation:
        steady_bonus = 4.0
        if float(row.get("price_return1", 0.0) or 0.0) <= -0.008:
            steady_bonus += 3.0
    return clamp(
        value
        + stage_adjustment
        + regime_adjustment
        + ma20_bonus
        + steady_bonus
        - falling_knife_penalty
        - 0.20 * distribution_score,
        0.0,
        100.0,
    )


def _timing_score(row: pd.Series) -> float:
    """Measure the transition from absorption to a tradable price/flow turn."""
    foreign_turn = (
        15.0 * scale(row.get("foreign_3_adtv", 0.0), -0.05, 0.40)
        + 10.0 * scale(row.get("foreign_5_adtv", 0.0), -0.10, 0.60)
        + 10.0 * scale(row.get("foreign_acceleration3", 0.0), -0.05, 0.12)
    )
    price_turn = (
        10.0 * scale(row.get("price_acceleration3", 0.0), -0.03, 0.06)
        + 7.0 * scale(row.get("price_return3", 0.0), -0.03, 0.025)
        + (5.0 if bool(row.get("above_ma5", False)) else 0.0)
        + (3.0 if bool(row.get("above_ma10", False)) else 0.0)
        + (6.0 if bool(row.get("ma20_support")) else 0.0)
        + (
            3.0
            if bool(row.get("touched_ma20_3")) and not bool(row.get("ma20_support"))
            else 0.0
        )
        + 5.0 * scale(row.get("ma5_slope", 0.0), -0.02, 0.03)
    )
    unextended = (
        12.0 * (1.0 - scale(row.get("price_return20", 0.0), 0.03, 0.12))
        + 8.0 * (1.0 - scale(row.get("rebound_from_low20", 0.0), 0.05, 0.15))
    )
    confirmation = (
        5.0 * scale(row.get("up_down_volume_ratio10", 1.0), 0.75, 1.60)
        + 5.0 * scale(row.get("price_residual3", 0.0), -0.03, 0.03)
        + 5.0 * scale(row.get("market_return20", 0.0), -0.15, 0.02)
    )
    return clamp(foreign_turn + price_turn + unextended + confirmation, 0.0, 100.0)


def _distribution_score(row: pd.Series) -> float:
    """Evidence that accumulated inventory is being distributed into strength."""
    selling = (
        24.0 * scale(-row.get("foreign_1_adtv", 0.0), 0.0, 0.45)
        + 20.0 * scale(-row.get("foreign_3_adtv", 0.0), 0.0, 0.65)
        + 14.0 * scale(row.get("sellback_ratio20", 0.0), 0.65, 1.20)
    )
    price_strength = max(
        12.0 * scale(row.get("gap_pct", 0.0), 0.03, 0.08),
        12.0 * scale(row.get("price_return10", 0.0), 0.06, 0.18),
        12.0 * scale(row.get("rebound_from_low20", 0.0), 0.10, 0.25),
    )
    tape = (
        12.0 * scale(row.get("upper_wick", 0.0), 0.25, 0.60)
        + 8.0 * scale(row.get("foreign_flow_concentration20", 0.0), 0.30, 0.55)
    )
    return clamp(selling + price_strength + tape, 0.0, 100.0)


def _entry_opportunity(
    row: pd.Series,
    campaign_strength: float,
    underreaction: float,
    structure: float,
    fundamentals: float,
    institution_overlay: float,
    campaign_regime: str,
    large_flow: bool,
    model: ModelConfig,
) -> tuple[float, float]:
    value = (
        0.50 * campaign_strength
        + 0.80 * underreaction
        + 0.80 * structure
        + 0.60 * fundamentals
        + 0.40 * max(institution_overlay, 0.0)
        + (6.0 if large_flow else 0.0)
    )
    heat_penalty = _price_heat_penalty(row)
    regime_penalty = {
        "기관주도·외인확인": 6.0,
        "장기재고·확인": 5.0,
        "이상대량·확인": 12.0,
        "분산·이탈": 20.0,
        "이벤트·구조확인": 25.0,
    }.get(campaign_regime, 0.0)
    return clamp(value - heat_penalty - regime_penalty, 0.0, 100.0), heat_penalty


def _stage(row: pd.Series, model: ModelConfig) -> str:
    inventory = (
        row.get("foreign_ewm120", 0.0) > 0.001
        or row.get("foreign_60_adtv", 0.0) > 0.5
    )
    strong = (
        row.get("foreign_20_adtv", 0.0) >= 0.45
        and row.get("foreign_flow_share20", 0.0) >= 0.01
        and row.get("tolerant_buy_ratio20", 0.0) >= 0.55
        and row.get("sellback_ratio20", 9.0) <= 0.75
    )
    recent = row.get("foreign_5_adtv", 0.0)
    today = row.get("foreign_1_adtv", 0.0)
    price20 = row.get("price_residual20", row.get("price_return20", 0.0))
    absolute20 = row.get("price_return20", 0.0)
    holdings_change = row.get("foreign_holding_change", np.nan)

    abnormal_day = row.get("max_abs_daily_return20", 0.0) >= model.corporate_action_daily_return_pct / 100.0
    structural_break = absolute20 <= model.structural_break_return20_pct / 100.0
    if abnormal_day or structural_break:
        return "이벤트확인"

    handoff = _long_inventory20(row) > 0.25 and row.get("institution_20_adtv", 0.0) > 0.50 and row.get("foreign_3_adtv", 0.0) > 0
    if (
        (row.get("foreign_20_adtv", 0.0) < -0.35 and not handoff)
        or (pd.notna(holdings_change) and holdings_change < 0 and recent < -0.15)
    ):
        return "분산·이탈"
    if (strong or (inventory and row.get("foreign_20_adtv", 0.0) > 0.15)) and absolute20 > 0.05 and today < -0.08:
        return "수익실현"
    if _is_dip_absorption(row, model) and (
        inventory or strong or row.get("foreign_20_adtv", 0.0) > 0.20
    ):
        return "매집·눌림"
    fast_price_move = _is_price_hot(row, model)
    if (inventory or strong) and fast_price_move and today >= -0.08:
        return "가격상승"
    if strong and price20 <= 0.05 and absolute20 < model.absolute_chase_return20_pct / 100.0:
        return "매집·눌림"
    if inventory and -0.15 <= today <= 0.05 and recent > -0.25:
        return "매집휴식"
    if row.get("foreign_20_adtv", 0.0) > 0.20:
        return "초기매집"
    return "중립"


def _absorption(row: pd.Series, quality: float, model: ModelConfig) -> tuple[str, float]:
    falling = row.get("price_return20", 0.0) < -0.01
    buying = row.get("foreign_20_adtv", 0.0) > 0.35
    structure_ok = (
        row.get("price_return60", 0.0) > -0.25
        and row.get("distance_high60", -1.0) > -0.30
        and quality >= 4.0
        and row.get("max_abs_daily_return20", 0.0) < model.corporate_action_daily_return_pct / 100.0
    )
    holding = row.get("foreign_holding_change", np.nan)
    holdings_ok = pd.isna(holding) or holding >= 0
    if falling and buying and structure_ok and holdings_ok:
        amount_share = row.get("absorption_share20", 0.0)
        bonus = 7.0 * scale(amount_share, 0.10, 0.65)
        return ("강함" if bonus >= 4.5 else "보통"), bonus
    if _is_dip_absorption(row, model) and structure_ok and holdings_ok:
        amount_share = row.get("absorption_share20", 0.0)
        bonus = max(
            7.0 * scale(amount_share, 0.10, 0.65),
            5.0 * scale(row.get("foreign_1_adtv", 0.0), 0.05, 0.40),
        )
        return ("강함" if bonus >= 4.5 else "보통"), bonus
    if falling and buying:
        return "주의", 0.5
    return "없음", 0.0


def _gap_action(row: pd.Series, threshold: float) -> str | None:
    if row.get("gap_pct", 0.0) < threshold:
        return None
    foreign_today = row.get("foreign_1_adtv", 0.0)
    weak_close = row.get("close", 0.0) < row.get("open", 0.0)
    if foreign_today < -0.20 and (weak_close or row.get("upper_wick", 0.0) >= 0.45):
        return "총 75% 익절·잔여 추적"
    if foreign_today < 0 or row.get("upper_wick", 0.0) >= 0.35:
        return "총 50% 익절·잔여 추적"
    return "25% 익절 후 보유"


def _entry_readiness(row: pd.Series, model: ModelConfig) -> tuple[str, bool, str]:
    """Separate an attractive accumulation candidate from an executable entry."""
    price3 = float(row.get("price_return3", 0.0))
    price20 = float(row.get("price_return20", 0.0))
    rebound20 = float(row.get("rebound_from_low20", 0.0))
    campaign = str(row.get("campaign_regime", "중립"))
    floor3 = model.entry_min_return3_pct / 100.0
    cap20 = model.entry_max_return20_pct / 100.0
    rebound_cap = model.entry_max_rebound20_pct / 100.0

    if price3 < floor3:
        return (
            "반전확인",
            False,
            f"3일 {price3:+.1%}: {floor3:+.1%} 이상으로 급락 진정 필요",
        )
    heat = []
    if price20 >= cap20:
        heat.append(f"20일 {price20:+.1%}")
    if rebound20 >= rebound_cap:
        heat.append(f"저점반등 {rebound20:+.1%}")
    if heat:
        return (
            "눌림대기",
            False,
            " · ".join(heat)
            + f": 20일 {cap20:+.1%} 미만·저점반등 {rebound_cap:+.1%} 미만 필요",
        )
    if campaign == "지속매집" and model.persistent_entry_requires_reacceleration:
        return (
            "재가속확인",
            False,
            "지속매집 단독 신호: 최근 외국인 재가속 국면 확인 필요",
        )
    return "진입가능", True, "단기 급락·20일 과열·저점반등 게이트 통과"


def _action(row: pd.Series, model: ModelConfig) -> str:
    stage = row.get("stage", "중립")
    campaign = row.get("campaign_regime", "중립")
    entry_score = row.get("entry_score", 0.0)
    entry_ready = bool(row.get("entry_gate_passed", False))
    readiness = str(row.get("entry_readiness", "관찰"))
    discovery_score = float(row.get("discovery_score", 0.0))
    timing_score = float(row.get("timing_score", 0.0))
    distribution_score = float(row.get("distribution_score", 0.0))
    if stage == "이벤트확인" or campaign == "이벤트·구조확인":
        return "기업 이벤트·급락 확인 전 매수 금지"
    gap_threshold = max(model.gap_take_profit_pct / 100.0, model.atr_gap_multiple * row.get("atr20_pct", 0.0))
    gap_action = _gap_action(row, gap_threshold)
    if gap_action:
        return gap_action

    score = row.get("score", 0.0)
    score_delta = row.get("score_delta", 0.0)
    if stage == "분산·이탈":
        return "비중 축소·신규매수 금지"
    if campaign == "이상대량·확인":
        return "블록·이벤트성 수급 확인"
    if campaign == "기관주도·외인확인":
        return "기관 주도·외국인 확인 대기"
    if stage == "수익실현":
        return "절반 익절·잔여 추적"
    if distribution_score >= model.distribution_exit_score and stage == "가격상승":
        return "수익실현 신호·절반 익절 검토"
    if campaign == "신규재진입":
        if not entry_ready:
            if readiness == "반전확인":
                return "급락 멈춤 확인·신규매수 대기"
            return "첫 눌림 대기·추격 금지"
        if discovery_score < model.discovery_candidate_score:
            return "매집 후보 강도 부족·대기"
        if distribution_score > model.entry_max_distribution_score:
            return "외국인 매도 압력 완화 대기"
        if model.timing_buy_score > 0 and timing_score < model.timing_buy_score:
            return "가격 반전·수급 재가속 확인 대기"
        if row.get("price_return5", 0.0) >= 0.03:
            return "첫 눌림·외국인 유지 확인"
        if entry_score >= model.entry_buy_score:
            return "초기 분할매수"
        return "재진입 신호 관찰"
    if campaign == "손바뀜·재확인":
        return "기관 흡수 후 외국인 재확인"
    if campaign == "장기매집·휴식" and row.get("foreign_3_adtv", 0.0) <= 0:
        return "외국인 재개 확인 후 분할매수"
    if stage == "가격상승" or campaign == "상승진행":
        return "추격 금지·보유"
    if row.get("absorption_signal") == "주의" and row.get("price_return20", 0.0) < 0:
        return "위험흡수 구조 확인 대기"
    # A large 20-day total is not enough for a fresh entry while the most
    # recent tape is still net selling. One- or two-day pauses remain
    # tolerable, but both the 3- and 5-day windows cannot be negative.
    if (
        campaign in {"지속매집", "장기매집·재가속"}
        and row.get("foreign_3_adtv", 0.0) <= 0
        and row.get("foreign_5_adtv", 0.0) <= 0
    ):
        return "외국인 재개 확인 후 분할매수"
    if bool(row.get("dip_absorption")) and not entry_ready:
        prefix = "외인 지속매수 중 " if bool(row.get("steady_accumulation")) else "매집 중 "
        still_above_ma20 = (
            not bool(row.get("ma20_support"))
            and float(row.get("ma20_distance", 0.0)) > 0.01
        )
        if still_above_ma20:
            return f"{prefix}하락흡수·20일선 접촉 대기"
        if readiness == "반전확인":
            return f"{prefix}하락흡수 확인·신규매수 대기"
        if readiness == "눌림대기":
            return f"{prefix}하락흡수·첫 눌림 대기"
        return f"{prefix}하락흡수 확인·신규매수 대기"
    if not entry_ready:
        if readiness == "반전확인":
            return "급락 멈춤 확인·신규매수 대기"
        if readiness == "눌림대기":
            return "첫 눌림 대기·추격 금지"
        if readiness == "재가속확인":
            return "외국인 재가속 확인 후 분할매수"
        return "진입 조건 확인 대기"
    if discovery_score < model.discovery_candidate_score:
        return "매집 후보 강도 부족·대기"
    if distribution_score > model.entry_max_distribution_score:
        return "외국인 매도 압력 완화 대기"
    if model.timing_buy_score > 0 and timing_score < model.timing_buy_score:
        return "가격 반전·수급 재가속 확인 대기"
    buy_campaigns = {"장기매집·재가속"}
    if not model.persistent_entry_requires_reacceleration:
        buy_campaigns.add("지속매집")
    if (
        entry_score >= model.entry_buy_score
        and campaign in buy_campaigns
        and stage in {"매집·눌림", "초기매집", "매집휴식"}
        and row.get("data_points", 0) >= 60
    ):
        return "초기 분할매수"
    if (
        score >= model.initial_buy_score
        and stage == "매집휴식"
        and row.get("price_return5", 0.0) <= 0
        and score_delta >= 0
    ):
        return "보유 시 분할 추가매수"
    if entry_score >= model.entry_watch_score or score >= model.watch_score:
        return "관찰·신호 강화 대기"
    return "대기"


def score_features(
    features: pd.DataFrame,
    model: ModelConfig,
    previous_state: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if features.empty:
        return features
    previous_symbols = (previous_state or {}).get("symbols", {})
    scored_rows: list[dict] = []

    for _, original in features.iterrows():
        row = original.copy()
        quality, outlook = _fundamental_score(row)
        holdings = row.get("foreign_holding_change", np.nan)
        holdings_score = 1.0 if pd.isna(holdings) else (2.0 if holdings > 0 else 0.0)

        accumulation = (
            12.0 * scale(row.get("foreign_20_adtv", 0.0), 0.0, 2.5)
            + 8.0 * scale(row.get("flow_residual20", 0.0), -0.002, 0.040)
            + 8.0 * scale(row.get("tolerant_buy_ratio20", 0.0), 0.35, 0.80)
            + 5.0 * (1.0 - scale(row.get("sellback_ratio20", 9.0), 0.15, 1.0))
            + 5.0 * scale(row.get("foreign_ewm120", 0.0), 0.0, 0.020)
            + holdings_score
        )
        accumulation = clamp(accumulation, 0.0, 40.0)

        expected_move = clamp(max(row.get("foreign_20_adtv", 0.0), 0.0) * 0.04, 0.0, 0.15)
        underreaction_gap = expected_move - row.get("price_residual20", 0.0)
        mismatch = 12.0 * scale(underreaction_gap, -0.02, 0.12)
        compression = 6.0 * (1.0 - scale(abs(row.get("price_residual20", 0.0)), 0.03, 0.18))
        absorption_label, absorption_bonus = _absorption(row, quality, model)
        underreaction = clamp(mismatch + compression + absorption_bonus, 0.0, 25.0)

        liquidity = 3.0 * scale(row.get("adtv20", 0.0), 1_000_000_000, 20_000_000_000)
        volatility = 3.0 * (1.0 - scale(row.get("atr20_pct", 0.0), 0.025, 0.090))
        volume = 4.0 * clamp(
            0.5 * scale(row.get("absorption_share20", 0.0), 0.05, 0.60)
            + 0.5 * (1.0 - scale(row.get("volume_ratio5_20", 1.0), 1.8, 4.0)),
            0.0,
            1.0,
        )
        not_extended = 3.0 * (1.0 - scale(row.get("price_return20", 0.0), 0.08, 0.25))
        base_intact = 2.0 if row.get("price_return60", 0.0) > -0.25 else 0.0
        structure = clamp(liquidity + volatility + volume + not_extended + base_intact, 0.0, 15.0)

        fundamentals = clamp(quality + outlook, 0.0, 20.0)
        institution_overlay = 5.0 * clamp(row.get("institution_20_adtv", 0.0) / 2.0, -1.0, 1.0)
        raw_score = accumulation + underreaction + structure + fundamentals + institution_overlay

        dip_absorption = _is_dip_absorption(row, model)
        steady_accumulation = is_steady_accumulation(row, model)
        stage = _stage(row, model)
        campaign_regime = _campaign_regime(row, model)
        campaign_strength, large_flow_signal = _campaign_strength(row, model)
        entry_score, price_heat_penalty = _entry_opportunity(
            row,
            campaign_strength,
            underreaction,
            structure,
            fundamentals,
            institution_overlay,
            campaign_regime,
            large_flow_signal,
            model,
        )
        readiness_row = row.copy()
        readiness_row["campaign_regime"] = campaign_regime
        entry_readiness, entry_gate_passed, entry_gate_reason = _entry_readiness(
            readiness_row, model
        )
        timing_score = _timing_score(row)
        distribution_score = _distribution_score(row)
        discovery_score = _discovery_score(
            row,
            campaign_strength,
            underreaction,
            fundamentals,
            stage,
            campaign_regime,
            distribution_score,
            dip_absorption,
            steady_accumulation,
        )
        signal_score = clamp(
            0.70 * discovery_score + 0.30 * timing_score - 0.20 * distribution_score,
            0.0,
            100.0,
        )
        penalty = 0.0
        if row.get("price_return20", 0.0) < -0.20 and absorption_label not in {"강함", "보통"}:
            penalty += 8.0
        if row.get("price_return20", 0.0) > 0.22:
            penalty += 5.0
        if stage == "분산·이탈":
            penalty += 10.0
        if stage == "이벤트확인":
            penalty += 20.0
        if row.get("data_points", 0) < 60:
            penalty += 2.0
        caution = str(row.get("investment_caution", "N")).upper() not in {"", "N", "0", "00", "NAN"}
        warning = str(row.get("market_warning", "00")).upper() not in {"", "N", "0", "00", "NAN"}
        managed = str(row.get("managed_status", "00")).upper() not in {"", "N", "0", "00", "NAN"}
        if caution or warning or managed:
            penalty += 15.0
        score = clamp(raw_score - penalty, 0.0, 100.0)

        symbol = str(row["symbol"]).zfill(6)
        previous = previous_symbols.get(symbol, {})
        previous_score = float(previous.get("score", score))
        score_delta = score - previous_score
        previous_entry = float(previous.get("entry_score", entry_score))
        entry_score_delta = entry_score - previous_entry
        stage_adjustment = {
            "매집·눌림": 0.04,
            "초기매집": 0.01,
            "매집휴식": 0.00,
            "가격상승": -0.02,
            "수익실현": -0.07,
            "분산·이탈": -0.12,
            "중립": -0.03,
            "이벤트확인": -0.15,
        }.get(stage, 0.0)
        heuristic_probability = clamp(0.22 + 0.0055 * signal_score + stage_adjustment, 0.25, 0.76)
        probability = _calibrated_probability(signal_score, heuristic_probability, calibration)
        target_return = clamp(max(0.06, 1.5 * row.get("atr20_pct", 0.0)), 0.06, 0.15)
        risk_return = clamp(max(0.04, 1.2 * row.get("atr20_pct", 0.0)), 0.04, 0.10)
        expected_value = probability * target_return - (1.0 - probability) * risk_return

        values = row.to_dict()
        values.update(
            {
                "accumulation_score": round(accumulation, 1),
                "underreaction_score": round(underreaction, 1),
                "structure_score": round(structure, 1),
                "fundamental_score": round(fundamentals, 1),
                "institution_overlay": round(institution_overlay, 1),
                "penalty": round(penalty, 1),
                "score": round(score, 1),
                "campaign_score": round(campaign_strength, 1),
                "entry_score": round(entry_score, 1),
                "discovery_score": round(discovery_score, 1),
                "timing_score": round(timing_score, 1),
                "distribution_score": round(distribution_score, 1),
                "signal_score": round(signal_score, 1),
                "entry_score_delta": round(entry_score_delta, 1),
                "price_heat_penalty": round(price_heat_penalty, 1),
                "price_heat_flag": _is_price_hot(row, model),
                "price_heat_reason": _price_heat_reason(row, model),
                "dip_absorption": dip_absorption,
                "steady_accumulation": steady_accumulation,
                "score_delta": round(score_delta, 1),
                "stage": stage,
                "campaign_regime": campaign_regime,
                "entry_readiness": entry_readiness,
                "entry_gate_passed": entry_gate_passed,
                "entry_gate_reason": entry_gate_reason,
                "large_flow_signal": large_flow_signal,
                "absorption_signal": absorption_label,
                "success_probability": round(probability, 4),
                "probability_source": "워크포워드 보정" if _calibration_active(calibration) else "미보정 추정",
                "expected_value": round(expected_value, 4),
                "is_new": symbol not in previous_symbols,
            }
        )
        values["action"] = _action(pd.Series(values), model)
        scored_rows.append(values)

    result = pd.DataFrame(scored_rows)
    return result.sort_values(
        ["discovery_score", "timing_score", "entry_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
