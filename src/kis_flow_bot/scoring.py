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


def _calibrated_probability(score: float, heuristic: float, calibration: dict[str, Any] | None) -> float:
    if not calibration or not calibration.get("active"):
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
    hot = (
        row.get("price_return20", 0.0) >= model.absolute_chase_return20_pct / 100.0
        or row.get("price_return5", 0.0) >= model.recent_chase_return5_pct / 100.0
    )

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
        25.0 * scale(current, 0.0, 2.0)
        + 15.0 * scale(amount, 0.0, model.large_flow_threshold_krw)
        + 15.0 * scale(row.get("tolerant_buy_ratio20", 0.0), 0.45, 0.75)
        + 10.0 * (1.0 - scale(row.get("sellback_ratio20", 9.0), 0.25, 1.0))
        + 20.0 * scale(long_inventory, 0.0, 1.5)
        + 10.0 * scale(row.get("foreign_acceleration5", 0.0), -0.03, 0.12)
        + 5.0 * (1.0 - scale(concentration, 0.25, 0.55))
    )
    return clamp(strength, 0.0, 100.0), large_flow


def _entry_opportunity(
    row: pd.Series,
    campaign_strength: float,
    underreaction: float,
    structure: float,
    fundamentals: float,
    institution_overlay: float,
    campaign_regime: str,
    large_flow: bool,
) -> tuple[float, float]:
    value = (
        0.50 * campaign_strength
        + 0.80 * underreaction
        + 0.80 * structure
        + 0.60 * fundamentals
        + 0.40 * max(institution_overlay, 0.0)
        + (6.0 if large_flow else 0.0)
    )
    heat_penalty = (
        16.0 * scale(row.get("price_return20", 0.0), 0.08, 0.22)
        + 12.0 * scale(row.get("price_return5", 0.0), 0.05, 0.15)
    )
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
    absolute5 = row.get("price_return5", 0.0)
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
    fast_price_move = (
        absolute20 >= model.absolute_chase_return20_pct / 100.0
        or absolute5 >= model.recent_chase_return5_pct / 100.0
    )
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


def _action(row: pd.Series, model: ModelConfig) -> str:
    stage = row.get("stage", "중립")
    campaign = row.get("campaign_regime", "중립")
    entry_score = row.get("entry_score", 0.0)
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
    if campaign == "신규재진입":
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
    if (
        entry_score >= model.entry_strong_score
        and campaign in {"지속매집", "장기매집·재가속"}
        and stage in {"매집·눌림", "초기매집", "매집휴식"}
        and row.get("data_points", 0) >= 60
    ):
        return "강한 신규매수"
    if (
        entry_score >= model.entry_buy_score
        and campaign in {"지속매집", "초기매집", "장기매집·재가속", "장기매집·휴식"}
        and stage in {"매집·눌림", "초기매집", "매집휴식"}
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
        heuristic_probability = clamp(0.24 + 0.0052 * score + stage_adjustment, 0.30, 0.78)
        probability = _calibrated_probability(score, heuristic_probability, calibration)
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
                "entry_score_delta": round(entry_score_delta, 1),
                "price_heat_penalty": round(price_heat_penalty, 1),
                "score_delta": round(score_delta, 1),
                "stage": stage,
                "campaign_regime": campaign_regime,
                "large_flow_signal": large_flow_signal,
                "absorption_signal": absorption_label,
                "success_probability": round(probability, 4),
                "probability_source": "워크포워드 보정" if calibration and calibration.get("active") else "미보정 추정",
                "expected_value": round(expected_value, 4),
                "is_new": symbol not in previous_symbols,
            }
        )
        values["action"] = _action(pd.Series(values), model)
        scored_rows.append(values)

    result = pd.DataFrame(scored_rows)
    return result.sort_values(
        ["entry_score", "campaign_score", "score"], ascending=[False, False, False]
    ).reset_index(drop=True)
