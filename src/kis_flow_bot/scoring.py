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
        outlook = 8.0 * scale(upside, -0.10, 0.40)
    return quality, outlook


def _stage(row: pd.Series) -> str:
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
    price20 = row.get("price_residual20", row.get("price_return20", 0.0))
    holdings_change = row.get("foreign_holding_change", np.nan)

    if (
        row.get("foreign_20_adtv", 0.0) < -0.35
        or (pd.notna(holdings_change) and holdings_change < 0 and recent < -0.15)
    ):
        return "분산·이탈"
    if inventory and row.get("price_return20", 0.0) > 0.05 and recent < -0.20:
        return "수익실현"
    if inventory and row.get("price_return20", 0.0) > 0.08 and recent >= -0.05:
        return "가격상승"
    if strong and price20 <= 0.05:
        return "매집·눌림"
    if inventory and -0.25 <= recent <= 0.15:
        return "매집휴식"
    if row.get("foreign_20_adtv", 0.0) > 0.20:
        return "초기매집"
    return "중립"


def _absorption(row: pd.Series, quality: float) -> tuple[str, float]:
    falling = row.get("price_return20", 0.0) < -0.01
    buying = row.get("foreign_20_adtv", 0.0) > 0.35
    structure_ok = (
        row.get("price_return60", 0.0) > -0.25
        and row.get("distance_high60", -1.0) > -0.30
        and quality >= 4.0
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
    foreign_today = row.get("foreign_5_adtv", 0.0)
    weak_close = row.get("close", 0.0) < row.get("open", 0.0)
    if foreign_today < -0.20 and (weak_close or row.get("upper_wick", 0.0) >= 0.45):
        return "총 75% 익절·잔여 추적"
    if foreign_today < 0 or row.get("upper_wick", 0.0) >= 0.35:
        return "총 50% 익절·잔여 추적"
    return "25% 익절 후 보유"


def _action(row: pd.Series, model: ModelConfig) -> str:
    gap_threshold = max(model.gap_take_profit_pct / 100.0, model.atr_gap_multiple * row.get("atr20_pct", 0.0))
    gap_action = _gap_action(row, gap_threshold)
    if gap_action:
        return gap_action

    score = row.get("score", 0.0)
    stage = row.get("stage", "중립")
    confidence = row.get("success_probability", 0.0)
    score_delta = row.get("score_delta", 0.0)
    if stage == "분산·이탈":
        return "비중 축소·신규매수 금지"
    if stage == "수익실현":
        return "절반 익절·잔여 추적"
    if stage == "가격상승":
        return "추격 금지·보유"
    if score >= model.strong_buy_score and confidence >= 0.62 and stage == "매집·눌림":
        return "강한 신규매수"
    if (
        score >= model.initial_buy_score
        and confidence >= 0.55
        and stage in {"매집·눌림", "초기매집"}
    ):
        return "초기 분할매수"
    if (
        score >= model.initial_buy_score
        and stage == "매집휴식"
        and row.get("price_return5", 0.0) <= 0
        and score_delta >= 0
    ):
        return "보유 시 분할 추가매수"
    if score >= model.watch_score:
        return "관찰·신호 강화 대기"
    return "대기"


def score_features(
    features: pd.DataFrame,
    model: ModelConfig,
    previous_state: dict[str, Any] | None = None,
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
        absorption_label, absorption_bonus = _absorption(row, quality)
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

        stage = _stage(row)
        penalty = 0.0
        if row.get("price_return20", 0.0) < -0.20 and absorption_label not in {"강함", "보통"}:
            penalty += 8.0
        if row.get("price_return20", 0.0) > 0.22:
            penalty += 5.0
        if stage == "분산·이탈":
            penalty += 10.0
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
        stage_adjustment = {
            "매집·눌림": 0.04,
            "초기매집": 0.01,
            "매집휴식": 0.00,
            "가격상승": -0.02,
            "수익실현": -0.07,
            "분산·이탈": -0.12,
            "중립": -0.03,
        }.get(stage, 0.0)
        probability = clamp(0.24 + 0.0052 * score + stage_adjustment, 0.30, 0.78)
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
                "score_delta": round(score_delta, 1),
                "stage": stage,
                "absorption_signal": absorption_label,
                "success_probability": round(probability, 4),
                "expected_value": round(expected_value, 4),
                "is_new": symbol not in previous_symbols,
            }
        )
        values["action"] = _action(pd.Series(values), model)
        scored_rows.append(values)

    result = pd.DataFrame(scored_rows)
    return result.sort_values(["score", "foreign_20_adtv"], ascending=[False, False]).reset_index(drop=True)
