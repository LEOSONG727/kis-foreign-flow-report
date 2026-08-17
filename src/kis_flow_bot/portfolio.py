from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig


PORTFOLIO_COLUMNS = [
    "symbol",
    "quantity",
    "average_price",
    "last_buy_price",
    "tranche_count",
]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _promotion_condition(row: dict, action: str, has_position: bool, model: ModelConfig) -> str:
    campaign = str(row.get("campaign_regime", "중립"))
    if action in {"강한 신규매수", "초기 분할매수"} or "추가매수 검토" in action:
        return "현재 조건 충족"
    if "위험흡수 구조 확인" in action:
        return "60일 가격구조 회복 + 외국인 보유수량 감소 없음 확인"
    if has_position and action == "보유·다음 눌림 대기":
        return f"마지막 매수가 대비 -{model.add_trigger_pct:.0f}% 도달 + 매집 신호 유지"
    if campaign == "신규재진입":
        return "첫 조정 후 최근 3일 외국인 순매수 유지"
    if campaign == "상승진행":
        return "5일 상승률 +3% 이하로 진정 + 외국인 5일 순매수 유지"
    if campaign == "장기매집·휴식":
        return f"최근 3일 외국인 순매수 전환 + 진입점수 {model.entry_buy_score:.0f}점 이상"
    if campaign == "손바뀜·재확인":
        return "외국인 5일 순매수 전환 + 기관 순매수 유지"
    if campaign == "기관주도·외인확인":
        return "외국인 20일 강도 0.45일 ADTV 이상으로 상승"
    if campaign == "이상대량·확인":
        return "대량거래 집중도 36% 미만 + 20일 매수일 비율 55% 이상"
    if campaign in {"분산·이탈", "이벤트·구조확인"}:
        return "현재 승격 대상 아님"
    return f"진입점수 {model.entry_watch_score:.0f}점 이상 + 외국인 20일 순매수 확인"


def _invalidation_condition(row: dict) -> str:
    campaign = str(row.get("campaign_regime", "중립"))
    if campaign == "이벤트·구조확인":
        return "기업 이벤트 확인 전 신호 무효"
    if campaign == "분산·이탈":
        return "외국인 20일 순매도 전환으로 이미 무효"
    if campaign == "기관주도·외인확인":
        return "기관 20일 순매도 전환 또는 60일 저점 이탈"
    if campaign == "손바뀜·재확인":
        return "기관 흡수 중단 + 외국인 최근 3일 재매도"
    return "외국인 20일 순매수 0 이하·되팔기 100% 이상·60일 저점 이탈"


def _decision_bucket(row: dict, action: str, has_position: bool, model: ModelConfig) -> str:
    campaign = str(row.get("campaign_regime", "중립"))
    if (
        any(token in action for token in ("익절", "비중 축소", "매수 금지", "이벤트"))
        or campaign in {"분산·이탈", "이벤트·구조확인", "이상대량·확인"}
    ):
        return "수익실현·중단"
    if has_position and "추가매수 검토" in action:
        return "추가매수"
    if not has_position and action in {"강한 신규매수", "초기 분할매수"}:
        return "신규매수"
    if has_position:
        return "보유관리"
    if any(token in action for token in ("눌림", "재개", "재확인", "확인 대기", "추격 금지")):
        return "눌림·확인대기"
    if float(row.get("entry_score", 0.0)) >= model.entry_watch_score:
        return "관찰"
    return "대기"


def _add_opportunity_score(row: dict, has_position: bool, position_drawdown: float) -> float:
    campaign = float(row.get("campaign_score", 0.0))
    entry = float(row.get("entry_score", 0.0))
    foreign5 = float(row.get("foreign_5_adtv", 0.0))
    price5 = float(row.get("price_return5", 0.0))
    price20 = float(row.get("price_return20", 0.0))
    score = 0.55 * campaign + 0.25 * entry
    score += 10.0 * _clamp(foreign5 / 0.50, 0.0, 1.0)
    score += 10.0 * _clamp(-price5 / 0.08, 0.0, 1.0)
    score -= 15.0 * _clamp((price20 - 0.08) / 0.14, 0.0, 1.0)
    if has_position and position_drawdown <= -0.03:
        score += 10.0
    if str(row.get("campaign_regime", "")) in {
        "상승진행", "기관주도·외인확인", "이상대량·확인", "분산·이탈", "이벤트·구조확인"
    }:
        score -= 15.0
    return round(_clamp(score), 1)


def load_portfolio(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=PORTFOLIO_COLUMNS)
        empty.to_csv(path, index=False, encoding="utf-8")
        return empty
    frame = pd.read_csv(path, dtype={"symbol": "string"})
    for column in PORTFOLIO_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame["symbol"] = frame["symbol"].astype("string").str.zfill(6)
    for column in PORTFOLIO_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame[PORTFOLIO_COLUMNS].drop_duplicates("symbol", keep="last")


def apply_portfolio_actions(
    scored: pd.DataFrame,
    portfolio: pd.DataFrame,
    model: ModelConfig,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    positions = portfolio.set_index("symbol") if not portfolio.empty else pd.DataFrame()
    enriched: list[dict] = []
    protected_actions = {
        "절반 익절·잔여 추적",
        "총 75% 익절·잔여 추적",
        "총 50% 익절·잔여 추적",
        "25% 익절 후 보유",
        "비중 축소·신규매수 금지",
        "기업 이벤트·급락 확인 전 매수 금지",
    }
    profit_actions = {
        "절반 익절·잔여 추적",
        "총 75% 익절·잔여 추적",
        "총 50% 익절·잔여 추적",
        "25% 익절 후 보유",
    }

    for _, original in scored.iterrows():
        row = original.to_dict()
        symbol = str(row["symbol"]).zfill(6)
        position = positions.loc[symbol] if not positions.empty and symbol in positions.index else None
        quantity = float(position.get("quantity", 0.0)) if position is not None else 0.0
        average_price = float(position.get("average_price", 0.0)) if position is not None else 0.0
        last_buy_price = float(position.get("last_buy_price", 0.0)) if position is not None else 0.0
        tranche_count = int(position.get("tranche_count", 0)) if position is not None else 0
        has_position = quantity > 0
        close = float(row.get("close", 0.0))
        reference_price = last_buy_price if last_buy_price > 0 else average_price
        position_drawdown = close / reference_price - 1.0 if reference_price > 0 else np.nan
        model_action = str(row.get("action", "대기"))
        action = model_action
        suggested_order = 0

        if not has_position:
            if model_action in profit_actions:
                action = "수익실현 신호·신규 추격 금지"
            elif model_action == "추격 금지·보유":
                action = "추격 금지·첫 눌림 대기"
            elif model_action in {"강한 신규매수", "초기 분할매수"}:
                suggested_order = model.initial_order_krw
        elif model_action not in protected_actions:
            stage = str(row.get("stage", "중립"))
            score = float(row.get("score", 0.0))
            trigger = reference_price * (1.0 - model.add_trigger_pct / 100.0) if reference_price > 0 else 0.0
            signal_alive = stage in {"매집·눌림", "초기매집", "매집휴식"} and score >= model.initial_buy_score
            if stage == "가격상승":
                action = "보유·추격매수 금지"
            elif signal_alive and trigger > 0 and close <= trigger and tranche_count < model.max_tranches:
                action = f"{tranche_count + 1}차 분할 추가매수 검토"
                suggested_order = model.add_order_krw
            elif signal_alive and tranche_count >= model.max_tranches:
                action = "보유·추가매수 한도 도달"
            elif signal_alive:
                action = "보유·다음 눌림 대기"
            elif score >= model.watch_score:
                action = "보유·신호 유지"
            else:
                action = "보유·추가매수 중단"

        add_score = _add_opportunity_score(
            row,
            has_position,
            float(position_drawdown) if pd.notna(position_drawdown) else 0.0,
        )
        decision_bucket = _decision_bucket(row, action, has_position, model)
        if decision_bucket == "신규매수":
            action_priority = float(row.get("entry_score", 0.0))
        elif decision_bucket == "추가매수":
            action_priority = add_score
        elif decision_bucket in {"눌림·확인대기", "관찰"}:
            action_priority = 0.6 * float(row.get("campaign_score", 0.0)) + 0.4 * float(row.get("entry_score", 0.0))
        elif decision_bucket == "수익실현·중단":
            action_priority = 100.0 - float(row.get("entry_score", 0.0))
        else:
            action_priority = add_score

        row.update(
            {
                "model_action": model_action,
                "action": action,
                "has_position": has_position,
                "position_quantity": quantity,
                "average_price": average_price if has_position else np.nan,
                "last_buy_price": last_buy_price if has_position else np.nan,
                "tranche_count": tranche_count,
                "position_return": close / average_price - 1.0 if has_position and average_price > 0 else np.nan,
                "position_drawdown": position_drawdown,
                "add_score": add_score,
                "decision_bucket": decision_bucket,
                "action_priority": round(_clamp(action_priority), 1),
                "promotion_condition": _promotion_condition(row, action, has_position, model),
                "invalidation_condition": _invalidation_condition(row),
                "suggested_order_krw": suggested_order,
            }
        )
        enriched.append(row)
    return pd.DataFrame(enriched)
