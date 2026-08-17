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
                "suggested_order_krw": suggested_order,
            }
        )
        enriched.append(row)
    return pd.DataFrame(enriched)
