from __future__ import annotations

import pandas as pd

from kis_flow_bot.config import ModelConfig
from kis_flow_bot.portfolio import apply_portfolio_actions


def _scored(close: float = 94_000.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "symbol": "000010", "close": close, "score": 78.0,
            "entry_score": 74.0, "campaign_score": 82.0,
            "campaign_regime": "지속매집", "foreign_5_adtv": 0.5,
            "price_return5": -0.02, "price_return20": 0.01,
            "stage": "매집·눌림", "action": "초기 분할매수",
        }]
    )


def test_new_position_gets_configured_initial_amount():
    result = apply_portfolio_actions(_scored(), pd.DataFrame(), ModelConfig()).iloc[0]
    assert result["suggested_order_krw"] == 10_000_000
    assert not result["has_position"]
    assert result["decision_bucket"] == "신규매수"
    assert result["promotion_condition"] == "현재 조건 충족"


def test_held_position_gets_add_action_after_price_decline():
    portfolio = pd.DataFrame(
        [{"symbol": "000010", "quantity": 100, "average_price": 100_000, "last_buy_price": 100_000, "tranche_count": 1}]
    )
    result = apply_portfolio_actions(_scored(94_000), portfolio, ModelConfig()).iloc[0]
    assert result["action"] == "2차 분할 추가매수 검토"
    assert result["suggested_order_krw"] == 5_000_000
    assert result["decision_bucket"] == "추가매수"
    assert result["add_score"] > 0


def test_exit_action_is_never_overwritten_by_portfolio_logic():
    scored = _scored()
    scored.loc[0, "stage"] = "분산·이탈"
    scored.loc[0, "action"] = "비중 축소·신규매수 금지"
    portfolio = pd.DataFrame(
        [{"symbol": "000010", "quantity": 100, "average_price": 100_000, "last_buy_price": 100_000, "tranche_count": 2}]
    )
    result = apply_portfolio_actions(scored, portfolio, ModelConfig()).iloc[0]
    assert result["action"] == "비중 축소·신규매수 금지"
    assert result["decision_bucket"] == "수익실현·중단"


def test_profit_taking_action_is_only_shown_as_execution_for_a_held_position():
    scored = _scored()
    scored.loc[0, "stage"] = "수익실현"
    scored.loc[0, "action"] = "절반 익절·잔여 추적"
    result = apply_portfolio_actions(scored, pd.DataFrame(), ModelConfig()).iloc[0]
    assert result["action"] == "수익실현 신호·신규 추격 금지"


def test_hot_reentry_is_sorted_into_pullback_wait_queue():
    scored = _scored()
    scored.loc[0, "campaign_regime"] = "신규재진입"
    scored.loc[0, "action"] = "첫 눌림·외국인 유지 확인"
    scored.loc[0, "price_return5"] = 0.12
    result = apply_portfolio_actions(scored, pd.DataFrame(), ModelConfig()).iloc[0]
    assert result["decision_bucket"] == "눌림·확인대기"
    assert "첫 조정" in result["promotion_condition"]
