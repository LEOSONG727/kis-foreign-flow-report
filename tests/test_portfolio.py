from __future__ import annotations

import pandas as pd

from kis_flow_bot.config import ModelConfig
from kis_flow_bot.portfolio import apply_portfolio_actions


def _scored(close: float = 94_000.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "symbol": "000010", "close": close, "score": 78.0,
            "entry_score": 74.0, "campaign_score": 82.0,
            "discovery_score": 78.0, "timing_score": 70.0,
            "campaign_regime": "지속매집", "foreign_5_adtv": 0.5,
            "foreign_3_adtv": 0.3, "price_return1": 0.0, "price_return3": 0.0,
            "price_return5": -0.02, "price_return20": 0.01, "distribution_score": 5.0,
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


def test_held_position_waits_when_price_is_still_falling_fast():
    scored = _scored(94_000)
    scored.loc[0, "price_return3"] = -0.03
    portfolio = pd.DataFrame(
        [{"symbol": "000010", "quantity": 100, "average_price": 100_000, "last_buy_price": 100_000, "tranche_count": 1}]
    )
    result = apply_portfolio_actions(scored, portfolio, ModelConfig()).iloc[0]
    assert result["action"] == "보유·추가매수 재확인 대기"
    assert result["suggested_order_krw"] == 0


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
    assert "20일 수익률 +3% 미만" in result["promotion_condition"]


def test_rising_campaign_requires_both_short_and_monthly_heat_to_cool():
    scored = _scored()
    scored.loc[0, "campaign_regime"] = "상승진행"
    scored.loc[0, "action"] = "추격 금지·보유"
    result = apply_portfolio_actions(scored, pd.DataFrame(), ModelConfig()).iloc[0]
    assert "20일 수익률 +3% 미만" in result["promotion_condition"]


def test_dip_absorption_keeps_buy_gate_and_explains_the_next_look():
    scored = _scored()
    scored.loc[0, "action"] = "매집 중 하락흡수·20일선 접촉 대기"
    scored.loc[0, "dip_absorption"] = True
    scored.loc[0, "entry_gate_passed"] = False
    scored.loc[0, "ma20_distance"] = 0.06
    scored.loc[0, "ma20_support"] = False
    result = apply_portfolio_actions(scored, pd.DataFrame(), ModelConfig()).iloc[0]
    assert result["action"] == "매집 중 하락흡수·20일선 접촉 대기"
    assert result["decision_bucket"] != "신규매수"
    assert "20일선 접촉" in result["promotion_condition"]


def test_low_rebound_is_the_binding_pullback_condition():
    scored = _scored()
    scored.loc[0, "campaign_regime"] = "상승진행"
    scored.loc[0, "action"] = "추격 금지·보유"
    scored.loc[0, "rebound_from_low20"] = 0.20
    result = apply_portfolio_actions(scored, pd.DataFrame(), ModelConfig()).iloc[0]
    assert "20일 저점반등 +5% 미만" in result["promotion_condition"]
