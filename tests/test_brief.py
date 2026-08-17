from __future__ import annotations

import pandas as pd

from kis_flow_bot.brief import (
    annotate_watch_brief,
    classify_watch_tier,
    flow_summary,
    select_daily_brief,
    watch_priority,
)
from kis_flow_bot.config import BriefConfig, ModelConfig
from kis_flow_bot.notify import build_telegram_text
from kis_flow_bot.scoring import score_features

from test_features import synthetic_features


def _row(**overrides) -> pd.Series:
    base = {
        "symbol": "000010",
        "name": "테스트",
        "discovery_score": 78.0,
        "timing_score": 55.0,
        "distribution_score": 5.0,
        "campaign_regime": "지속매집",
        "stage": "매집·눌림",
        "action": "첫 눌림 대기·추격 금지",
        "decision_bucket": "눌림·확인대기",
        "entry_readiness": "눌림대기",
        "entry_gate_passed": False,
        "price_return3": 0.0,
        "price_return20": 0.045,
        "rebound_from_low20": 0.06,
        "foreign_net1": 400_000_000,
        "foreign_net3": 1_200_000_000,
        "foreign_net5": 1_800_000_000,
        "foreign_net20": 3_100_000_000,
        "foreign_3_adtv": 0.18,
        "foreign_20_adtv": 0.80,
        "buy_days20": 11,
        "current_buy_streak": 3,
        "sellback_ratio20": 0.15,
        "foreign_acceleration3": 0.04,
        "absorption_signal": "없음",
        "large_flow_signal": False,
        "promotion_condition": "20일 수익률 +3% 미만·20일 저점반등 +5% 미만 + 외국인 매집 유지",
    }
    base.update(overrides)
    return pd.Series(base)


def test_executable_buy_is_immediate_tier():
    assert classify_watch_tier(_row(action="초기 분할매수", decision_bucket="신규매수")) == "즉시매수"


def test_slightly_extended_name_is_watch_candidate_not_buy():
    assert classify_watch_tier(_row()) == "관심후보"


def test_high_discovery_extended_name_is_conviction_pullback():
    row = _row(
        discovery_score=91.8,
        price_return20=0.044,
        rebound_from_low20=0.082,
    )
    assert classify_watch_tier(row) == "고확신눌림"


def test_falling_knife_is_risk_not_top_brief():
    knife = _row(
        name="급락주",
        symbol="000030",
        discovery_score=84.9,
        price_return20=-0.133,
        rebound_from_low20=0.115,
        absorption_signal="주의",
        action="위험흡수 구조 확인 대기",
        entry_readiness="눌림대기",
    )
    ready = _row(name="이노션", symbol="000020")
    conviction = _row(
        name="오리온",
        symbol="000040",
        discovery_score=91.8,
        rebound_from_low20=0.082,
    )
    frame = annotate_watch_brief(pd.DataFrame([knife, ready, conviction]))
    assert frame.set_index("symbol").loc["000030", "watch_tier"] == "위험확인"
    brief = select_daily_brief(frame, BriefConfig(size=2))
    assert list(brief["name"]) == ["이노션", "오리온"]
    assert "급락주" not in set(brief["name"])


def test_daily_brief_prefers_ready_then_conviction_over_crash():
    buy = _row(
        name="매수주",
        symbol="000011",
        action="초기 분할매수",
        decision_bucket="신규매수",
        entry_gate_passed=True,
        price_return20=0.01,
        rebound_from_low20=0.02,
    )
    frame = annotate_watch_brief(pd.DataFrame([
        buy,
        _row(name="관심", symbol="000012"),
        _row(name="고확신", symbol="000013", discovery_score=88.0, rebound_from_low20=0.11),
        _row(
            name="급락",
            symbol="000014",
            discovery_score=90.0,
            price_return20=-0.14,
            absorption_signal="주의",
            action="위험흡수 구조 확인 대기",
        ),
    ]))
    brief = select_daily_brief(frame, BriefConfig(size=3))
    assert list(brief["name"]) == ["매수주", "관심", "고확신"]


def test_live_flow_ranks_above_stale_twenty_day_total():
    live = _row(name="살아있는수급", symbol="000021")
    stale = _row(
        name="식은수급",
        symbol="000022",
        foreign_net1=-800_000_000,
        foreign_net3=-1_500_000_000,
        foreign_net5=-400_000_000,
        foreign_net20=8_000_000_000,
        foreign_3_adtv=-0.20,
        foreign_20_adtv=1.40,
        buy_days20=6,
        current_buy_streak=0,
        sellback_ratio20=0.95,
        foreign_acceleration3=-0.06,
        large_flow_signal=True,
    )
    assert watch_priority(live) > watch_priority(stale)
    frame = annotate_watch_brief(pd.DataFrame([stale, live]))
    brief = select_daily_brief(frame, BriefConfig(size=1))
    assert list(brief["name"]) == ["살아있는수급"]


def test_flow_summary_shows_recent_windows():
    text = flow_summary(_row())
    assert "20일" in text
    assert "3일" in text
    assert "당일" in text


def test_telegram_explains_why_not_to_buy():
    scored = score_features(synthetic_features(), ModelConfig())
    text = build_telegram_text(scored)
    assert "오늘 볼 종목" in text
    assert "다음:" in text
    assert "매집후보" in text
    assert ("막힘:" in text) or ("할 일:" in text)
