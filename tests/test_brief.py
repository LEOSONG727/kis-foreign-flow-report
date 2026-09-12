from __future__ import annotations

import pandas as pd

from kis_flow_bot.brief import (
    annotate_watch_brief,
    classify_watch_tier,
    flow_summary,
    discovery_grade,
    is_cheap_accumulation,
    is_steady_price_ok,
    next_trigger,
    proposal_status,
    select_daily_brief,
    watch_priority,
)
from kis_flow_bot.scoring import is_steady_accumulation
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
        "steady_accumulation": False,
        "foreign_buy_days10": 4,
        "foreign_sell_days10": 6,
        "foreign_buy_streak": 1,
        "foreign_bought_today": False,
        "foreign_flow_concentration20": 0.20,
        "price_return1": 0.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_executable_buy_is_immediate_tier():
    assert classify_watch_tier(_row(action="초기 분할매수", decision_bucket="신규매수")) == "즉시매수"


def test_proposal_status_requires_both_discovery_quality_and_price_gate():
    ready = _row(
        discovery_score=79.0,
        action="초기 분할매수",
        decision_bucket="신규매수",
        watch_tier="즉시매수",
    )
    high_wait = _row(discovery_score=88.0, watch_tier="고확신눌림")
    low_ready = _row(
        discovery_score=71.0,
        action="초기 분할매수",
        decision_bucket="신규매수",
        watch_tier="즉시매수",
    )
    assert discovery_grade(ready) == "B"
    assert proposal_status(ready) == "오늘 신규매수 검토"
    assert proposal_status(high_wait) == "강한 후보·가격 대기"
    assert proposal_status(low_ready) == "관찰"


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
    assert list(brief["name"]) == ["오리온", "이노션"]
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
    assert list(brief["name"]) == ["매수주", "고확신", "관심"]


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


def test_flow_summary_names_rising_ma_and_bullish_divergence():
    text = flow_summary(_row(ma20_rising=True, ma20_falling=False, flow_price_divergence="bullish"))
    assert "20일선 우상향" in text
    assert "가격↓수급↑" in text


def test_falling_ma_without_divergence_loses_watch_priority():
    live = _row(ma20_rising=True, ma20_falling=False, ma20_support=True)
    falling = _row(ma20_rising=False, ma20_falling=True, flow_price_divergence="", ma20_support=True)
    assert watch_priority(live) > watch_priority(falling)
    assert "우상향 회복" in next_trigger(falling)


def test_bullish_divergence_at_ma20_support_raises_watch_priority():
    plain = _row(ma20_support=True, flow_price_divergence="")
    split = _row(ma20_support=True, flow_price_divergence="bullish")
    assert watch_priority(split) > watch_priority(plain)


def test_dip_absorption_is_a_watch_name_not_an_immediate_buy():
    row = _row(
        name="셀트리온형",
        symbol="068270",
        discovery_score=88.0,
        dip_absorption=True,
        campaign_regime="장기매집·재가속",
        stage="매집·눌림",
        action="매집 중 하락흡수 확인·신규매수 대기",
        entry_readiness="반전확인",
        entry_gate_passed=False,
        price_return1=-0.027,
        price_return3=-0.032,
        price_return20=0.133,
        rebound_from_low20=0.162,
        foreign_net1=19_274_000_000,
        foreign_net20=247_136_000_000,
        large_flow_signal=True,
        ma20_distance=0.06,
        touched_ma20=False,
        ma20_support=False,
    )
    assert classify_watch_tier(row) == "감시"
    assert "하락흡수" in flow_summary(row)
    assert classify_watch_tier(row) != "즉시매수"
    assert next_trigger(row) == "20일선 접촉 + 하락일 외인 매수 유지"
    assert not is_cheap_accumulation(row)


def test_dip_absorption_gets_a_brief_slot_among_watch_candidates():
    rows = [
        _row(name=f"관심{index}", symbol=f"00001{index}", discovery_score=70.0)
        for index in range(5)
    ]
    rows.append(
        _row(
            name="셀트리온형",
            symbol="068270",
            discovery_score=88.0,
            dip_absorption=True,
            campaign_regime="장기매집·재가속",
            stage="매집·눌림",
            action="매집 중 하락흡수 확인·신규매수 대기",
            entry_readiness="반전확인",
            entry_gate_passed=False,
            price_return1=-0.027,
            price_return3=-0.032,
            price_return20=0.133,
            rebound_from_low20=0.162,
            foreign_net1=19_274_000_000,
            foreign_net20=247_136_000_000,
            large_flow_signal=True,
        )
    )
    frame = annotate_watch_brief(pd.DataFrame(rows))
    brief = select_daily_brief(frame, BriefConfig(size=5))
    assert "셀트리온형" in set(brief["name"])
    assert frame.set_index("symbol").loc["068270", "watch_tier"] != "즉시매수"


def test_ma20_support_turns_dip_absorption_into_a_watch_candidate():
    row = _row(
        name="20일선접촉",
        symbol="068270",
        discovery_score=88.0,
        dip_absorption=True,
        ma20_support=True,
        touched_ma20=True,
        ma20_distance=0.003,
        campaign_regime="장기매집·재가속",
        stage="매집·눌림",
        action="매집 중 하락흡수 확인·신규매수 대기",
        entry_readiness="반전확인",
        entry_gate_passed=False,
        price_return1=-0.015,
        price_return3=-0.032,
    )
    assert classify_watch_tier(row) == "관심후보"
    assert "20일선 접촉" in flow_summary(row)
    assert "20일선 지지" in next_trigger(row)


def test_unextended_foreign_buying_is_cheap_accumulation():
    row = _row(
        price_return20=0.01,
        rebound_from_low20=0.02,
        ma20_distance=0.0,
        distance_high60=-0.12,
        foreign_20_adtv=0.80,
        discovery_score=72.0,
    )
    assert is_cheap_accumulation(row)
    assert classify_watch_tier(row) in {"관심후보", "눌림대기"}


def test_absorption_caution_is_not_called_cheap():
    row = _row(
        price_return20=-0.09,
        rebound_from_low20=0.01,
        ma20_distance=-0.19,
        distance_high60=-0.20,
        foreign_20_adtv=2.3,
        discovery_score=66.0,
        absorption_signal="주의",
        action="위험흡수 구조 확인 대기",
    )
    assert not is_cheap_accumulation(row)


def _steady_row(**overrides) -> pd.Series:
    values = dict(
        discovery_score=80.0,
        foreign_net20=220_000_000_000,
        foreign_net1=15_000_000_000,
        large_flow_signal=True,
        steady_accumulation=True,
        foreign_buy_days10=9,
        foreign_sell_days10=1,
        foreign_buy_streak=8,
        foreign_bought_today=True,
        foreign_flow_concentration20=0.16,
        sellback_ratio20=0.28,
        price_return1=-0.005,
        price_return3=-0.02,
        price_return20=0.14,
        rebound_from_low20=0.19,
        ma20_distance=0.07,
        campaign_regime="장기매집·재가속",
        stage="매집·눌림",
        action="차곡차곡 누적·하락일 흡수 확인",
        entry_readiness="눌림대기",
        dip_absorption=True,
    )
    values.update(overrides)
    return _row(**values)


def test_brief_keeps_cheap_majority_and_one_dip():
    rows = [
        _row(
            name=f"싼매집{index}",
            symbol=f"00002{index}",
            discovery_score=70.0 + index,
            price_return20=0.01,
            rebound_from_low20=0.02,
            ma20_distance=0.0,
            distance_high60=-0.12,
            foreign_20_adtv=0.70,
            action="관찰·신호 강화 대기",
            entry_readiness="눌림대기",
        )
        for index in range(4)
    ]
    for index in range(4):
        rows.append(
            _row(
                name=f"하락흡수{index}",
                symbol=f"00003{index}",
                discovery_score=88.0,
                dip_absorption=True,
                campaign_regime="장기매집·재가속",
                ma20_distance=0.08,
                touched_ma20=False,
                ma20_support=False,
                price_return1=-0.02,
                price_return3=-0.03,
                price_return20=0.12,
                rebound_from_low20=0.16,
                large_flow_signal=True,
                action="매집 중 하락흡수·20일선 접촉 대기",
                entry_readiness="반전확인",
            )
        )
    frame = annotate_watch_brief(pd.DataFrame(rows))
    brief = select_daily_brief(frame, BriefConfig(size=5, cheap_slots=3, dip_slots=1, leftover_slots=1, steady_slots=0))
    cheap_count = int(brief["cheap_accumulation"].sum())
    dip_count = int(
        brief["dip_absorption"].map(
            lambda value: bool(value) if pd.notna(value) else False
        ).sum()
    )
    assert cheap_count >= 3
    assert dip_count == 1
    assert len(brief) == 5


def test_brief_prefers_staircasing_inventory_over_cheap_and_generic_dip():
    rows = [
        _row(
            name=f"싼매집{index}",
            symbol=f"00002{index}",
            discovery_score=70.0 + index,
            price_return20=0.01,
            rebound_from_low20=0.02,
            ma20_distance=0.0,
            distance_high60=-0.12,
            foreign_20_adtv=0.70,
        )
        for index in range(4)
    ]
    for index in range(3):
        rows.append(
            _row(
                name=f"하락흡수{index}",
                symbol=f"00003{index}",
                discovery_score=88.0,
                dip_absorption=True,
                ma20_distance=0.08,
                price_return1=-0.02,
                price_return20=0.12,
                rebound_from_low20=0.16,
                large_flow_signal=True,
            )
        )
    rows.extend(
        [
            _steady_row(name="셀트리온형", symbol="068270"),
            _steady_row(name="한화오션형", symbol="042660", price_return1=-0.012),
            _steady_row(name="HMM형", symbol="011200", price_return1=0.004, dip_absorption=False),
        ]
    )
    frame = annotate_watch_brief(pd.DataFrame(rows))
    brief = select_daily_brief(frame, BriefConfig(size=5))
    names = set(brief["name"])
    assert {"셀트리온형", "한화오션형", "HMM형"}.issubset(names)
    assert int(brief["steady_accumulation"].sum()) >= 3
    assert is_steady_accumulation(_steady_row())
    assert "외인 순매수" in flow_summary(_steady_row())
    assert "순매수" in next_trigger(_steady_row())


def test_brief_prefers_relative_flow_strength_over_absolute_won_amount():
    rows = [
        _steady_row(
            name="작은고득점",
            symbol="241560",
            discovery_score=100.0,
            foreign_net20=16_000_000_000,
            foreign_20_adtv=1.80,
            large_flow_signal=False,
            watch_tier="관심후보",
        ),
        _steady_row(
            name="셀트리온",
            symbol="068270",
            discovery_score=88.0,
            foreign_net20=243_000_000_000,
            foreign_20_adtv=0.45,
            foreign_buy_days10=10,
        ),
        _steady_row(
            name="한화오션",
            symbol="042660",
            discovery_score=90.0,
            foreign_net20=264_000_000_000,
            foreign_20_adtv=0.55,
        ),
    ]
    frame = annotate_watch_brief(pd.DataFrame(rows))
    brief = select_daily_brief(frame, BriefConfig(size=2))
    assert list(brief["name"])[0] == "작은고득점"


def test_steady_price_ok_requires_a_controlled_pullback():
    cheap_enough = _steady_row(
        price_return1=-0.01,
        price_return3=-0.02,
        price_return20=0.04,
        rebound_from_low20=0.06,
        ma20_distance=0.008,
        ma20_support=True,
        near_ma20=True,
        watch_tier="관심후보",
    )
    too_hot = _steady_row(
        price_return20=0.14,
        rebound_from_low20=0.19,
        ma20_distance=0.07,
        watch_tier="고확신눌림",
    )
    assert is_steady_price_ok(cheap_enough)
    assert not is_steady_price_ok(too_hot)


def test_brief_puts_priced_staircases_ahead_of_expensive_ones():
    rows = [
        _steady_row(name="비싼메인", symbol="068270"),
        _steady_row(
            name="가격괜찮음",
            symbol="015760",
            foreign_net20=80_000_000_000,
            price_return1=-0.01,
            price_return3=-0.015,
            price_return20=0.02,
            rebound_from_low20=0.04,
            ma20_distance=0.005,
            ma20_support=True,
            near_ma20=True,
            watch_tier="관심후보",
        ),
    ]
    frame = annotate_watch_brief(pd.DataFrame(rows))
    brief = select_daily_brief(frame, BriefConfig(size=2))
    assert list(brief["name"]) == ["가격괜찮음", "비싼메인"]
    assert bool(frame.set_index("symbol").loc["015760", "steady_price_ok"])
    assert not bool(frame.set_index("symbol").loc["068270", "steady_price_ok"])


def test_telegram_explains_why_not_to_buy():
    scored = score_features(synthetic_features(), ModelConfig())
    text = build_telegram_text(scored)
    assert "매집후보" in text
    assert "오늘 결론" in text
    assert "진입" in text
    assert "막힘:" not in text
