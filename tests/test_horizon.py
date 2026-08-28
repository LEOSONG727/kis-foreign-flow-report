from __future__ import annotations

import pandas as pd

from kis_flow_bot.brief import annotate_watch_brief
from kis_flow_bot.horizon import (
    annotate_horizon,
    build_horizon_board,
    long_horizon_score,
    mid_horizon_score,
)
from kis_flow_bot.notify import build_telegram_text
from kis_flow_bot.report import render_report
from datetime import datetime

from test_brief import _row


def _accum(**overrides) -> pd.Series:
    base = dict(
        as_of="2026-08-27",
        market="KOSPI",
        discovery_score=92.0,
        discovery_grade="A",
        accumulation_score=36.0,
        campaign_regime="지속매집",
        stage="매집·눌림",
        proposal_status="강한 후보·가격 대기",
        watch_tier="고확신눌림",
        foreign_net20=80_000_000_000,
        foreign_net60=180_000_000_000,
        foreign_net120=90_000_000_000,
        foreign_20_adtv=2.4,
        foreign_buy_days10=8,
        foreign_buy_days20=15,
        sellback_ratio20=0.12,
        large_flow_signal=True,
        steady_accumulation=True,
        distribution_score=6.0,
        price_return20=0.04,
        rebound_from_low20=0.07,
        discovery_eligible=True,
        absorption_signal="없음",
    )
    base.update(overrides)
    return _row(**base)


def test_mid_score_ignores_closed_price_gate():
    waiting = _accum(
        name="한국금융지주",
        symbol="071050",
        proposal_status="강한 후보·가격 대기",
        entry_gate_passed=False,
    )
    buyable = _accum(
        name="NH투자증권",
        symbol="005940",
        discovery_score=88.0,
        proposal_status="오늘 신규매수 검토",
        entry_gate_passed=True,
        foreign_net20=34_400_000_000,
        foreign_net60=66_900_000_000,
        large_flow_signal=False,
        foreign_buy_days10=6,
    )
    wait_score = mid_horizon_score(waiting, appearance_days=3)
    buy_score = mid_horizon_score(buyable, appearance_days=3)
    assert wait_score > 70
    assert wait_score > buy_score


def test_extended_name_is_penalized_but_terminal_is_zero():
    extended = _accum(price_return20=0.22, rebound_from_low20=0.28)
    dead = _accum(campaign_regime="분산·이탈", stage="분산·이탈")
    assert mid_horizon_score(extended) < mid_horizon_score(_accum())
    assert mid_horizon_score(dead) == 0
    assert long_horizon_score(dead) == 0


def test_sticky_core_does_not_churn_for_two_points():
    previous = {
        "as_of": "2026-08-26",
        "mid_core": [
            {"symbol": "071050", "name": "한국금융지주", "since": "2026-08-21", "score": 90.0},
            {"symbol": "207940", "name": "삼성바이오", "since": "2026-08-21", "score": 88.0},
            {"symbol": "001040", "name": "CJ", "since": "2026-08-21", "score": 84.0},
        ],
        "long_core": [],
        "tactical": [{"symbol": "000080", "name": "하이트진로", "status": "오늘 신규매수 검토"}],
    }
    frame = annotate_watch_brief(
        pd.DataFrame(
            [
                _accum(name="한국금융지주", symbol="071050", discovery_score=94.0),
                _accum(name="삼성바이오", symbol="207940", discovery_score=93.0, foreign_net60=400_000_000_000),
                _accum(name="CJ", symbol="001040", discovery_score=90.0, campaign_regime="장기매집·재가속"),
                _accum(name="BGF리테일", symbol="282330", discovery_score=99.0, rebound_from_low20=0.26, price_return20=0.20),
            ]
        )
    )
    board = build_horizon_board(frame, previous=previous)
    mid_symbols = [item.symbol for item in board.mid_core]
    assert mid_symbols == ["071050", "207940", "001040"]
    cj = next(item for item in board.mid_core if item.symbol == "001040")
    assert cj.streak_days >= 7
    assert not cj.is_new


def test_yesterday_tactical_dropout_is_not_a_sell():
    previous = {
        "tactical": [{"symbol": "000080", "name": "하이트진로", "status": "오늘 신규매수 검토"}],
        "mid_core": [],
        "long_core": [],
    }
    frame = annotate_watch_brief(
        pd.DataFrame(
            [
                _accum(
                    name="NH투자증권",
                    symbol="005940",
                    as_of="2026-08-27",
                    discovery_score=88.0,
                    action="초기 분할매수",
                    decision_bucket="신규매수",
                    entry_gate_passed=True,
                    price_return20=0.025,
                    rebound_from_low20=0.045,
                    watch_tier="즉시매수",
                ),
                _accum(
                    name="하이트진로",
                    symbol="000080",
                    as_of="2026-08-27",
                    discovery_score=69.9,
                    discovery_grade="C",
                    campaign_regime="초기매집",
                    stage="초기매집",
                    proposal_status="관찰",
                    foreign_net20=3_900_000_000,
                    foreign_net60=13_700_000_000,
                    large_flow_signal=False,
                    distribution_score=27.8,
                ),
            ]
        )
    )
    board = build_horizon_board(frame, previous=previous)
    kinds = {note.kind for note in board.continuity}
    assert "tactical_dropout" in kinds
    text = " ".join(note.text for note in board.continuity)
    assert "하이트진로" in text
    assert "팔지" in text
    assert any("NH투자증권" in line for line in board.verdict_lines)
    assert any("오늘 살 것" in line and "NH투자증권" in line for line in board.decision_lines)
    assert any("팔지 말 것" in line and "하이트진로" in line for line in board.decision_lines)


def test_telegram_and_html_keep_daily_list_then_horizon():
    frame = annotate_watch_brief(
        pd.DataFrame(
            [
                _accum(
                    name="NH투자증권",
                    symbol="005940",
                    as_of="2026-08-27",
                    action="초기 분할매수",
                    decision_bucket="신규매수",
                    entry_gate_passed=True,
                    price_return20=0.02,
                    rebound_from_low20=0.03,
                ),
                _accum(name="한국금융지주", symbol="071050", as_of="2026-08-27"),
                _accum(
                    name="삼성바이오",
                    symbol="207940",
                    as_of="2026-08-27",
                    foreign_net60=420_000_000_000,
                    foreign_net120=150_000_000_000,
                ),
            ]
        )
    )
    _, board = annotate_horizon(frame)
    telegram = build_telegram_text(frame, horizon=board)
    assert telegram.index("오늘의 핵심 요약") < telegram.index("오늘 판단")
    assert telegram.index("오늘 판단") < telegram.index("그래서 뭘 사나")
    assert "오늘 살 것" in telegram
    assert "중기 코어" in telegram
    rendered = render_report(frame, 5, (), datetime(2026, 8, 27, 20, 40), horizon=board)
    assert rendered.html.index("이 리스트만 보세요") < rendered.html.index("오늘 판단")
    assert "중기 코어" in rendered.html
    assert "오늘 판단" in rendered.markdown
    assert "그래서 뭘 사나" in rendered.markdown
