from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kis_flow_bot.brief import annotate_watch_brief
from kis_flow_bot.notify import (
    build_email_bodies,
    build_email_subject,
    build_headline,
    build_telegram_text,
    deliver_notifications,
)
from test_brief import _row


def _scored() -> pd.DataFrame:
    return annotate_watch_brief(
        pd.DataFrame(
            [
                _row(name="이노션", symbol="214320", as_of="2026-08-14"),
                _row(
                    name="오리온",
                    symbol="271560",
                    as_of="2026-08-14",
                    discovery_score=91.8,
                    rebound_from_low20=0.082,
                ),
            ]
        )
    )


def test_email_and_telegram_share_the_daily_brief():
    scored = _scored()
    telegram = build_telegram_text(scored)
    subject = build_email_subject(scored)
    plain, html_body = build_email_bodies(scored)
    assert "외인 매집 브리핑" in telegram
    assert "오늘의 핵심 요약" in telegram
    assert "이 리스트만 보세요" in telegram
    assert telegram.index("이 리스트만 보세요") < telegram.index("타이밍")
    assert "이노션" in telegram
    assert "타이밍" in telegram
    assert "수급" in telegram
    assert "막힘:" not in telegram
    assert "조회 전용" in telegram
    assert subject == "외인 매집 브리핑 · 8월 14일"
    assert "이노션" in plain
    assert "타이밍" in plain
    assert "이노션" in html_body
    assert "상세 HTML 리포트" in html_body


def test_telegram_card_puts_setup_and_ma20_on_separate_lines():
    scored = annotate_watch_brief(
        pd.DataFrame(
            [
                _row(
                    name="셀트리온",
                    symbol="068270",
                    as_of="2026-08-18",
                    discovery_score=88.0,
                    dip_absorption=True,
                    large_flow_signal=True,
                    campaign_regime="장기매집·재가속",
                    ma20_distance=0.06,
                    touched_ma20=False,
                    ma20_support=False,
                    price_return1=-0.027,
                    price_return3=-0.032,
                    price_return20=0.133,
                    rebound_from_low20=0.162,
                    foreign_net1=19_274_000_000,
                    foreign_net20=247_136_000_000,
                    action="매집 중 하락흡수·20일선 접촉 대기",
                    entry_readiness="반전확인",
                    next_trigger="20일선 접촉 + 하락일 외인 매수 유지",
                )
            ]
        )
    )
    text = build_telegram_text(scored)
    assert "셀트리온" in text
    assert "타이밍" in text
    assert "사지 않음" in text
    assert "20일선" in text
    assert "막힘" not in text


def test_deliver_skips_when_secrets_are_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for key in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "EMAIL_SMTP_HOST",
        "EMAIL_SMTP_USER",
        "EMAIL_SMTP_PASSWORD",
        "EMAIL_TO",
    ):
        monkeypatch.delenv(key, raising=False)
    report = tmp_path / "latest.html"
    report.write_text("<html>ok</html>", encoding="utf-8")
    assert deliver_notifications(_scored(), report) == []


def test_headline_lists_names_before_detail_and_explains_large_flow_dropout():
    scored = annotate_watch_brief(
        pd.DataFrame(
            [
                _row(
                    name="한국전력",
                    symbol="015760",
                    as_of="2026-08-14",
                    discovery_score=67.6,
                    price_return20=-0.023,
                    rebound_from_low20=0.039,
                    ma20_distance=-0.039,
                    foreign_net20=77_424_000_000,
                    large_flow_signal=True,
                    action="외국인 재개 확인 후 분할매수",
                ),
                _row(
                    name="삼성전자",
                    symbol="005930",
                    as_of="2026-08-14",
                    discovery_score=36.7,
                    campaign_regime="중립",
                    stage="중립",
                    price_return20=0.076,
                    rebound_from_low20=0.451,
                    ma20_distance=0.123,
                    foreign_net20=1_040_935_000_000,
                    large_flow_signal=True,
                    action="첫 눌림 대기·추격 금지",
                ),
            ]
        )
    )
    headline = build_headline(scored)
    telegram = build_telegram_text(scored)
    plain, html_body = build_email_bodies(scored)
    assert [item.name for item in headline.items] == ["한국전력"]
    assert "500억+" in headline.funnel
    assert "최종 1개" in headline.funnel
    assert "이미 오름" in headline.funnel
    assert "강도 약함" in headline.funnel
    assert telegram.index("오늘의 핵심 요약") < telegram.index("수급")
    assert "이 리스트만 보세요" in telegram
    assert "이 리스트만 보세요" in plain
    assert "이 리스트만 보세요" in html_body
    assert "삼성전자" not in " ".join(item.name for item in headline.items)
