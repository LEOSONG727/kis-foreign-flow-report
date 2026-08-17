from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kis_flow_bot.brief import annotate_watch_brief
from kis_flow_bot.notify import (
    build_email_bodies,
    build_email_subject,
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
    assert "오늘 볼 종목" in telegram
    assert "이노션" in telegram
    assert "수급:" in telegram
    assert subject == "외인 매집 브리핑 · 08-14"
    assert "이노션" in plain
    assert "3일" in plain
    assert "이노션" in html_body
    assert "상세 HTML 리포트" in html_body


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
