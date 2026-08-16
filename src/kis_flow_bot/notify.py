from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


def _telegram_config() -> tuple[str, str] | None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return (token, chat_id) if token and chat_id else None


def send_telegram(scored: pd.DataFrame, report_path: Path) -> bool:
    config = _telegram_config()
    if not config:
        LOGGER.info("Telegram secrets are not configured; notification skipped")
        return False
    token, chat_id = config
    api = f"https://api.telegram.org/bot{token}"
    lines = ["<b>오늘의 외국인 매집 TOP 10</b>"]
    for rank, (_, row) in enumerate(scored.head(10).iterrows(), 1):
        lines.append(
            f"{rank}. <b>{row['name']}</b> {row['score']:.1f}점 · "
            f"{row['stage']} · {row['action']}"
        )
    lines.append("\n<i>확률은 백테스트 보정 전·주문 자동실행 없음</i>")
    response = requests.post(
        f"{api}/sendMessage",
        data={"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "HTML"},
        timeout=20,
    )
    response.raise_for_status()
    with report_path.open("rb") as handle:
        response = requests.post(
            f"{api}/sendDocument",
            data={"chat_id": chat_id, "caption": "모바일 HTML 상세 리포트"},
            files={"document": (report_path.name, handle, "text/html")},
            timeout=60,
        )
    response.raise_for_status()
    LOGGER.info("Telegram report delivered")
    return True

