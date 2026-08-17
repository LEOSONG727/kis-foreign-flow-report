from __future__ import annotations

import html
import logging
import os
from pathlib import Path

import pandas as pd
import requests

from .brief import annotate_watch_brief, format_pct, iter_brief_rows
from .config import BriefConfig, ModelConfig

LOGGER = logging.getLogger(__name__)


def _telegram_config() -> tuple[str, str] | None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return (token, chat_id) if token and chat_id else None


def build_telegram_text(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> str:
    frame = scored if "watch_tier" in scored else annotate_watch_brief(scored, model, brief)
    as_of = ""
    if "as_of" in frame and not frame.empty:
        as_of = pd.to_datetime(frame["as_of"].max()).strftime("%m-%d")
    title = f"오늘의 외인 매집 브리핑{f' · {as_of}' if as_of else ''}"
    lines = [f"<b>{html.escape(title)}</b>", ""]
    rows = list(iter_brief_rows(frame, brief))
    if not rows:
        lines.append("오늘 사람이 볼 매집 후보가 없습니다.")
    else:
        lines.append(f"<b>오늘 볼 종목 {len(rows)}</b>")
        for rank, row in enumerate(rows, 1):
            name = html.escape(str(row.get("name", "")))
            symbol = html.escape(str(row.get("symbol", "")).zfill(6))
            tier = html.escape(str(row.get("watch_tier", "")))
            flow = html.escape(str(row.get("flow_summary", "")))
            block = html.escape(str(row.get("block_reason", "")))
            nxt = html.escape(str(row.get("next_trigger", "")))
            discovery = float(row.get("discovery_score", 0.0) or 0.0)
            reason_label = "할 일" if str(row.get("watch_tier", "")) in {"즉시매수", "추가매수"} else "막힘"
            lines.append(
                f"{rank}. <b>{name}</b> ({symbol}) · {tier} · 발견 {discovery:.0f}\n"
                f"수급: {flow}\n"
                f"가격: 20일 {html.escape(format_pct(float(row.get('price_return20', 0.0) or 0.0)))}"
                f" · 반등 {html.escape(format_pct(float(row.get('rebound_from_low20', 0.0) or 0.0)))}\n"
                f"{reason_label}: {block}\n"
                f"다음: {nxt}"
            )
    lines.append("")
    lines.append("<i>즉시매수 게이트는 그대로입니다. 주문은 실행하지 않습니다.</i>")
    return "\n".join(lines)


def send_telegram(
    scored: pd.DataFrame,
    report_path: Path,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> bool:
    config = _telegram_config()
    if not config:
        LOGGER.info("Telegram secrets are not configured; notification skipped")
        return False
    token, chat_id = config
    api = f"https://api.telegram.org/bot{token}"
    response = requests.post(
        f"{api}/sendMessage",
        data={"chat_id": chat_id, "text": build_telegram_text(scored, brief, model), "parse_mode": "HTML"},
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

