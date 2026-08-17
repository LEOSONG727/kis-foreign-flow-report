from __future__ import annotations

import html
import logging
import os
import smtplib
from email.message import EmailMessage
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


def _email_config() -> dict[str, object] | None:
    host = os.getenv("EMAIL_SMTP_HOST", "").strip()
    user = os.getenv("EMAIL_SMTP_USER", "").strip()
    password = os.getenv("EMAIL_SMTP_PASSWORD", "").strip()
    to_raw = os.getenv("EMAIL_TO", "").strip()
    if not (host and user and password and to_raw):
        return None
    port_text = os.getenv("EMAIL_SMTP_PORT", "587").strip() or "587"
    try:
        port = int(port_text)
    except ValueError:
        port = 587
    ssl_flag = os.getenv("EMAIL_SMTP_SSL", "").strip().lower()
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "from_addr": os.getenv("EMAIL_FROM", user).strip() or user,
        "to_addrs": [item.strip() for item in to_raw.replace(";", ",").split(",") if item.strip()],
        "use_ssl": ssl_flag in {"1", "true", "yes"} or port == 465,
    }


def _as_of_label(frame: pd.DataFrame) -> str:
    if "as_of" not in frame or frame.empty:
        return ""
    return pd.to_datetime(frame["as_of"].max()).strftime("%m-%d")


def _brief_frame(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> pd.DataFrame:
    return scored if "watch_tier" in scored else annotate_watch_brief(scored, model, brief)


def build_telegram_text(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> str:
    frame = _brief_frame(scored, brief, model)
    as_of = _as_of_label(frame)
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


def build_email_subject(scored: pd.DataFrame) -> str:
    frame = scored if not scored.empty else pd.DataFrame()
    as_of = _as_of_label(frame)
    return f"외인 매집 브리핑{f' · {as_of}' if as_of else ''}"


def build_email_bodies(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> tuple[str, str]:
    frame = _brief_frame(scored, brief, model)
    rows = list(iter_brief_rows(frame, brief))
    as_of = _as_of_label(frame)
    title = f"오늘의 외인 매집 브리핑{f' · {as_of}' if as_of else ''}"
    if not rows:
        plain = f"{title}\n\n오늘 사람이 볼 매집 후보가 없습니다.\n"
        html_body = f"<h2>{html.escape(title)}</h2><p>오늘 사람이 볼 매집 후보가 없습니다.</p>"
        return plain, html_body

    plain_lines = [title, "", f"오늘 볼 종목 {len(rows)}"]
    cards: list[str] = []
    for rank, row in enumerate(rows, 1):
        name = str(row.get("name", ""))
        symbol = str(row.get("symbol", "")).zfill(6)
        tier = str(row.get("watch_tier", ""))
        flow = str(row.get("flow_summary", ""))
        block = str(row.get("block_reason", ""))
        nxt = str(row.get("next_trigger", ""))
        discovery = float(row.get("discovery_score", 0.0) or 0.0)
        price20 = format_pct(float(row.get("price_return20", 0.0) or 0.0))
        rebound = format_pct(float(row.get("rebound_from_low20", 0.0) or 0.0))
        reason_label = "할 일" if tier in {"즉시매수", "추가매수"} else "막힘"
        plain_lines.extend(
            [
                "",
                f"{rank}. {name} ({symbol}) · {tier} · 발견 {discovery:.0f}",
                f"수급: {flow}",
                f"가격: 20일 {price20} · 반등 {rebound}",
                f"{reason_label}: {block}",
                f"다음: {nxt}",
            ]
        )
        cards.append(
            "<div style='border:1px solid #d8dee6;border-radius:10px;padding:14px 16px;margin:0 0 12px'>"
            f"<div style='color:#667084;font-size:12px'>{html.escape(tier)} · 발견 {discovery:.0f}</div>"
            f"<div style='font-size:18px;font-weight:700;margin:4px 0 8px'>{rank}. {html.escape(name)} "
            f"<span style='color:#667084;font-size:13px'>({html.escape(symbol)})</span></div>"
            f"<div>수급: {html.escape(flow)}</div>"
            f"<div>가격: 20일 {html.escape(price20)} · 반등 {html.escape(rebound)}</div>"
            f"<div style='margin-top:8px'><b>{html.escape(reason_label)}</b>: {html.escape(block)}</div>"
            f"<div><b>다음</b>: {html.escape(nxt)}</div>"
            "</div>"
        )
    plain_lines.extend(["", "즉시매수 게이트는 그대로입니다. 주문은 실행하지 않습니다.", "상세 HTML 리포트를 첨부합니다."])
    html_body = (
        "<div style='font-family:Segoe UI,Apple SD Gothic Neo,sans-serif;color:#111827;max-width:680px'>"
        f"<h2 style='margin:0 0 12px'>{html.escape(title)}</h2>"
        f"<p>오늘 볼 종목 {len(rows)}</p>"
        + "".join(cards)
        + "<p style='color:#667084;font-size:13px'>즉시매수 게이트는 그대로입니다. 주문은 실행하지 않습니다. "
        "상세 HTML 리포트를 첨부합니다.</p></div>"
    )
    return "\n".join(plain_lines), html_body


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


def send_email(
    scored: pd.DataFrame,
    report_path: Path,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> bool:
    config = _email_config()
    if not config:
        LOGGER.info("Email SMTP secrets are not configured; notification skipped")
        return False
    plain, html_body = build_email_bodies(scored, brief, model)
    message = EmailMessage()
    message["Subject"] = build_email_subject(scored)
    message["From"] = str(config["from_addr"])
    message["To"] = ", ".join(config["to_addrs"])  # type: ignore[arg-type]
    message.set_content(plain)
    message.add_alternative(html_body, subtype="html")
    if report_path.exists():
        message.add_attachment(
            report_path.read_bytes(),
            maintype="text",
            subtype="html",
            filename=report_path.name,
        )
    if config["use_ssl"]:
        client: smtplib.SMTP = smtplib.SMTP_SSL(str(config["host"]), int(config["port"]), timeout=20)
    else:
        client = smtplib.SMTP(str(config["host"]), int(config["port"]), timeout=20)
    with client:
        if not config["use_ssl"]:
            client.starttls()
        client.login(str(config["user"]), str(config["password"]))
        client.send_message(message)
    LOGGER.info("Email report delivered to %s", message["To"])
    return True


def deliver_notifications(
    scored: pd.DataFrame,
    report_path: Path,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> list[str]:
    """Send Telegram and email independently. Failures do not invalidate the report."""
    sent: list[str] = []
    try:
        if send_telegram(scored, report_path, brief, model):
            sent.append("telegram")
    except Exception as exc:
        LOGGER.error("Telegram delivery failed without invalidating the report: %s", exc)
    try:
        if send_email(scored, report_path, brief, model):
            sent.append("email")
    except Exception as exc:
        LOGGER.error("Email delivery failed without invalidating the report: %s", exc)
    if not sent:
        LOGGER.info("No notification channel delivered a report")
    return sent
