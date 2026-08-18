from __future__ import annotations

import html
import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import requests

from .brief import (
    annotate_watch_brief,
    format_eok,
    format_pct,
    iter_brief_rows,
    _bool,
    _float,
    _text,
)
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


def _as_of_timestamp(frame: pd.DataFrame) -> pd.Timestamp | None:
    if "as_of" not in frame or frame.empty:
        return None
    value = pd.to_datetime(frame["as_of"].max(), errors="coerce")
    return None if pd.isna(value) else pd.Timestamp(value)


def _as_of_label(frame: pd.DataFrame) -> str:
    stamp = _as_of_timestamp(frame)
    return f"{stamp.month}월 {stamp.day}일" if stamp is not None else ""


def _brief_frame(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> pd.DataFrame:
    return scored if "watch_tier" in scored else annotate_watch_brief(scored, model, brief)


def _actionable(tier: str) -> bool:
    return tier in {"즉시매수", "추가매수"}


def _count_line(rows: list[pd.Series]) -> str:
    buy = sum(1 for row in rows if _text(row, "watch_tier") == "즉시매수")
    add = sum(1 for row in rows if _text(row, "watch_tier") == "추가매수")
    watch = max(0, len(rows) - buy - add)
    parts = [f"오늘 {len(rows)}종목", f"즉시매수 {buy}"]
    if add:
        parts.append(f"추가매수 {add}")
    parts.append(f"관찰 {watch}")
    return "  ·  ".join(parts)


def _setup_tags(row: pd.Series | dict) -> list[str]:
    tags: list[str] = []
    if _bool(row, "dip_absorption"):
        tags.append("하락흡수")
    if _bool(row, "touched_ma20"):
        tags.append("20일선 접촉")
    elif _bool(row, "ma20_support") or _bool(row, "near_ma20"):
        tags.append("20일선 근접")
    if _bool(row, "large_flow_signal"):
        tags.append("500억+")
    campaign = _text(row, "campaign_regime")
    if campaign and campaign != "중립":
        tags.append(campaign)
    return tags


def _flow_line(row: pd.Series | dict) -> str:
    return (
        f'20일 {format_eok(_float(row, "foreign_net20"))}  ·  '
        f'3일 {format_eok(_float(row, "foreign_net3"))}  ·  '
        f'당일 {format_eok(_float(row, "foreign_net1"))}'
    )


def _price_line(row: pd.Series | dict) -> str:
    parts = [
        f'당일 {format_pct(_float(row, "price_return1"))}',
        f'20일 {format_pct(_float(row, "price_return20"))}',
    ]
    if "ma20_distance" in getattr(row, "index", []) or (
        isinstance(row, dict) and "ma20_distance" in row
    ):
        label = "접촉" if _bool(row, "touched_ma20") or _bool(row, "ma20_support") else "이격"
        parts.append(f'20일선 {format_pct(_float(row, "ma20_distance"))} {label}')
    rebound = _float(row, "rebound_from_low20")
    if rebound >= 0.05:
        parts.append(f"저점 {format_pct(rebound)}")
    return "  ·  ".join(parts)


def _judgment_line(row: pd.Series | dict, model: ModelConfig | None = None) -> str:
    model = model or ModelConfig()
    if _actionable(_text(row, "watch_tier")):
        return "게이트 통과 · 분할 신규 검토"
    price3 = _float(row, "price_return3")
    floor3 = model.entry_min_return3_pct / 100.0
    if price3 < floor3:
        return f"3일 {format_pct(price3)} · {model.entry_min_return3_pct:+.0f}% 이상으로 진정 필요"
    if (
        _bool(row, "dip_absorption")
        and not _bool(row, "ma20_support")
        and _float(row, "ma20_distance") > 0.01
    ):
        return f'20일선 {format_pct(_float(row, "ma20_distance"))} · 접촉 전'
    rebound = _float(row, "rebound_from_low20")
    rebound_cap = model.entry_max_rebound20_pct / 100.0
    if rebound >= rebound_cap:
        return f"저점반등 {format_pct(rebound)} · 과열 진정 필요"
    price20 = _float(row, "price_return20")
    cap20 = model.entry_max_return20_pct / 100.0
    if price20 >= cap20:
        return f"20일 {format_pct(price20)} · 과열 진정 필요"
    block = _text(row, "block_reason")
    if block:
        return block.split(" · ")[0]
    return _text(row, "action") or "추가 확인 필요"


def _card_fields(
    row: pd.Series | dict,
    model: ModelConfig | None = None,
) -> dict[str, str]:
    tier = _text(row, "watch_tier")
    return {
        "name": _text(row, "name"),
        "symbol": _text(row, "symbol").zfill(6) if _text(row, "symbol") else "",
        "tier": tier,
        "discovery": f"{_float(row, 'discovery_score'):.0f}",
        "flow": _flow_line(row),
        "tags": "  ·  ".join(_setup_tags(row)),
        "price": _price_line(row),
        "verdict_label": "할 일" if _actionable(tier) else "판단",
        "verdict": _judgment_line(row, model),
        "next": _text(row, "next_trigger") or "매집 신호 유지 여부 확인",
    }


def _title(frame: pd.DataFrame) -> str:
    as_of = _as_of_label(frame)
    return f"외인 매집 브리핑{f'  ·  {as_of}' if as_of else ''}"


def _plain_card(rank: int, fields: dict[str, str]) -> list[str]:
    lines = [
        f"{rank}. {fields['name']}  {fields['symbol']}",
        f"{fields['tier']}  ·  발견 {fields['discovery']}",
        "",
        f"수급  {fields['flow']}",
    ]
    if fields["tags"]:
        lines.append(f"특징  {fields['tags']}")
    lines.extend(
        [
            f"가격  {fields['price']}",
            f"{fields['verdict_label']}  {fields['verdict']}",
            f"다음  {fields['next']}",
        ]
    )
    return lines


def build_telegram_text(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> str:
    frame = _brief_frame(scored, brief, model)
    rows = list(iter_brief_rows(frame, brief))
    lines = [f"<b>{html.escape(_title(frame))}</b>"]
    if not rows:
        lines.extend(["", "오늘 검토할 매집 후보가 없습니다."])
        return "\n".join(lines)
    lines.extend([html.escape(_count_line(rows)), ""])
    for rank, row in enumerate(rows, 1):
        fields = _card_fields(row, model)
        name = html.escape(fields["name"])
        symbol = html.escape(fields["symbol"])
        card = [
            f"{rank}. <b>{name}</b>  <code>{symbol}</code>",
            f"{html.escape(fields['tier'])}  ·  발견 {html.escape(fields['discovery'])}",
            "",
            f"수급  {html.escape(fields['flow'])}",
        ]
        if fields["tags"]:
            card.append(f"특징  {html.escape(fields['tags'])}")
        card.extend(
            [
                f"가격  {html.escape(fields['price'])}",
                f"{html.escape(fields['verdict_label'])}  {html.escape(fields['verdict'])}",
                f"다음  {html.escape(fields['next'])}",
            ]
        )
        lines.append("\n".join(card))
    lines.extend(["", "<i>조회 전용 · 주문 미실행</i>"])
    return "\n\n".join(line for line in lines if line)


def build_email_subject(scored: pd.DataFrame) -> str:
    as_of = _as_of_label(scored if not scored.empty else pd.DataFrame())
    return f"외인 매집 브리핑{f' · {as_of}' if as_of else ''}"


def build_email_bodies(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> tuple[str, str]:
    frame = _brief_frame(scored, brief, model)
    rows = list(iter_brief_rows(frame, brief))
    title = _title(frame)
    if not rows:
        plain = f"{title}\n\n오늘 검토할 매집 후보가 없습니다.\n"
        html_body = f"<h2>{html.escape(title)}</h2><p>오늘 검토할 매집 후보가 없습니다.</p>"
        return plain, html_body

    counts = _count_line(rows)
    plain_lines = [title, counts]
    cards: list[str] = []
    for rank, row in enumerate(rows, 1):
        fields = _card_fields(row, model)
        plain_lines.extend(["", *_plain_card(rank, fields)])
        tag_html = (
            f"<div style='color:#4b5563;margin:2px 0'>특징  {html.escape(fields['tags'])}</div>"
            if fields["tags"]
            else ""
        )
        cards.append(
            "<div style='border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;margin:0 0 12px'>"
            f"<div style='color:#6b7280;font-size:12px;letter-spacing:0.02em'>"
            f"{html.escape(fields['tier'])}  ·  발견 {html.escape(fields['discovery'])}</div>"
            f"<div style='font-size:18px;font-weight:700;margin:6px 0 10px'>{rank}. {html.escape(fields['name'])} "
            f"<span style='color:#6b7280;font-size:13px;font-weight:500'>{html.escape(fields['symbol'])}</span></div>"
            f"<div style='color:#111827;line-height:1.55'>"
            f"<div>수급  {html.escape(fields['flow'])}</div>"
            f"{tag_html}"
            f"<div>가격  {html.escape(fields['price'])}</div>"
            f"<div style='margin-top:8px'><b>{html.escape(fields['verdict_label'])}</b>  {html.escape(fields['verdict'])}</div>"
            f"<div><b>다음</b>  {html.escape(fields['next'])}</div>"
            "</div></div>"
        )
    plain_lines.extend(["", "조회 전용 · 주문 미실행", "상세 HTML 리포트를 첨부합니다."])
    html_body = (
        "<div style='font-family:Segoe UI,Apple SD Gothic Neo,sans-serif;color:#111827;max-width:680px'>"
        f"<h2 style='margin:0 0 6px;font-size:22px'>{html.escape(title)}</h2>"
        f"<p style='color:#6b7280;margin:0 0 16px'>{html.escape(counts)}</p>"
        + "".join(cards)
        + "<p style='color:#6b7280;font-size:13px;margin:8px 0 0'>조회 전용 · 주문 미실행. "
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
