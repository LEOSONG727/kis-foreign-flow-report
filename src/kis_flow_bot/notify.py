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
    cheap = sum(1 for row in rows if _is_cheap_row(row))
    if buy:
        return f"오늘 살 자리 {buy}개  ·  싼 매집 {cheap}개"
    if cheap:
        return f"살 타이밍 아님  ·  싼 매집 {cheap}개 주목"
    return f"살 타이밍 아님  ·  관찰 {len(rows)}개"


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


def _is_cheap_row(row: pd.Series | dict) -> bool:
    return _bool(row, "cheap_accumulation") or _text(row, "screen_bucket") == "싼매집"


def _timing(row: pd.Series | dict, model: ModelConfig | None = None) -> tuple[str, str]:
    """Human timing label and the single reason not to buy yet."""
    model = model or ModelConfig()
    tier = _text(row, "watch_tier")
    if _actionable(tier) or _bool(row, "entry_gate_passed"):
        return "지금 분할 검토", "즉시매수 게이트 통과"
    if _is_cheap_row(row):
        return "주목 · 아직 대기", _judgment_line(row, model)
    if _bool(row, "dip_absorption") or _text(row, "screen_bucket").startswith("하락흡수"):
        return "참고 · 사지 않음", _judgment_line(row, model)
    if tier == "감시":
        return "감시만", "20일선에서 멀리 오른 하락 매수"
    if tier == "급락진정":
        return "진정 대기", _judgment_line(row, model)
    return "관찰", _judgment_line(row, model)


def _card_fields(
    row: pd.Series | dict,
    model: ModelConfig | None = None,
) -> dict[str, str]:
    timing, reason = _timing(row, model)
    return {
        "name": _text(row, "name"),
        "symbol": _text(row, "symbol").zfill(6) if _text(row, "symbol") else "",
        "timing": timing,
        "reason": reason,
        "flow": _flow_line(row),
        "price": _price_line(row),
        "section": _section_of(row),
    }


def _section_of(row: pd.Series | dict) -> str:
    timing, _ = _timing(row)
    if timing.startswith("지금"):
        return "buy"
    if _is_cheap_row(row):
        return "cheap"
    return "other"


def _title(frame: pd.DataFrame) -> str:
    as_of = _as_of_label(frame)
    return f"외인 매집 브리핑{f'  ·  {as_of}' if as_of else ''}"


def _plain_card(rank: int, fields: dict[str, str]) -> list[str]:
    return [
        f"{rank}. {fields['name']}  {fields['symbol']}",
        f"타이밍  {fields['timing']}",
        f"이유  {fields['reason']}",
        f"수급  {fields['flow']}",
        f"가격  {fields['price']}",
    ]


def _section_title(key: str) -> str:
    return {
        "buy": "지금 살 타이밍",
        "cheap": "지금 주목  ·  싼 매집",
        "other": "참고",
    }.get(key, "참고")


def build_telegram_text(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> str:
    frame = _brief_frame(scored, brief, model)
    rows = list(iter_brief_rows(frame, brief))
    lines = [f"<b>{html.escape(_title(frame))}</b>"]
    if not rows:
        lines.extend(["", "오늘 주목할 싼 매집이 없습니다."])
        return "\n".join(lines)
    lines.append(html.escape(_count_line(rows)))
    grouped: dict[str, list[pd.Series]] = {"buy": [], "cheap": [], "other": []}
    for row in rows:
        grouped[_section_of(row)].append(row)
    rank = 1
    for key in ("buy", "cheap", "other"):
        part = grouped[key]
        if not part:
            continue
        lines.append(f"<b>{html.escape(_section_title(key))}</b>")
        for row in part:
            fields = _card_fields(row, model)
            card = (
                f"{rank}. <b>{html.escape(fields['name'])}</b>  "
                f"<code>{html.escape(fields['symbol'])}</code>\n"
                f"타이밍  {html.escape(fields['timing'])}\n"
                f"이유  {html.escape(fields['reason'])}\n"
                f"수급  {html.escape(fields['flow'])}\n"
                f"가격  {html.escape(fields['price'])}"
            )
            lines.append(card)
            rank += 1
    lines.append("<i>조회 전용 · 주문 미실행</i>")
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
        cards.append(
            "<div style='border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;margin:0 0 12px'>"
            f"<div style='color:#6b7280;font-size:12px;letter-spacing:0.02em'>"
            f"{html.escape(fields['timing'])}</div>"
            f"<div style='font-size:18px;font-weight:700;margin:6px 0 10px'>{rank}. {html.escape(fields['name'])} "
            f"<span style='color:#6b7280;font-size:13px;font-weight:500'>{html.escape(fields['symbol'])}</span></div>"
            f"<div style='color:#111827;line-height:1.55'>"
            f"<div><b>이유</b>  {html.escape(fields['reason'])}</div>"
            f"<div>수급  {html.escape(fields['flow'])}</div>"
            f"<div>가격  {html.escape(fields['price'])}</div>"
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
