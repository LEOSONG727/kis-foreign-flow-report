from __future__ import annotations

import html
import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import requests

from .brief import (
    annotate_watch_brief,
    format_pct,
    iter_brief_rows,
    _bool,
    _float,
    _text,
)
from .config import BriefConfig, ModelConfig
from .horizon import HorizonBoard

LOGGER = logging.getLogger(__name__)
LARGE_FLOW_KRW = 50_000_000_000


@dataclass(frozen=True)
class HeadlineItem:
    rank: int
    name: str
    symbol: str
    timing: str
    reason: str
    stability: str
    entry_rule: str


@dataclass(frozen=True)
class Headline:
    verdict: str
    items: tuple[HeadlineItem, ...]
    wait_items: tuple[HeadlineItem, ...]
    funnel: str


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


def _numeric_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _large_flow_funnel(frame: pd.DataFrame, brief_rows: list[pd.Series]) -> str:
    """Why 20-day 500억+ names rarely reach the human shortlist."""
    if frame.empty:
        return ""
    large = frame.loc[_numeric_col(frame, "foreign_net20") >= LARGE_FLOW_KRW]
    if large.empty:
        return "20일 외인 500억+ 없음"
    brief_symbols = {
        _text(row, "symbol").zfill(6) for row in brief_rows if _text(row, "symbol")
    }
    symbols = (
        large["symbol"].astype(str).str.zfill(6)
        if "symbol" in large.columns
        else pd.Series("", index=large.index)
    )
    n_in = int(symbols.isin(brief_symbols).sum())
    drop = large.loc[~symbols.isin(brief_symbols)]
    reasons: list[str] = []
    if not drop.empty:
        already_up = float(
            (
                (_numeric_col(drop, "rebound_from_low20") >= 0.05)
                | (_numeric_col(drop, "price_return20") >= 0.03)
            ).mean()
        )
        weak = float((_numeric_col(drop, "discovery_score") < 65).mean())
        if already_up >= 0.4:
            reasons.append("대부분 이미 오름")
        if weak >= 0.4:
            reasons.append("금액만 크고 강도 약함")
    line = f"20일 외인 500억+ {len(large)}개 중 최종 {n_in}개"
    return f"{line} · {' · '.join(reasons)}" if reasons else line


def build_headline(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
) -> Headline:
    frame = _brief_frame(scored, brief, model)
    rows = list(iter_brief_rows(frame, brief))
    section_rank = {"buy": 0, "priced": 1, "steady": 2, "cheap": 3, "other": 4}
    rows = sorted(rows, key=lambda row: (section_rank.get(_section_of(row), 9),))
    buy_rows = [row for row in rows if _text(row, "proposal_status") == "오늘 신규매수 검토"]
    wait_rows = [row for row in rows if _text(row, "proposal_status") != "오늘 신규매수 검토"]

    def item(row: pd.Series, index: int) -> HeadlineItem:
        timing, reason = _timing(row, model)
        return HeadlineItem(
            rank=index,
            name=_text(row, "name"),
            symbol=_text(row, "symbol").zfill(6) if _text(row, "symbol") else "",
            timing=timing,
            reason=reason,
            stability=_stability_line(row),
            entry_rule="다음 거래일 시가 +1% 초과면 추격하지 않고 대기",
        )

    items = tuple(item(row, index) for index, row in enumerate(buy_rows[:3], 1))
    wait_items = tuple(
        item(row, index) for index, row in enumerate(wait_rows[:2], 1)
    )
    verdict = (
        f"모델상 오늘 살 후보 있음 · {len(buy_rows)}개"
        if buy_rows
        else "모델상 오늘 살 후보 없음 · 현금 유지"
    )
    return Headline(
        verdict=verdict,
        items=items,
        wait_items=wait_items,
        funnel=_large_flow_funnel(frame, rows),
    )


def render_headline_plain(headline: Headline) -> list[str]:
    lines = ["오늘 결론", headline.verdict]
    if headline.items:
        lines.extend(["", "살 후보 · 최대 3개"])
        for item in headline.items:
            lines.extend(
                [
                    f"{item.rank}. {item.name}  {item.symbol}",
                    f"   이유: {item.reason}",
                    f"   안정성 체크: {item.stability}",
                    f"   진입 조건: {item.entry_rule}",
                ]
            )
    elif headline.wait_items:
        lines.extend(["", "다음 후보 · 아직 사지 않음"])
        for item in headline.wait_items:
            lines.append(f"{item.rank}. {item.name}  {item.symbol} · {item.reason}")
    return lines


def render_headline_html(headline: Headline) -> str:
    items = "".join(
        "<li>"
        f"<b>{html.escape(item.name)}</b> {html.escape(item.symbol)}"
        f"<span>{html.escape(item.reason)}</span>"
        f"<small>{html.escape(item.stability)}</small>"
        f"<small>{html.escape(item.entry_rule)}</small></li>"
        for item in headline.items
    )
    wait_items = "".join(
        "<li>"
        f"<b>{html.escape(item.name)}</b> {html.escape(item.symbol)} "
        f"<span>{html.escape(item.reason)}</span></li>"
        for item in headline.wait_items
    )
    if items:
        list_html = f"<h3>살 후보 · 최대 3개</h3><ol>{items}</ol>"
    elif wait_items:
        list_html = f"<h3>다음 후보 · 아직 사지 않음</h3><ol>{wait_items}</ol>"
    else:
        list_html = ""
    return (
        '<section class="headline">'
        '<div class="headline-kicker">오늘 결론</div>'
        "<h2>살 만한 종목이 있나?</h2>"
        f'<p class="headline-verdict">{html.escape(headline.verdict)}</p>'
        f"{list_html}</section>"
    )


def render_headline_markdown(headline: Headline) -> str:
    lines = [
        "### 오늘 결론 · 살 만한 종목이 있나?",
        "",
        f"**{headline.verdict}**",
    ]
    if headline.items:
        lines.extend(["", "#### 살 후보 · 최대 3개", ""])
        for item in headline.items:
            lines.append(f"{item.rank}. **{item.name}** ({item.symbol})")
            lines.append(f"   - 이유: {item.reason}")
            lines.append(f"   - 안정성 체크: {item.stability}")
            lines.append(f"   - 진입 조건: {item.entry_rule}")
    elif headline.wait_items:
        lines.extend(["", "#### 다음 후보 · 아직 사지 않음", ""])
        for item in headline.wait_items:
            lines.append(f"{item.rank}. **{item.name}** ({item.symbol}) · {item.reason}")
    return "\n".join(lines)


def _stability_line(row: pd.Series | dict) -> str:
    """Compact evidence line; describes filters, never promises a stable rise."""

    buy_days = int(
        _float(
            row,
            "foreign_buy_days10",
            _float(row, "foreign_roll20_up_days10"),
        )
    )
    if _bool(row, "ma20_rising"):
        trend = "20일선 우상향"
    elif _bool(row, "ma20_falling"):
        trend = "20일선 우하향 주의"
    else:
        trend = "20일선 횡보"
    distribution = _float(row, "distribution_score")
    return f"외인 순매수 {buy_days}/10일 · {trend} · 분산위험 {distribution:.0f}/100"


def _judgment_line(row: pd.Series | dict, model: ModelConfig | None = None) -> str:
    model = model or ModelConfig()
    if _actionable(_text(row, "watch_tier")):
        return "게이트 통과 · 분할 신규 검토"
    price3 = _float(row, "price_return3")
    floor3 = model.entry_min_return3_pct / 100.0
    if price3 < floor3:
        return f"3일 {format_pct(price3)} · {model.entry_min_return3_pct:+.0f}% 이상으로 진정 필요"
    if _bool(row, "ma20_falling") and _text(row, "flow_price_divergence") != "bullish":
        return "20일선 우하향 · 우상향 회복 필요"
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


def _is_steady_row(row: pd.Series | dict) -> bool:
    bucket = _text(row, "screen_bucket")
    return _bool(row, "steady_accumulation") or bucket.startswith("차곡차곡")


def _is_priced_steady_row(row: pd.Series | dict) -> bool:
    return _bool(row, "steady_price_ok") or _text(row, "screen_bucket") == "차곡차곡가격"


def _timing(row: pd.Series | dict, model: ModelConfig | None = None) -> tuple[str, str]:
    """Human timing label and the single reason not to buy yet."""
    model = model or ModelConfig()
    tier = _text(row, "watch_tier")
    proposal = _text(row, "proposal_status")
    if proposal == "오늘 신규매수 검토":
        return "오늘 신규매수 검토", f"발견 {_text(row, 'discovery_grade')}등급 · 가격 게이트 통과"
    if proposal == "강한 후보·가격 대기":
        return "강한 후보 · 가격 대기", _judgment_line(row, model)
    if "discovery_eligible" in row and not _bool(row, "discovery_eligible"):
        return "데이터 갱신 대기", _text(row, "eligibility_reason", "당일 데이터 미수신")
    if _actionable(tier) or _bool(row, "entry_gate_passed"):
        return "관찰", "가격 게이트는 통과했지만 발견 75점 미만"
    if _is_priced_steady_row(row):
        return "가격 괜찮음 · 조건 대기", _judgment_line(row, model)
    if _is_steady_row(row):
        if _float(row, "price_return1") < 0:
            return "가격 대기 · 하락에도 외인 순매수", _judgment_line(row, model)
        return "가격 대기 · 차곡차곡", _judgment_line(row, model)
    if _is_cheap_row(row):
        return "주목 · 아직 대기", _judgment_line(row, model)
    if _bool(row, "dip_absorption") or _text(row, "screen_bucket").startswith("하락흡수"):
        return "참고 · 사지 않음", _judgment_line(row, model)
    if tier == "감시":
        return "감시만", "20일선에서 멀리 오른 하락 매수"
    if tier == "급락진정":
        return "진정 대기", _judgment_line(row, model)
    return "관찰", _judgment_line(row, model)


def _section_of(row: pd.Series | dict) -> str:
    if _text(row, "proposal_status") == "오늘 신규매수 검토":
        return "buy"
    timing, _ = _timing(row)
    if timing.startswith(("지금", "오늘")):
        return "buy"
    if _is_priced_steady_row(row):
        return "priced"
    if _is_steady_row(row):
        return "steady"
    if _is_cheap_row(row):
        return "cheap"
    return "other"


def _title(frame: pd.DataFrame) -> str:
    as_of = _as_of_label(frame)
    return f"외인 매집 브리핑{f'  ·  {as_of}' if as_of else ''}"


def _telegram_headline(headline: Headline) -> list[str]:
    lines = [
        "<b>오늘 결론 · 살 만한 종목이 있나?</b>",
        f"<b>{html.escape(headline.verdict)}</b>",
    ]
    if headline.items:
        compact = "\n\n".join(
            f"{item.rank}. <b>{html.escape(item.name)}</b>  "
            f"<code>{html.escape(item.symbol)}</code>\n"
            f"이유  {html.escape(item.reason)}\n"
            f"안정성  {html.escape(item.stability)}\n"
            f"진입  {html.escape(item.entry_rule)}"
            for item in headline.items
        )
        lines.append(compact)
    elif headline.wait_items:
        compact = "\n".join(
            f"{item.rank}. <b>{html.escape(item.name)}</b>  "
            f"<code>{html.escape(item.symbol)}</code> · {html.escape(item.reason)}"
            for item in headline.wait_items
        )
        lines.extend(["<b>다음 후보 · 아직 사지 않음</b>", compact])
    return lines


def build_telegram_text(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
    horizon: HorizonBoard | None = None,
) -> str:
    frame = _brief_frame(scored, brief, model)
    headline = build_headline(frame, brief, model)
    lines = [f"<b>{html.escape(_title(frame))}</b>", *_telegram_headline(headline)]
    lines.append("<i>모델 선별 후보이며 상승을 보장하지 않습니다 · 주문 미실행</i>")
    if scored.attrs.get("jev_monitor_note"):
        lines.append(html.escape(scored.attrs["jev_monitor_note"]))
    return "\n\n".join(line for line in lines if line)


def build_email_subject(scored: pd.DataFrame) -> str:
    as_of = _as_of_label(scored if not scored.empty else pd.DataFrame())
    headline = build_headline(scored) if not scored.empty else None
    result = f"살 후보 {len(headline.items)}개" if headline and headline.items else "오늘 살 후보 없음"
    return f"외인 매집 · {result}{f' · {as_of}' if as_of else ''}"


def build_email_bodies(
    scored: pd.DataFrame,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
    horizon: HorizonBoard | None = None,
) -> tuple[str, str]:
    frame = _brief_frame(scored, brief, model)
    title = _title(frame)
    headline = build_headline(frame, brief, model)
    plain_lines = [title, *render_headline_plain(headline)]
    plain_lines.extend(["", "모델 선별 후보이며 상승을 보장하지 않습니다 · 주문 미실행"])
    html_body = (
        "<div style='font-family:Segoe UI,Apple SD Gothic Neo,sans-serif;color:#111827;max-width:680px'>"
        f"<h2 style='margin:0 0 6px;font-size:22px'>{html.escape(title)}</h2>"
        f"{render_headline_html(headline)}"
        + "<p style='color:#6b7280;font-size:13px;margin:8px 0 0'>조회 전용 · 주문 미실행. "
        "모델 선별 후보이며 상승을 보장하지 않습니다. 상세 HTML 리포트를 첨부합니다.</p></div>"
    )
    if scored.attrs.get("jev_monitor_note"):
        note = scored.attrs["jev_monitor_note"]
        plain_lines.extend(["", note])
        html_body += "<p>" + html.escape(note).replace("\n", "<br>") + "</p>"
    return "\n".join(plain_lines), html_body


def send_telegram(
    scored: pd.DataFrame,
    report_path: Path,
    brief: BriefConfig | None = None,
    model: ModelConfig | None = None,
    horizon: HorizonBoard | None = None,
) -> bool:
    config = _telegram_config()
    if not config:
        LOGGER.info("Telegram secrets are not configured; notification skipped")
        return False
    token, chat_id = config
    api = f"https://api.telegram.org/bot{token}"
    response = requests.post(
        f"{api}/sendMessage",
        data={"chat_id": chat_id, "text": build_telegram_text(scored, brief, model, horizon), "parse_mode": "HTML"},
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
    horizon: HorizonBoard | None = None,
) -> bool:
    config = _email_config()
    if not config:
        LOGGER.info("Email SMTP secrets are not configured; notification skipped")
        return False
    plain, html_body = build_email_bodies(scored, brief, model, horizon)
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
    horizon: HorizonBoard | None = None,
) -> list[str]:
    """Send Telegram and email independently. Failures do not invalidate the report."""
    sent: list[str] = []
    try:
        if send_telegram(scored, report_path, brief, model, horizon):
            sent.append("telegram")
    except Exception as exc:
        LOGGER.error("Telegram delivery failed without invalidating the report: %s", exc)
    try:
        if send_email(scored, report_path, brief, model, horizon):
            sent.append("email")
    except Exception as exc:
        LOGGER.error("Email delivery failed without invalidating the report: %s", exc)
    if not sent:
        LOGGER.info("No notification channel delivered a report")
    return sent

