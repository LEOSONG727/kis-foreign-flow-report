from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RenderedReport:
    html: str
    markdown: str


STAGE_CLASS = {
    "매집·눌림": "green",
    "초기매집": "blue",
    "매집휴식": "yellow",
    "가격상승": "purple",
    "수익실현": "orange",
    "분산·이탈": "red",
    "이벤트확인": "red",
    "중립": "gray",
}

CAMPAIGN_CLASS = {
    "지속매집": "green",
    "장기매집·재가속": "green",
    "장기매집·휴식": "yellow",
    "신규재진입": "blue",
    "초기매집": "blue",
    "상승진행": "purple",
    "손바뀜·재확인": "yellow",
    "기관주도·외인확인": "yellow",
    "이상대량·확인": "orange",
    "분산·이탈": "red",
    "이벤트·구조확인": "red",
}


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def pct(value, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def eok(value) -> str:
    try:
        if pd.isna(value):
            return "-"
        amount = float(value) / 100_000_000
        if abs(amount) < 0.05:
            return "0억"
        if abs(amount) < 1.0:
            return f"{amount:+,.1f}억"
        return f"{amount:+,.0f}억"
    except (TypeError, ValueError):
        return "-"


def number(value, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def won(value) -> str:
    try:
        if pd.isna(value) or float(value) <= 0:
            return "-"
        return f"{float(value):,.0f}원"
    except (TypeError, ValueError):
        return "-"


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {esc(kind)}">{esc(text)}</span>'


def sparkline(values, width: int = 124, height: int = 38) -> str:
    if isinstance(values, str):
        try:
            values = json.loads(values.replace("'", '"'))
        except (json.JSONDecodeError, TypeError):
            values = []
    try:
        data = [float(v) for v in values if np.isfinite(float(v))]
    except (TypeError, ValueError):
        data = []
    if len(data) < 2:
        return '<span class="muted">차트 없음</span>'
    lo, hi = min(data), max(data)
    spread = max(hi - lo, 1e-9)
    points = []
    for index, value in enumerate(data):
        x = index * width / (len(data) - 1)
        y = height - 3 - (value - lo) * (height - 6) / spread
        points.append(f"{x:.1f},{y:.1f}")
    color = "#28d17c" if data[-1] >= data[0] else "#ff6b6b"
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" aria-label="60일 가격 추이">'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def _change_badge(row: pd.Series) -> str:
    if bool(row.get("is_new", False)):
        return badge("NEW", "blue")
    delta = float(row.get("entry_score_delta", row.get("score_delta", 0.0)))
    if delta > 0.4:
        return badge(f"▲ {delta:.1f}", "green")
    if delta < -0.4:
        return badge(f"▼ {abs(delta):.1f}", "red")
    return badge("—", "gray")


def _table(frame: pd.DataFrame, start_rank: int = 1, empty_message: str = "해당 종목이 없습니다.") -> str:
    if frame.empty:
        return f'<div class="empty">{esc(empty_message)}</div>'
    rows = []
    for offset, (_, row) in enumerate(frame.iterrows()):
        campaign = str(row.get("campaign_regime", "중립"))
        large_badge = badge("500억+", "green") if bool(row.get("large_flow_signal", False)) else ""
        rows.append(
            "<tr>"
            f'<td class="rank">{start_rank + offset}</td>'
            f'<td><strong>{esc(row.get("name", ""))}</strong><small>{esc(row.get("symbol", ""))} · {esc(row.get("market", ""))}</small></td>'
            f'<td class="score">{number(row.get("entry_score"))}<small>종합 {number(row.get("score"))} · {esc(row.get("entry_readiness", "-"))}</small></td>'
            f'<td>{number(row.get("campaign_score"))}</td>'
            f'<td>{badge(campaign, CAMPAIGN_CLASS.get(campaign, "gray"))}<small>가격 {esc(row.get("stage", "중립"))}</small></td>'
            f'<td>{eok(row.get("foreign_net20"))} {large_badge}<small>{number(row.get("foreign_20_adtv"))}일 ADTV</small></td>'
            f'<td>{pct(row.get("price_return20"))}'
            f'<small>3일 {pct(row.get("price_return3"))} · 10일 {pct(row.get("price_return10"))}</small>'
            f'<small>20일 저점반등 {pct(row.get("rebound_from_low20"))}</small></td>'
            f'<td class="action">{esc(row.get("action", ""))}'
            f'<small>{won(row.get("suggested_order_krw")) if row.get("suggested_order_krw", 0) else ""}</small></td>'
            f'<td class="condition">{esc(row.get("promotion_condition", "-"))}'
            f'<small>무효: {esc(row.get("invalidation_condition", "-"))}</small></td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        '<th>#</th><th>종목</th><th>진입점수</th><th>매집강도</th><th>매집 국면</th>'
        '<th>20일 외인</th><th>가격·단기열기</th><th>액션</th><th>다음 조건</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _reason_list(row: pd.Series) -> str:
    reasons = []
    if row.get("foreign_20_adtv", 0.0) > 0:
        reasons.append(f'20일 외국인 순매수 {eok(row.get("foreign_net20"))} ({number(row.get("foreign_20_adtv"))}일치 거래대금)')
    if bool(row.get("large_flow_signal", False)):
        reasons.append("20거래일 누적 외국인 순매수 500억원 이상")
    reasons.append(f'매집 국면 {row.get("campaign_regime", "중립")}·매집 강도 {number(row.get("campaign_score"))}점')
    reasons.append(
        f'매수일 {int(row.get("buy_days20", 0))}일·매도일 {int(row.get("sell_days20", 0))}일, 되팔기 비율 {number(row.get("sellback_ratio20") * 100, 0)}%'
    )
    reasons.append(
        f'가격 1·3·5·10·20일 {pct(row.get("price_return1"))}·{pct(row.get("price_return3"))}·'
        f'{pct(row.get("price_return5"))}·{pct(row.get("price_return10"))}·{pct(row.get("price_return20"))}'
    )
    reasons.append(
        f'20일 저점 대비 {pct(row.get("rebound_from_low20"))}·20일선 대비 {pct(row.get("ma20_distance"))}·'
        f'20일 고점 대비 {pct(row.get("distance_high20"))}'
    )
    if bool(row.get("price_heat_flag", False)):
        reasons.append(
            f'단기 가격과열: {row.get("price_heat_reason", "-")} (진입 벌점 {number(row.get("price_heat_penalty"))}점)'
        )
    if row.get("absorption_signal") in {"강함", "보통", "주의"}:
        reasons.append(f'가격 하락 중 외국인 위험흡수 신호: {row.get("absorption_signal")}')
    if row.get("opinion_coverage", 0):
        reasons.append(
            f'120일 내 목표가 {int(row.get("opinion_coverage", 0))}건 중앙값 {won(row.get("target_price"))}, '
            f'현 주가 대비 {pct(row.get("target_upside"))}·최신 {number(row.get("opinion_recency_days"), 0)}일 전'
        )
    else:
        reasons.append("증권사 목표가 데이터 없음: 전망 점수는 중립 처리")
    if row.get("promotion_condition"):
        reasons.append(f'다음 액션 조건: {row.get("promotion_condition")}')
    if row.get("invalidation_condition"):
        reasons.append(f'신호 무효 조건: {row.get("invalidation_condition")}')
    return "".join(f"<li>{esc(reason)}</li>" for reason in reasons)


def _hero_cards(frame: pd.DataFrame) -> str:
    cards = []
    for rank, (_, row) in enumerate(frame.head(3).iterrows(), 1):
        campaign = str(row.get("campaign_regime", "중립"))
        cards.append(
            '<article class="hero-card">'
            f'<div class="hero-head"><span class="hero-rank">#{rank}</span>{badge(campaign, CAMPAIGN_CLASS.get(campaign, "gray"))}</div>'
            f'<h3>{esc(row.get("name"))} <small>{esc(row.get("symbol"))}</small></h3>'
            f'<div class="hero-score"><strong>{number(row.get("entry_score"))}</strong><span>/100 진입</span></div>'
            f'<div class="chart">{sparkline(row.get("price_series", []), 210, 54)}</div>'
            f'<p class="primary-action">{esc(row.get("action"))}</p>'
            f'<ul>{_reason_list(row)}</ul>'
            '</article>'
        )
    return '<div class="hero-grid">' + "".join(cards) + "</div>"


def _detail_cards(frame: pd.DataFrame) -> str:
    cards = []
    for _, row in frame.iterrows():
        components = [
            ("매집 강도", row.get("campaign_score", 0), 100),
            ("가격 미반응", row.get("underreaction_score", 0), 25),
            ("가격·거래량", row.get("structure_score", 0), 15),
            ("실적·전망", row.get("fundamental_score", 0), 20),
        ]
        bars = "".join(
            '<div class="metric"><span>{}</span><div><i style="width:{:.0f}%"></i></div><b>{:.1f}/{}</b></div>'.format(
                esc(label), clamp_percent(float(value) / maximum * 100), float(value), maximum
            )
            for label, value, maximum in components
        )
        cards.append(
            '<article class="detail-card">'
            f'<div class="detail-title"><div><h3>{esc(row.get("name"))}</h3><small>{esc(row.get("symbol"))} · {esc(row.get("market"))}</small></div>'
            f'<div class="big-score">진입 {number(row.get("entry_score"))}</div></div>'
            f'<div class="detail-chart">{sparkline(row.get("price_series", []), 360, 76)}</div>'
            f'<div class="metrics">{bars}</div>'
            '<div class="facts">'
            f'<span>매집 국면 <b>{esc(row.get("campaign_regime", "중립"))}</b></span>'
            f'<span>외인 5일 <b>{eok(row.get("foreign_net5"))}</b></span>'
            f'<span>외인 20일 <b>{eok(row.get("foreign_net20"))}</b></span>'
            f'<span>당일 외인 <b>{eok(row.get("foreign_net1"))}</b></span>'
            f'<span>외인 60일 <b>{eok(row.get("foreign_net60"))}</b></span>'
            f'<span>가격 3일 <b>{pct(row.get("price_return3"))}</b></span>'
            f'<span>가격 10일 <b>{pct(row.get("price_return10"))}</b></span>'
            f'<span>저점반등 20일 <b>{pct(row.get("rebound_from_low20"))}</b></span>'
            f'<span>20일선 이격 <b>{number(row.get("ma20_atr_distance"))} ATR</b></span>'
            f'<span>반등 거래량 <b>{number(row.get("rebound_volume_ratio20"))}배</b></span>'
            f'<span>기관 보조 <b>{number(row.get("institution_overlay"))}</b></span>'
            f'<span>ROE <b>{number(row.get("roe"))}%</b></span>'
            f'<span>PER <b>{number(row.get("per"))}</b></span>'
            f'<span>PBR <b>{number(row.get("pbr"))}</b></span>'
            '</div>'
            f'<p class="primary-action">{esc(row.get("action"))}</p>'
            f'<ul>{_reason_list(row)}</ul>'
            '</article>'
        )
    return '<div class="detail-grid">' + "".join(cards) + "</div>"


def clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


CSS = """
:root{color-scheme:dark;--bg:#0a0d12;--panel:#111720;--line:#263140;--text:#f4f7fb;--muted:#8f9baa;--green:#28d17c;--blue:#55a7ff;--red:#ff6b6b;--yellow:#f4c95d;--purple:#ae7cff;--orange:#ff9f55}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Pretendard,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}.page{max-width:1240px;margin:auto;padding:28px 20px 72px}header{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:24px}h1{font-size:30px;margin:0 0 5px}h2{font-size:21px;margin:36px 0 14px}h3{margin:0}small,.muted{display:block;color:var(--muted);font-size:12px}.stamp{text-align:right;color:var(--muted)}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.summary>div,.hero-card,.detail-card,.panel{background:linear-gradient(145deg,#151c26,#0e141d);border:1px solid var(--line);border-radius:16px}.summary>div{padding:15px}.summary b{font-size:25px;display:block}.summary span{color:var(--muted);font-size:12px}.hero-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.hero-card{padding:18px}.hero-head{display:flex;justify-content:space-between}.hero-rank{font-weight:800;color:var(--blue)}.hero-card h3{font-size:22px;margin-top:13px}.hero-card h3 small{display:inline}.hero-score strong{font-size:42px}.hero-score span{color:var(--muted)}.chart,.detail-chart{margin:8px 0}.spark{width:100%;height:auto}.primary-action{font-weight:800;color:#fff;background:#1c2836;border-left:3px solid var(--green);padding:10px 12px;border-radius:8px}.hero-card ul,.detail-card ul{padding-left:19px;color:#c7d0da;font-size:13px}.table-wrap{border:1px solid var(--line);border-radius:14px;overflow:auto;background:var(--panel)}table{width:100%;border-collapse:collapse;min-width:1020px}th,td{padding:12px 10px;border-bottom:1px solid #202a37;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#151c26;color:var(--muted);font-size:12px}td{font-size:13px}td small{margin-top:3px}.rank{color:var(--muted)}.score{font-size:18px;font-weight:850}.action{font-weight:750}.badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:11px;font-weight:800;background:#263140}.badge.green{color:var(--green);background:#123326}.badge.blue{color:var(--blue);background:#142d49}.badge.red{color:var(--red);background:#3a1d22}.badge.yellow{color:var(--yellow);background:#332d18}.badge.purple{color:var(--purple);background:#2c2142}.badge.orange{color:var(--orange);background:#3b2819}.badge.gray{color:#aab4c0}.detail-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.detail-card{padding:18px}.detail-title{display:flex;justify-content:space-between}.big-score{font-size:30px;font-weight:900}.metric{display:grid;grid-template-columns:80px 1fr 62px;gap:8px;align-items:center;margin:8px 0;font-size:12px}.metric>div{height:7px;background:#25303e;border-radius:8px;overflow:hidden}.metric i{height:100%;display:block;background:linear-gradient(90deg,var(--blue),var(--green))}.metric b{text-align:right}.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:14px 0}.facts span{background:#0c121a;padding:8px;border-radius:8px;color:var(--muted);font-size:11px}.facts b{display:block;color:var(--text);font-size:13px}.panel{padding:16px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}.empty{padding:22px;color:var(--muted);background:var(--panel);border-radius:12px}.note{color:var(--muted);font-size:12px;margin-top:30px}.legend{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.legend div{background:#101721;border:1px solid var(--line);padding:11px;border-radius:10px;font-size:12px}.legend b{display:block;margin-bottom:3px}
table{min-width:1220px}.condition{max-width:330px;white-space:normal}.condition small{color:#e38a8a}
@media(max-width:800px){.page{padding:20px 12px 50px}header{display:block}.stamp{text-align:left;margin-top:8px}.summary{grid-template-columns:repeat(2,1fr)}.hero-grid,.detail-grid,.two-col{grid-template-columns:1fr}.hero-card h3{font-size:20px}.legend{grid-template-columns:1fr}.facts{grid-template-columns:repeat(2,1fr)}h1{font-size:25px}}
"""


def render_report(
    scored: pd.DataFrame,
    top_n: int,
    watchlist: tuple[str, ...],
    generated_at: datetime,
) -> RenderedReport:
    ranked = scored.copy().reset_index(drop=True)
    if "decision_bucket" not in ranked:
        ranked["decision_bucket"] = np.where(
            ranked["action"].isin(["강한 신규매수", "초기 분할매수"]),
            "신규매수",
            "관찰",
        )
    if "action_priority" not in ranked:
        ranked["action_priority"] = ranked.get("entry_score", pd.Series(0.0, index=ranked.index))
    queue_sort = ["action_priority", "entry_score", "campaign_score"]
    new_buys = ranked[ranked["decision_bucket"] == "신규매수"].sort_values(queue_sort, ascending=False).head(10)
    add_buys = ranked[ranked["decision_bucket"] == "추가매수"].sort_values(queue_sort, ascending=False).head(10)
    wait_candidates = ranked[ranked["decision_bucket"].isin(["눌림·확인대기", "관찰"])].sort_values(queue_sort, ascending=False).head(12)
    held_management = ranked[ranked["decision_bucket"] == "보유관리"].sort_values(queue_sort, ascending=False).head(10)
    risk_actions = ranked[ranked["decision_bucket"] == "수익실현·중단"].sort_values(queue_sort, ascending=False).head(12)
    actionable = pd.concat([new_buys, add_buys, wait_candidates]).drop_duplicates("symbol")
    top = actionable.head(top_n) if not actionable.empty else ranked.head(top_n)
    watched = ranked[ranked["symbol"].astype(str).str.zfill(6).isin(watchlist)]
    falling = ranked[
        (ranked["price_return20"] < 0)
        & (ranked["foreign_net20"] > 0)
        & (ranked["absorption_signal"].isin(["강함", "보통"]))
    ].head(10)
    large_flow = ranked[ranked["large_flow_signal"].fillna(False)].head(15)

    new_count = int((ranked["decision_bucket"] == "신규매수").sum())
    add_count = int((ranked["decision_bucket"] == "추가매수").sum())
    wait_count = int(ranked["decision_bucket"].isin(["눌림·확인대기", "관찰"]).sum())
    risk_count = int((ranked["decision_bucket"] == "수익실현·중단").sum())
    large_flow_count = int(ranked["large_flow_signal"].fillna(False).sum())
    report_date = pd.to_datetime(ranked["as_of"].max()).strftime("%Y-%m-%d") if not ranked.empty else generated_at.strftime("%Y-%m-%d")

    html_text = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>외국인 매집 리포트 {report_date}</title><style>{CSS}</style></head><body><main class="page">
<header><div><h1>외국인 매집 레이더</h1><div class="muted">많이 샀지만 아직 덜 오른 종목 · KIS 일별 확정 데이터</div></div><div class="stamp">기준일 {report_date}<br>생성 {generated_at.strftime('%Y-%m-%d %H:%M KST')}</div></header>
<section class="summary"><div><b>{new_count}</b><span>신규매수 후보</span></div><div><b>{add_count}</b><span>추가매수 실행 후보</span></div><div><b>{wait_count}</b><span>눌림·확인 대기</span></div><div><b>{risk_count}</b><span>수익실현·중단</span></div></section>
<h2>오늘의 우선 확인 3종목</h2>{_hero_cards(top)}
<h2>① 오늘 신규매수 후보</h2>{_table(new_buys, empty_message="오늘은 신규매수 기준을 모두 통과한 종목이 없습니다.")}
<h2>② 보유종목 추가매수 후보</h2>{_table(add_buys, empty_message="추가매수 실행 후보가 없습니다. portfolio.csv에 실제 보유정보가 있어야 개인화됩니다.")}
<h2>③ 좋은 종목·첫 눌림/확인 대기</h2>{_table(wait_candidates)}
<div class="two-col"><section><h2>④ 보유·추가매수 준비</h2>{_table(held_management, empty_message="등록된 보유종목 관리 항목이 없습니다.")}</section><section><h2>⑤ 수익실현·신호 중단</h2>{_table(risk_actions)}</section></div>
<h2>진입점수 전체 TOP {top_n}</h2>{_table(ranked.head(top_n))}
<div class="two-col"><section><h2>20일 외인 500억원 이상 ({large_flow_count})</h2>{_table(large_flow)}</section><section><h2>가격 하락 + 외국인 매수</h2>{_table(falling)}</section></div>
<h2>내 관찰 종목</h2>{_table(watched)}
<h2>상세 분석</h2>{_detail_cards(pd.concat([new_buys, add_buys, wait_candidates, watched]).drop_duplicates('symbol').head(12))}
<h2>국면 해석</h2><div class="legend"><div><b>지속매집</b>20일 순매수·매수일 비율·되팔기 비율이 함께 양호</div><div><b>장기매집·휴식</b>60·120일 재고는 남아 있으나 최근 5일은 잠시 매도</div><div><b>신규재진입</b>장기 순매도 뒤 최근 20·5일 외국인이 다시 들어오는 전환</div><div><b>상승진행</b>매집은 유효하지만 가격이 먼저 올라 신규 추격은 제한</div><div><b>기관주도·외인확인</b>기관이 가격을 이끌고 외국인 확인은 아직 부족</div><div><b>이상대량·확인</b>소수 대량 거래가 전체 수급을 왜곡할 수 있어 블록·이벤트 확인 필요</div></div>
<p class="note">진입점수는 후보 우선순위이며 매수 허가가 아니다. 실제 신규·추가매수는 최근 3일 급락 진정, 20일 상승률 +3% 미만, 20일 저점 반등 +5% 미만, 외국인 재가속을 별도 통과해야 한다. 20거래일 외국인 500억원 이상에는 대형 매집 신호를 부여한다. 확률은 양(+)의 워크포워드 기대값과 홀드아웃 검증을 통과하기 전에는 보정하지 않는다. 주문 API는 포함되어 있지 않다.</p>
</main></body></html>"""

    def md_section(title: str, frame: pd.DataFrame) -> str:
        lines = [f"### {title}", "", "| 순위 | 종목 | 진입 | 매집 | 국면 | 외인 20일 | 가격 20일 | 저점반등 20일 | 액션 |", "|---:|---|---:|---:|---|---:|---:|---:|---|"]
        if frame.empty:
            lines.append("| - | 해당 없음 | - | - | - | - | - | - | - |")
        else:
            for rank, (_, row) in enumerate(frame.head(10).iterrows(), 1):
                lines.append(f"| {rank} | {row['name']} ({row['symbol']}) | {row['entry_score']:.1f} | {row['campaign_score']:.1f} | {row['campaign_regime']} | {eok(row['foreign_net20'])} | {pct(row['price_return20'])} | {pct(row.get('rebound_from_low20'))} | {row['action']} |")
        return "\n".join(lines)

    markdown = "\n\n".join(
        [
            f"## 외국인 매집 액션 리포트 · {report_date}",
            md_section("① 신규매수", new_buys),
            md_section("② 추가매수", add_buys),
            md_section("③ 눌림·확인 대기", wait_candidates),
            md_section("④ 수익실현·중단", risk_actions),
        ]
    ) + "\n\n*진입점수는 백테스트 보정 전이며, 주문은 실행하지 않습니다.*\n"
    return RenderedReport(html=html_text, markdown=markdown)
