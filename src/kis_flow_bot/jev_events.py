"""Jev reads DART filings; a keyword rule reads the same titles; both are tested.

Why this exists: the numeric Jev questions were 86-97% reproducible by code
from their own inputs and added no value. Filing text is information the rule
model never sees, and deciding *why* foreigners are buying (dilution supply,
an adverse event, a real catalyst) is a judgment on text.

Rules kept from the numeric replay:
- Jev never sees a company name, stock code, price or return. Filing titles
  are official report names; any occurrence of the company name is removed.
  Dates become "within 7 days" / "8-30 days" buckets, which also lets
  consecutive days with the same filings reuse one cached answer.
- Only filings received on or before the signal date are included.
- Jev must beat a plain keyword rule on the same titles, not just the
  baseline. If keywords do as well, the verdict says so and code should use
  the keywords instead of paying for Jev.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .dart import DartClient, update_filings, window_needs
from .jev_shadow import LIMIT, MODEL, _number, request_review, select_candidates
from .research import build_research_panel, summarize_series
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)

EVENT_VERSION = "jev-events-v1"
LOOKBACK_DAYS = 30
RECENT_DAYS = 7
MAX_FILINGS = 25
FLAG_THRESHOLD = 0.5
SEVERE_LOSS = -0.15
TOP_N = 3
MIN_DATES = 40

QUESTIONS = {
    "dilution": {
        "type": "noul",
        "instructions": (
            "Using only these regulatory filing titles, do they announce or indicate new share supply: "
            "a new share issue for cash, convertible bonds, bonds with warrants, exchangeable bonds, "
            "or a major holder disposing of shares?"
        ),
        "criteria": {
            "true": "At least one filing clearly indicates new or additional shares that could reach the market.",
            "false": "No such filing, or only routine, bonus-issue, buyback or unrelated filings. Do not infer future returns.",
        },
    },
    "adverse": {
        "type": "noul",
        "instructions": (
            "Using only these regulatory filing titles, do they indicate a serious adverse event: litigation, "
            "embezzlement or breach of trust, an audit opinion problem, trading suspension, delisting or "
            "listing-eligibility review, default, rehabilitation, or an unfaithful-disclosure designation?"
        ),
        "criteria": {
            "true": "At least one filing clearly indicates such an event.",
            "false": "Only routine filings. Corrections alone are not adverse. Do not infer future returns.",
        },
    },
    "catalyst": {
        "type": "noul",
        "instructions": (
            "Using only these regulatory filing titles, do they indicate a positive fundamental catalyst: "
            "a large supply or sales contract, a share buyback or cancellation, or a dividend increase?"
        ),
        "criteria": {
            "true": "At least one filing clearly indicates such a catalyst.",
            "false": "Only routine or negative filings. A routine earnings release is not a catalyst by itself.",
        },
    },
}
PROMPT_HASH = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()[:16]
EVENTS = tuple(QUESTIONS)

# The baseline Jev has to beat. Same titles, no model.
KEYWORDS = {
    "dilution": re.compile(r"유상증자|전환사채|신주인수권|교환사채|자기주식처분"),
    "adverse": re.compile(r"소송|횡령|배임|감사의견|거래정지|매매거래정지|상장적격성|상장폐지|불성실공시|회생|부도|파산|관리종목|해산"),
    "catalyst": re.compile(r"공급계약|판매ㆍ공급|자기주식취득|자기주식소각|주식소각|배당"),
}
NEGATION = re.compile(r"무상증자|철회")

LEDGER_COLUMNS = [
    "state_hash",
    "prompt_hash",
    "model",
    "status",
    "error",
    *(f"event_{name}" for name in EVENTS),
    "input_tokens",
    "captured_at",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _clean_title(title: str, corp_name: str) -> str:
    text = str(title or "").strip()
    if corp_name:
        text = text.replace(corp_name, "회사")
    return re.sub(r"\s+", " ", text)


def filing_window(filings: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> pd.DataFrame:
    if filings.empty:
        return filings
    received = pd.to_datetime(filings["rcept_dt"].astype(str), format="%Y%m%d", errors="coerce")
    start = pd.Timestamp(as_of) - pd.Timedelta(days=LOOKBACK_DAYS)
    mask = (filings["symbol"] == symbol) & (received <= pd.Timestamp(as_of)) & (received > start)
    window = filings.loc[mask].copy()
    window["_received"] = received[mask]
    return window.sort_values("_received", ascending=False)


def build_state(window: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any] | None:
    if window.empty:
        return None
    items = []
    for _, row in window.head(MAX_FILINGS).iterrows():
        age = (pd.Timestamp(as_of) - row["_received"]).days
        corp_name = str(row.get("corp_name", ""))
        filer = str(row.get("flr_nm", ""))
        remark = str(row.get("rm", ""))
        items.append(
            {
                "when": f"within {RECENT_DAYS} days" if age <= RECENT_DAYS else f"{RECENT_DAYS + 1}-{LOOKBACK_DAYS} days before",
                "title": _clean_title(row.get("report_nm", ""), corp_name),
                "filer": "the company itself" if filer and filer == corp_name else "another party",
                "correction": "정" in remark,
            }
        )
    # Order-independent within a bucket so equal filing sets share one answer.
    items.sort(key=lambda item: (item["when"] != f"within {RECENT_DAYS} days", item["title"], item["filer"]))
    return {
        "scope": (
            f"Korean regulatory filing titles about one listed company, received during the {LOOKBACK_DAYS} days "
            "up to the decision date. Company names are replaced by '회사'. No prices, returns or later events."
        ),
        "filing_count": len(window),
        "filings": items,
    }


def state_hash(state: dict[str, Any] | None) -> str:
    if state is None:
        return ""
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def keyword_flags(window: pd.DataFrame) -> dict[str, bool]:
    titles = [
        _clean_title(title, name)
        for title, name in zip(window.get("report_nm", []), window.get("corp_name", []))
    ]
    flags = {}
    for name, pattern in KEYWORDS.items():
        flags[name] = any(pattern.search(title) and not NEGATION.search(title) for title in titles)
    return flags


def _parse_events(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        raise ValueError("Unexpected Jev model")
    result: dict[str, Any] = {}
    for key in EVENTS:
        answer = payload["answers"][key]
        value = _number(answer.get("noul"))
        if answer.get("type") != "noul" or value is None or not 0 <= value <= 1:
            raise ValueError("Invalid Jev response")
        result[f"event_{key}"] = value
    result["input_tokens"] = (payload.get("usage") or {}).get("input_tokens")
    return result


# ---------------------------------------------------------------------------
# Rows and calls
# ---------------------------------------------------------------------------


def build_rows(candidates: pd.DataFrame, filings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Per (as_of, symbol): filing count, keyword flags and the Jev state hash."""

    rows = []
    states: dict[str, dict] = {}
    for _, row in candidates.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        as_of = pd.Timestamp(row["as_of"])
        window = filing_window(filings, symbol, as_of)
        state = build_state(window, as_of)
        digest = state_hash(state)
        if state is not None:
            states[digest] = state
        flags = keyword_flags(window) if not window.empty else {name: False for name in EVENTS}
        rows.append(
            {
                "as_of": as_of,
                "symbol": symbol,
                "filing_count": int(len(window)),
                "state_hash": digest,
                **{f"kw_{name}": flags[name] for name in EVENTS},
            }
        )
    return pd.DataFrame(rows), states


def call_jev(
    states: dict[str, dict],
    ledger: pd.DataFrame,
    *,
    api_key: str,
    max_calls: int,
    session: Any | None = None,
    pause: Callable[[float], None] = time.sleep,
    spacing_seconds: float = 0.2,
) -> pd.DataFrame:
    """Ask Jev once per distinct filing state (cached across runs)."""

    existing = ledger.copy() if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS)
    for column in LEDGER_COLUMNS:
        if column not in existing:
            existing[column] = pd.NA
    done = set(
        existing.loc[
            existing["status"].astype(str).eq("OK") & existing["prompt_hash"].astype(str).eq(PROMPT_HASH),
            "state_hash",
        ].astype(str)
    )
    todo = [digest for digest in states if digest and digest not in done]
    if not todo or max_calls <= 0 or not api_key:
        return existing[LEDGER_COLUMNS]
    http = session or requests.Session()
    records = []
    try:
        for count, digest in enumerate(todo[:max_calls], 1):
            status, values, error, auth_failed = request_review(
                http, api_key, states[digest], pause, questions=QUESTIONS, parser=_parse_events
            )
            records.append(
                {
                    "state_hash": digest,
                    "prompt_hash": PROMPT_HASH,
                    "model": MODEL,
                    "status": status,
                    "error": error,
                    **values,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if auth_failed:
                LOGGER.error("Jev event review stopped: authentication/validation failure (%s)", error)
                break
            if count % 100 == 0:
                LOGGER.info("Jev event review progress: %d/%d", count, len(todo))
            if spacing_seconds > 0:
                pause(spacing_seconds)
    finally:
        if session is None:
            http.close()
    if not records:
        return existing[LEDGER_COLUMNS]
    incoming = pd.DataFrame(records)
    for column in LEDGER_COLUMNS:
        if column not in incoming:
            incoming[column] = pd.NA
    parts = [frame[LEDGER_COLUMNS] for frame in (existing, incoming) if not frame.empty]
    combined = pd.concat(parts, ignore_index=True)
    combined["_ok"] = combined["status"].astype(str).eq("OK")
    combined = combined.sort_values(["state_hash", "prompt_hash", "_ok", "captured_at"])
    combined = combined.drop_duplicates(["state_hash", "prompt_hash"], keep="last")
    return combined.drop(columns="_ok").reset_index(drop=True)


def attach_answers(rows: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """No filings → no event (0.0). Filings without an answer stay NaN."""

    result = rows.copy()
    answers = ledger[
        ledger["status"].astype(str).eq("OK") & ledger["prompt_hash"].astype(str).eq(PROMPT_HASH)
    ] if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS)
    answers = answers[["state_hash", *(f"event_{name}" for name in EVENTS)]].drop_duplicates("state_hash")
    result = result.merge(answers, on="state_hash", how="left")
    empty = result["filing_count"].eq(0)
    for name in EVENTS:
        column = f"event_{name}"
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[empty, column] = 0.0
        flag = (result[column] >= FLAG_THRESHOLD).astype("boolean")
        flag[result[column].isna()] = pd.NA
        result[f"jev_{name}"] = flag
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _flag_table(frame: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    rows = []
    for name in EVENTS:
        for source in ("jev", "kw"):
            flag = frame[f"{source}_{name}"].astype("boolean")
            valid = flag.notna() & frame[target].notna()
            flagged = frame.loc[valid & flag.fillna(False)]
            clean = frame.loc[valid & ~flag.fillna(False)]
            rows.append(
                {
                    "event": name,
                    "source": source,
                    "flag_rate": float(flag[valid].mean()) if valid.any() else float("nan"),
                    "flagged": int(len(flagged)),
                    "excess_flagged": float(flagged[target].mean()) if len(flagged) else float("nan"),
                    "excess_clean": float(clean[target].mean()) if len(clean) else float("nan"),
                    "severe_flagged": float((flagged["worst20"] <= SEVERE_LOSS).mean()) if len(flagged) else float("nan"),
                    "severe_clean": float((clean["worst20"] <= SEVERE_LOSS).mean()) if len(clean) else float("nan"),
                }
            )
        jev = frame[f"jev_{name}"].astype("boolean")
        both = jev.notna()
        rows[-1]["agreement"] = float((jev[both] == frame.loc[both, f"kw_{name}"].astype(bool)).mean()) if both.any() else float("nan")
    return rows


def veto_portfolios(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Top 3 by discovery order, skipping names flagged for dilution or adverse events."""

    out = []
    for as_of, group in frame.groupby("as_of", sort=True):
        group = group.dropna(subset=[target]).sort_values("baseline_rank")
        if len(group) < TOP_N * 2:
            continue
        jev_risk = group["jev_dilution"].astype("boolean") | group["jev_adverse"].astype("boolean")
        if jev_risk.isna().any():
            continue  # incomplete Jev coverage: do not score this day
        kw_risk = group["kw_dilution"].astype(bool) | group["kw_adverse"].astype(bool)
        picks = {
            "baseline": group.head(TOP_N),
            "jev_veto": group[~jev_risk.astype(bool)].head(TOP_N),
            "keyword_veto": group[~kw_risk].head(TOP_N),
            # Pre-registered H2 (2026-09-28): adverse filings only.
            "adverse_veto": group[~group["kw_adverse"].astype(bool)].head(TOP_N),
        }
        record: dict[str, Any] = {"as_of": as_of}
        for label, chosen in picks.items():
            # Fewer than three survivors: empty slots are cash.
            record[label] = float(chosen[target].sum() / TOP_N)
            record[f"{label}_severe"] = float((chosen["worst20"] <= SEVERE_LOSS).sum() / TOP_N)
        out.append(record)
    return pd.DataFrame(out)


def evaluate_events(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    horizon = settings.research.horizon
    target = f"excess{horizon}"
    confidence = settings.research.confidence
    labelled = frame[frame[target].notna()] if not frame.empty else frame
    result: dict[str, Any] = {
        "event_version": EVENT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "model": MODEL,
        "rows": int(len(labelled)),
        "rows_with_filings": int(labelled["filing_count"].gt(0).sum()) if not labelled.empty else 0,
        "jev_answered": int(labelled["event_dilution"].notna().sum()) if not labelled.empty else 0,
    }
    if labelled.empty:
        result.update(verdict="INSUFFICIENT_DATA", reasons=["표본 없음"], flags=[], portfolios={}, dates=0)
        return result
    result["flags"] = _flag_table(labelled, target)
    series = veto_portfolios(labelled, target)
    result["dates"] = int(len(series))
    portfolios: dict[str, Any] = {}
    if not series.empty:
        for column in ("baseline", "jev_veto", "keyword_veto"):
            portfolios[column] = summarize_series(series[column], horizon, confidence=confidence).as_dict()
            portfolios[f"{column}_severe"] = float(series[f"{column}_severe"].mean())
        for a, b in (("jev_veto", "baseline"), ("keyword_veto", "baseline"), ("jev_veto", "keyword_veto")):
            portfolios[f"{a}_minus_{b}"] = summarize_series(series[a] - series[b], horizon, confidence=confidence).as_dict()
    result["portfolios"] = portfolios
    result["h2"] = _h2_by_window(series)

    reasons: list[str] = []
    filings_share = result["rows_with_filings"] / max(result["rows"], 1)
    if filings_share < 0.2:
        # Nearly every listed company files something within 30 days. Almost no
        # filings means the collection failed (e.g. missing DART key), not that
        # filings are useless.
        verdict = "INSUFFICIENT_DATA"
        reasons.append(f"공시가 있는 표본 {filings_share:.0%}: DART 수집 실패 여부를 확인하세요")
    elif result["dates"] < MIN_DATES:
        verdict = "INSUFFICIENT_DATA"
        reasons.append(f"완결 신호일 {result['dates']}/{MIN_DATES}")
    else:
        jev_vs_kw = portfolios["jev_veto_minus_keyword_veto"]
        jev_vs_base = portfolios["jev_veto_minus_baseline"]
        kw_vs_base = portfolios["keyword_veto_minus_baseline"]
        if jev_vs_kw["low"] > 0 and jev_vs_base["mean"] > 0:
            verdict = "ADDS_VALUE"
            reasons.append(f"Jev 거부 − 키워드 거부 {jev_vs_kw['mean']:+.2%} (하한 {jev_vs_kw['low']:+.2%})")
        elif kw_vs_base["low"] > 0:
            verdict = "KEYWORDS_ENOUGH"
            reasons.append(f"키워드 거부만으로 기존 대비 {kw_vs_base['mean']:+.2%} (하한 {kw_vs_base['low']:+.2%}), Jev 추가 우위 없음")
        else:
            verdict = "NO_VALUE"
            reasons.append(
                f"Jev 거부 − 키워드 거부 하한 {jev_vs_kw['low']:+.2%}, 키워드 거부 − 기존 하한 {kw_vs_base['low']:+.2%}"
            )
    result["verdict"] = verdict
    result["reasons"] = reasons
    return result


def _h2_by_window(series: pd.DataFrame) -> dict[str, Any]:
    """H2: dropping adverse-filing names lowers severe-loss share, judged outside the discovery window."""

    from .hypotheses import MIN_CONFIRMATION_DATES, window_of

    out: dict[str, Any] = {}
    if series.empty:
        return out
    windows = window_of(series["as_of"])
    for window in ("discovery", "confirmation"):
        part = series[windows == window]
        if part.empty:
            out[window] = {"dates": 0}
            continue
        severe_delta = part["adverse_veto_severe"] - part["baseline_severe"]
        summary = summarize_series(severe_delta, 20)
        out[window] = {
            "dates": int(len(part)),
            "baseline_severe": float(part["baseline_severe"].mean()),
            "veto_severe": float(part["adverse_veto_severe"].mean()),
            "severe_delta": summary.as_dict(),
            "excess_delta": summarize_series(part["adverse_veto"] - part["baseline"], 20).as_dict(),
        }
    confirmation = out.get("confirmation", {})
    if confirmation.get("dates", 0) < MIN_CONFIRMATION_DATES:
        out["verdict"] = "INSUFFICIENT_DATA"
    elif confirmation["severe_delta"]["high"] < 0:
        out["verdict"] = "CONFIRMED"
    else:
        out["verdict"] = "NOT_CONFIRMED"
    return out


VERDICT_LABELS = {
    "INSUFFICIENT_DATA": "표본 부족",
    "ADDS_VALUE": "Jev 추가 가치 확인",
    "KEYWORDS_ENOUGH": "키워드 규칙으로 충분",
    "NO_VALUE": "공시 거부 효과 없음",
}


def _fmt(value: Any, pct: bool = True) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number * 100:+.2f}%" if pct else f"{number:.1%}"


def render_events_markdown(result: dict[str, Any], generated_at: datetime, horizon: int) -> str:
    verdict = result.get("verdict", "INSUFFICIENT_DATA")
    lines = [
        "# Jev 공시 판단 재생 검증",
        "",
        f"- 갱신: {generated_at.isoformat(timespec='seconds')} · `{EVENT_VERSION}` · 모델 `{result.get('model')}` · 프롬프트 `{result.get('prompt_hash')}`",
        f"- 입력: 신호일까지 접수된 최근 {LOOKBACK_DAYS}일 DART 공시 제목(회사명 제거, 날짜는 7일 이내/그 이전으로만 표시)",
        f"- 표본: {result.get('rows', 0)}건 중 공시 있음 {result.get('rows_with_filings', 0)}건 · Jev 응답 {result.get('jev_answered', 0)}건 · 평가 신호일 {result.get('dates', 0)}일",
        f"- 성과: 다음 거래일 시가 → D+{horizon} 종가, 같은 시장 동일가중 평균 대비 · 급락 = 보유 중 진입가 대비 {SEVERE_LOSS:.0%} 이하",
        "",
        f"## 판정: **{VERDICT_LABELS.get(verdict, verdict)}**",
        "",
        *[f"- {reason}" for reason in result.get("reasons", [])],
        "- 기준: Jev 거부가 같은 제목의 키워드 거부보다 나아야 합니다(하한 > 0). 키워드로 충분하면 Jev 대신 코드 규칙을 씁니다.",
        "",
    ]
    if result.get("flags"):
        lines.extend(
            [
                "## 1. 판정별 이후 성과",
                "",
                "| 사건 | 판단 | 해당 비율 | 해당 건수 | 해당 초과수익 | 비해당 초과수익 | 해당 급락률 | 비해당 급락률 | Jev·키워드 일치 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["flags"]:
            lines.append(
                f"| {row['event']} | {'Jev' if row['source'] == 'jev' else '키워드'} | {_fmt(row['flag_rate'], False)} | {row['flagged']} | "
                f"{_fmt(row['excess_flagged'])} | {_fmt(row['excess_clean'])} | {_fmt(row['severe_flagged'], False)} | "
                f"{_fmt(row['severe_clean'], False)} | {_fmt(row.get('agreement'), False) if row['source'] == 'kw' else ''} |"
            )
        lines.append("")
    portfolios = result.get("portfolios") or {}
    if portfolios:
        labels = {
            "baseline": "기존 발견 Top 3",
            "jev_veto": "Jev 희석·악재 제외 Top 3",
            "keyword_veto": "키워드 희석·악재 제외 Top 3",
            "jev_veto_minus_baseline": "Jev 제외 − 기존",
            "keyword_veto_minus_baseline": "키워드 제외 − 기존",
            "jev_veto_minus_keyword_veto": "Jev 제외 − 키워드 제외",
        }
        lines.extend(
            [
                "## 2. 거부 규칙을 적용한 Top 3 (같은 날)",
                "",
                "| 방식 | 신호일 | 평균 초과수익 | 하한 | 상한 | 급락 비율 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key, label in labels.items():
            item = portfolios.get(key, {})
            severe = portfolios.get(f"{key}_severe")
            lines.append(
                f"| {label} | {item.get('dates', 0)} | {_fmt(item.get('mean'))} | {_fmt(item.get('low'))} | "
                f"{_fmt(item.get('high'))} | {_fmt(severe, False) if severe is not None else ''} |"
            )
        lines.append("")
    h2 = result.get("h2") or {}
    if h2:
        from .hypotheses import VERDICT_LABELS as H_LABELS

        lines.extend(
            [
                "## 3. 사전등록 가설 H2: 악재 공시 종목 제외",
                "",
                "기존 Top 3에서 최근 30일 악재 공시(키워드) 종목을 빼면 급락(−15%) 비율이 줄어드는가. 발견 구간(2026-04-06 ~ 08-26)은 증거가 아니며 검증 구간으로만 판정합니다.",
                "",
                "| 구간 | 신호일 | 기존 급락 비율 | 제외 후 급락 비율 | 급락 비율 변화 상한 | 초과수익 변화 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for window, label in (("discovery", "발견"), ("confirmation", "검증")):
            item = h2.get(window, {})
            if not item.get("dates"):
                lines.append(f"| {label} | 0 | - | - | - | - |")
                continue
            lines.append(
                f"| {label} | {item['dates']} | {_fmt(item['baseline_severe'], False)} | {_fmt(item['veto_severe'], False)} | "
                f"{_fmt(item['severe_delta']['high'])} | {_fmt(item['excess_delta']['mean'])} |"
            )
        lines.extend(["", f"H2 판정: **{H_LABELS.get(h2.get('verdict', ''), h2.get('verdict', ''))}**", ""])
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _clients(dart_key: str | None, jev_key: str | None) -> tuple[DartClient | None, str]:
    dart = dart_key if dart_key is not None else os.getenv("DART_API_KEY", "")
    jev = jev_key if jev_key is not None else os.getenv("TYPESAFE_API_KEY", "")
    return (DartClient(dart) if dart else None), jev


def run_event_replay(
    settings: Settings,
    *,
    max_calls: int = 2000,
    workers: int | None = None,
    panel: pd.DataFrame | None = None,
    dart_client: DartClient | None = None,
    dart_key: str | None = None,
    jev_key: str | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(ZoneInfo("Asia/Seoul"))
    horizon = settings.research.horizon
    if panel is None:
        panel = build_research_panel(settings, workers=workers)
    labelled = panel[pd.to_numeric(panel.get(f"excess{horizon}"), errors="coerce").notna()]
    parts = [select_candidates(group, LIMIT) for _, group in labelled.groupby("as_of", sort=True)]
    shortlist = pd.concat([part for part in parts if not part.empty], ignore_index=True) if parts else labelled.iloc[0:0]
    shortlist = shortlist.copy()
    shortlist["baseline_rank"] = shortlist.groupby("as_of").cumcount() + 1

    client, jev = _clients(dart_key, jev_key)
    client = dart_client or client
    if client is None:
        LOGGER.warning("DART_API_KEY is missing; using cached filings only")
    filings = update_filings(client, settings.data_dir, window_needs(shortlist, LOOKBACK_DAYS))
    rows, states = build_rows(shortlist, filings)
    ledger_path = settings.data_dir / "jev_event_answers.csv"
    ledger = read_csv(ledger_path)
    ledger = call_jev(states, ledger, api_key=jev, max_calls=max_calls, session=session)
    write_csv(ledger if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS), ledger_path)
    answered = attach_answers(rows, ledger)
    merged = shortlist.merge(answered, on=["as_of", "symbol"], how="left")
    result = evaluate_events(merged, settings)
    result["generated_at"] = generated_at.isoformat(timespec="seconds")
    result["distinct_states"] = len(states)
    result["dart_calls"] = client.calls if client is not None else 0
    (settings.reports_dir / "jev_events.md").write_text(
        render_events_markdown(result, generated_at, horizon), encoding="utf-8"
    )
    (settings.data_dir / "jev_event_verdict.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Jev event replay: verdict=%s dates=%s states=%d dart_calls=%s",
        result["verdict"],
        result.get("dates"),
        len(states),
        result["dart_calls"],
    )
    return result


# ---------------------------------------------------------------------------
# Daily shadow: record what the filings said before the outcome is known
# ---------------------------------------------------------------------------

LIVE_COLUMNS = [
    "as_of",
    "symbol",
    "filing_count",
    "state_hash",
    *(f"kw_{name}" for name in EVENTS),
    *(f"event_{name}" for name in EVENTS),
    "prompt_hash",
    "captured_at",
]


def apply_event_review(
    scored: pd.DataFrame,
    settings: Settings,
    *,
    dart_client: DartClient | None = None,
    jev_key: str | None = None,
    session: Any | None = None,
    max_calls: int = 60,
) -> pd.DataFrame:
    """Flag today's shortlist from DART titles. Never changes ranking or gates."""

    result = scored.copy()
    result["event_filings"] = pd.NA
    for name in EVENTS:
        result[f"event_{name}"] = np.nan
        result[f"kw_{name}"] = pd.NA
    if result.empty or "as_of" not in result:
        return result
    candidates = select_candidates(result, LIMIT)
    if candidates.empty:
        return result
    client, jev = _clients(None, jev_key)
    client = dart_client or client
    if client is None:
        LOGGER.info("DART_API_KEY missing; filing review skipped")
        return result
    try:
        filings = update_filings(client, settings.data_dir, window_needs(candidates, LOOKBACK_DAYS), max_calls=200)
        rows, states = build_rows(candidates, filings)
        ledger_path = settings.data_dir / "jev_event_answers.csv"
        ledger = call_jev(states, read_csv(ledger_path), api_key=jev, max_calls=max_calls, session=session)
        write_csv(ledger if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS), ledger_path)
        answered = attach_answers(rows, ledger)
    except Exception as exc:  # the filing check must never block the report
        LOGGER.warning("Filing review failed without blocking the report: %s", exc)
        return result
    lookup = answered.set_index("symbol")
    for index, row in candidates.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        if symbol not in lookup.index:
            continue
        item = lookup.loc[symbol]
        result.at[index, "event_filings"] = int(item["filing_count"])
        for name in EVENTS:
            result.at[index, f"event_{name}"] = item[f"event_{name}"]
            result.at[index, f"kw_{name}"] = bool(item[f"kw_{name}"])
    live_path = settings.data_dir / "jev_event_live.csv"
    live = read_csv(live_path)
    today = answered.assign(
        as_of=pd.to_datetime(answered["as_of"]).dt.strftime("%Y-%m-%d"),
        prompt_hash=PROMPT_HASH,
        captured_at=datetime.now(timezone.utc).isoformat(),
    )[LIVE_COLUMNS]
    if not live.empty:
        known = set(zip(live["as_of"].astype(str), live["symbol"].astype(str).str.zfill(6)))
        today = today[[key not in known for key in zip(today["as_of"], today["symbol"])]]
        today = pd.concat([live, today], ignore_index=True) if not today.empty else live
    write_csv(today, live_path)
    return result


def event_section(scored: pd.DataFrame, verdict: dict[str, Any] | None) -> tuple[str, str]:
    """Report block: flagged shortlist names and the replay verdict."""

    import html as html_lib

    if "event_filings" not in scored or scored["event_filings"].isna().all():
        return "", ""
    label = VERDICT_LABELS.get((verdict or {}).get("verdict", ""), "검증 전")
    names = {"dilution": "희석·물량", "adverse": "악재", "catalyst": "호재"}
    lines = []
    for _, row in scored[scored["event_filings"].notna()].iterrows():
        tags = []
        for name in EVENTS:
            score = row.get(f"event_{name}")
            keyword = row.get(f"kw_{name}")
            jev_flag = pd.notna(score) and float(score) >= FLAG_THRESHOLD
            if jev_flag or keyword is True:
                source = "Jev·키워드" if jev_flag and keyword is True else ("Jev" if jev_flag else "키워드")
                tags.append(f"{names[name]}({source})")
        if tags:
            lines.append(f"{row.get('name', row['symbol'])} ({row['symbol']}): " + ", ".join(tags))
    note = f"검증 판정: {label}. 판정 통과 전에는 참고용이며 순위·매수 조건에 반영하지 않습니다."
    body = lines or ["최근 30일 공시에서 표시할 사건 없음"]
    markdown = "\n\n## 공시 점검 (DART)\n- " + note + "\n" + "\n".join(f"- {line}" for line in body) + "\n"
    html_block = (
        "<section class='panel'><h2>공시 점검 (DART)</h2><p>"
        + html_lib.escape(note)
        + "</p><ul>"
        + "".join(f"<li>{html_lib.escape(line)}</li>" for line in body)
        + "</ul></section>"
    )
    return markdown, html_block
