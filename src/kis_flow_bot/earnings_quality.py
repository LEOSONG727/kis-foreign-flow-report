"""Jev reads *why* profit changed, to protect the operating-yield signal.

Operating yield (H1) is the only signal that survived style controls. Its
known failure mode is a profit that will not repeat: an asset sale, a
valuation or FX gain, a reversed provision. Korean issuers must explain large
profit changes in the "매출액또는손익구조 30%(대규모법인 15%) 이상 변동" filing
and in preliminary earnings filings, in free text. Reading that text is a
judgment code cannot make with numbers, so it is a fair job for Jev.

Pre-registered as H3 on 2026-09-28, before any answer was seen:
    Among each day's H1 top 20, drop names whose latest profit explanation
    Jev judges one-off (score >= 0.5), then take the top 5. This must beat
    the same veto by a keyword rule on the same text (lower bound > 0) and
    beat plain H1 top 5 on average, over at least 40 signal dates.
H3 was not shaped by any result, so every labelled date counts as evidence.

Leakage guard: Jev sees only the filing text received on or before the
signal date, with the company name replaced and years masked.
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
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .dart import DartClient, DartError, update_filings, window_needs
from .hypotheses import HYPOTHESES, hypothesis_score
from .jev_shadow import MODEL, _number, request_review
from .research import build_research_panel, summarize_series
from .storage import read_csv, write_csv
from .point_in_time import PIT_VERSION, backward_asof, local_dates, observable_answers

LOGGER = logging.getLogger(__name__)

EQ_VERSION = "jev-earnings-quality-v1"
H1_ID = "H1-value-quality-lowvol"
EXPLAIN_TITLES = re.compile(r"손익구조|잠정\)?실적|\(잠정\)실적")
LOOKBACK_DAYS = 400
CANDIDATES_PER_DAY = 20
PICKS = 5
FLAG_THRESHOLD = 0.5
MAX_TEXT = 2500
MIN_DATES = 40
SEVERE_LOSS = -0.15

QUESTIONS = {
    "one_off": {
        "type": "noul",
        "instructions": (
            "Using only this Korean company filing text, is the reported profit change driven mainly by "
            "one-off or non-operating items, such as gains on asset or subsidiary sales, valuation or FX gains, "
            "reversals of provisions or impairments, insurance or subsidy income, or bargain-purchase gains, "
            "rather than by recurring operations?"
        ),
        "criteria": {
            "true": "The text attributes most of the profit change to such one-off or non-operating items.",
            "false": "The change is attributed to recurring operations, or the text gives no such attribution. Do not infer future returns.",
        },
    },
    "recurring": {
        "type": "noul",
        "instructions": (
            "Using only this Korean company filing text, is the profit change explained mainly by recurring "
            "operations, such as sales volume, pricing, product mix, cost structure or capacity utilisation?"
        ),
        "criteria": {
            "true": "Recurring operating drivers are the stated main reason.",
            "false": "Operating drivers are not the stated main reason, or no reason is given. Do not infer future returns.",
        },
    },
}
PROMPT_HASH = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()[:16]
ANSWERS = tuple(QUESTIONS)

KEYWORDS = {
    "one_off": re.compile(
        r"처분이익|평가이익|환입|일회성|보험금|국고보조금|채무면제|매각이익|손상차손환입|염가매수|외환차익|외화환산이익|파생상품"
    ),
    "recurring": re.compile(r"판매량|판매 ?증가|매출 ?증가|수주|단가|판가|가동률|원가 ?절감|수익성 ?개선|신제품|제품 ?믹스"),
}

LEDGER_COLUMNS = ["state_hash", "prompt_hash", "model", "status", "error", *(f"eq_{name}" for name in ANSWERS), "captured_at"]
DOC_COLUMNS = ["rcept_no", "text"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def latest_explanation(filings: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> pd.Series | None:
    """Most recent profit-explanation filing public on the signal date."""

    if filings.empty:
        return None
    received = local_dates(filings["rcept_dt"])
    mask = (
        (filings["symbol"] == symbol)
        & (received < pd.Timestamp(as_of).normalize())
        & (received > pd.Timestamp(as_of) - pd.Timedelta(days=LOOKBACK_DAYS))
        & filings["report_nm"].astype(str).str.contains(EXPLAIN_TITLES)
        & ~filings["report_nm"].astype(str).str.contains("철회")
    )
    rows = filings.loc[mask].assign(_received=received[mask]).sort_values(["_received", "rcept_no"])
    return None if rows.empty else rows.iloc[-1]


def redact(text: str, corp_name: str) -> str:
    cleaned = str(text or "")
    if corp_name:
        cleaned = cleaned.replace(corp_name, "회사")
    cleaned = re.sub(r"(19|20)\d{2}", "YYYY", cleaned)
    cleaned = re.sub(r"\b\d{6}\b", "######", cleaned)  # stock codes
    return cleaned[:MAX_TEXT]


def build_state(title: str, text: str, corp_name: str) -> dict[str, Any]:
    return {
        "scope": (
            "Text of one Korean regulatory filing that explains a company's profit or preliminary results. "
            "Company names are replaced by '회사' and years by 'YYYY'. No prices, returns or later events."
        ),
        "filing_title": redact(title, corp_name),
        "text": redact(text, corp_name),
    }


def state_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def keyword_flags(text: str) -> dict[str, bool]:
    return {name: bool(pattern.search(str(text or ""))) for name, pattern in KEYWORDS.items()}


def _parse(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        raise ValueError("Unexpected Jev model")
    result: dict[str, Any] = {}
    for key in ANSWERS:
        answer = payload["answers"][key]
        value = _number(answer.get("noul"))
        if answer.get("type") != "noul" or value is None or not 0 <= value <= 1:
            raise ValueError("Invalid Jev response")
        result[f"eq_{key}"] = value
    return result


def fetch_documents(client: DartClient | None, data_dir, rcept_nos: list[str], max_docs: int) -> pd.DataFrame:
    path = data_dir / "dart_documents.csv"
    docs = read_csv(path)
    if docs.empty:
        docs = pd.DataFrame(columns=DOC_COLUMNS)
    docs["rcept_no"] = docs["rcept_no"].astype(str).str.replace(r"\.0$", "", regex=True)
    have = set(docs["rcept_no"].astype(str))
    todo = [number for number in dict.fromkeys(rcept_nos) if number and number not in have]
    if client is None or not todo:
        return docs
    rows = []
    for number in todo[:max_docs]:
        try:
            rows.append({"rcept_no": number, "text": client.document_text(number)})
        except (DartError, requests.RequestException, ValueError) as exc:
            LOGGER.warning("DART document %s failed: %s", number, exc)
            if "020" in str(exc):
                break
    if rows:
        docs = pd.concat([docs, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("rcept_no", keep="last")
        write_csv(docs, path)
    return docs


def call_jev(
    states: dict[str, dict],
    ledger: pd.DataFrame,
    *,
    api_key: str,
    max_calls: int,
    session: Any | None = None,
    pause: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    existing = ledger.copy() if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS)
    for column in LEDGER_COLUMNS:
        if column not in existing:
            existing[column] = pd.NA
    done = set(
        existing.loc[
            existing["status"].astype(str).eq("OK") & existing["prompt_hash"].astype(str).eq(PROMPT_HASH)
            & existing["model"].astype(str).eq(MODEL),
            "state_hash",
        ].astype(str)
    )
    todo = [digest for digest in states if digest not in done]
    if not todo or not api_key or max_calls <= 0:
        return existing[LEDGER_COLUMNS]
    http = session or requests.Session()
    records = []
    try:
        for count, digest in enumerate(todo[:max_calls], 1):
            status, values, error, auth_failed = request_review(
                http, api_key, states[digest], pause, questions=QUESTIONS, parser=_parse
            )
            records.append(
                {"state_hash": digest, "prompt_hash": PROMPT_HASH, "model": MODEL, "status": status,
                 "error": error, **values, "captured_at": datetime.now(timezone.utc).isoformat()}
            )
            if auth_failed:
                break
            if count % 100 == 0:
                LOGGER.info("Jev earnings-quality progress: %d/%d", count, len(todo))
            pause(0.2)
    finally:
        if session is None:
            http.close()
    if not records:
        return existing[LEDGER_COLUMNS]
    incoming = pd.DataFrame(records)
    for column in LEDGER_COLUMNS:
        if column not in incoming:
            incoming[column] = pd.NA
    combined = pd.concat([frame[LEDGER_COLUMNS] for frame in (existing, incoming) if not frame.empty], ignore_index=True)
    combined["_ok"] = combined["status"].astype(str).eq("OK")
    combined = combined.sort_values(["state_hash", "_ok", "captured_at"], ascending=[True, False, True]).drop_duplicates(["state_hash", "prompt_hash", "model"], keep="first")
    return combined.drop(columns="_ok").reset_index(drop=True)


def review_rows(
    candidates: pd.DataFrame,
    settings: Settings,
    *,
    client: DartClient | None,
    jev_key: str,
    max_calls: int,
    max_docs: int,
    session: Any | None = None,
) -> pd.DataFrame:
    """For each (as_of, symbol): explanation text, keyword flags and Jev answers."""

    filings = update_filings(client, settings.data_dir, window_needs(candidates, LOOKBACK_DAYS))
    relevant = filings[filings["report_nm"].astype(str).str.contains(EXPLAIN_TITLES)
                       & ~filings["report_nm"].astype(str).str.contains("철회")].copy()
    relevant["available"] = local_dates(relevant["rcept_dt"])
    if "corp_name" not in relevant:
        relevant["corp_name"] = ""
    events = relevant[["symbol", "rcept_no", "report_nm", "corp_name", "available"]].rename(columns={"report_nm": "title"})
    events = events.sort_values(["available", "rcept_no"], kind="stable")
    rows = backward_asof(candidates[["as_of", "symbol"]], events)
    stale = (local_dates(rows["as_of"]) - rows["available"]).dt.days >= LOOKBACK_DAYS
    rows.loc[stale, ["rcept_no", "title", "corp_name"]] = pd.NA
    for column in ("rcept_no", "title", "corp_name"):
        rows[column] = rows[column].fillna("").astype(str)
    docs = fetch_documents(client, settings.data_dir, rows["rcept_no"].tolist(), max_docs)
    text_by_no = dict(zip(docs["rcept_no"].astype(str), docs["text"].fillna("").astype(str)))
    states: dict[str, dict] = {}
    hashes, texts = [], []
    for _, row in rows.iterrows():
        text = text_by_no.get(row["rcept_no"], "")
        if not text:
            hashes.append("")
            texts.append("")
            continue
        state = build_state(row["title"], text, row["corp_name"])
        digest = state_hash(state)
        states[digest] = state
        hashes.append(digest)
        texts.append(state["text"])
    rows["state_hash"] = hashes
    rows["has_text"] = [bool(text) for text in texts]
    for name in ANSWERS:
        rows[f"kw_{name}"] = [keyword_flags(text)[name] if text else False for text in texts]
    ledger_path = settings.data_dir / "earnings_quality_answers.csv"
    ledger = call_jev(states, read_csv(ledger_path), api_key=jev_key, max_calls=max_calls, session=session)
    write_csv(ledger if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS), ledger_path)
    answers = ledger[ledger["status"].astype(str).eq("OK") & ledger["prompt_hash"].astype(str).eq(PROMPT_HASH)
                     & ledger["model"].astype(str).eq(MODEL)] if not ledger.empty else ledger
    rows = observable_answers(rows, answers, [f"eq_{name}" for name in ANSWERS])
    for name in ANSWERS:
        rows[f"eq_{name}"] = pd.to_numeric(rows[f"eq_{name}"], errors="coerce")
        rows.loc[~rows["has_text"], f"eq_{name}"] = 0.0  # no explanation → nothing to veto
    rows["jev_veto"] = (rows["eq_one_off"] >= FLAG_THRESHOLD).astype("boolean")
    rows.loc[rows["eq_one_off"].isna(), "jev_veto"] = pd.NA
    rows["kw_veto"] = rows["kw_one_off"] & ~rows["kw_recurring"]
    return rows


# ---------------------------------------------------------------------------
# Replay and evaluation
# ---------------------------------------------------------------------------


def h1_candidates(panel: pd.DataFrame, per_day: int = CANDIDATES_PER_DAY) -> pd.DataFrame:
    hypothesis = next(h for h in HYPOTHESES if h.id == H1_ID)
    if panel.empty or "op_yield" not in panel or panel["op_yield"].isna().all():
        return panel.iloc[0:0].copy()
    scored = panel.assign(_h1=hypothesis_score(panel, hypothesis))
    top = scored.sort_values(["as_of", "_h1"], ascending=[True, False]).groupby("as_of").head(per_day).copy()
    top["h1_rank"] = top.groupby("as_of").cumcount() + 1
    return top


def veto_series(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = []
    for as_of, group in frame.groupby("as_of", sort=True):
        group = group.dropna(subset=[target]).sort_values("h1_rank")
        if len(group) < PICKS * 2:
            continue
        picks = {
            "baseline": group.head(PICKS),
            "keyword_veto": group[~group["kw_veto"].astype(bool)].head(PICKS),
        }
        record: dict[str, Any] = {"as_of": as_of}
        if group["jev_veto"].isna().any():
            # Jev calls are off for new dates: keep the day for keyword-only comparisons.
            record["jev_veto"] = float("nan")
            record["jev_veto_severe"] = float("nan")
        else:
            picks["jev_veto"] = group[~group["jev_veto"].astype(bool)].head(PICKS)
        for label, chosen in picks.items():
            record[label] = float(chosen[target].sum() / PICKS)
            record[f"{label}_severe"] = float((chosen["worst20"] <= SEVERE_LOSS).sum() / PICKS) if "worst20" in chosen else float("nan")
        out.append(record)
    return pd.DataFrame(out)


def evaluate(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    horizon = settings.research.horizon
    target = f"excess{horizon}"
    confidence = settings.research.confidence
    labelled = frame[frame[target].notna()] if not frame.empty else frame
    result: dict[str, Any] = {
        "version": EQ_VERSION,
        "prompt_hash": PROMPT_HASH,
        "model": MODEL,
        "rows": int(len(labelled)),
        "rows_with_text": int(labelled["has_text"].sum()) if not labelled.empty else 0,
        "jev_answered": int(labelled["eq_one_off"].notna().sum()) if not labelled.empty else 0,
    }
    flags = []
    for source, column in (("jev", "jev_veto"), ("kw", "kw_veto")):
        if labelled.empty:
            break
        flag = labelled[column].astype("boolean")
        valid = flag.notna()
        hit = labelled[valid & flag.fillna(False)]
        rest = labelled[valid & ~flag.fillna(False)]
        flags.append(
            {
                "source": source,
                "flag_rate": float(flag[valid].mean()) if valid.any() else float("nan"),
                "flagged": int(len(hit)),
                "excess_flagged": float(hit[target].mean()) if len(hit) else float("nan"),
                "excess_clean": float(rest[target].mean()) if len(rest) else float("nan"),
                "severe_flagged": float((hit["worst20"] <= SEVERE_LOSS).mean()) if len(hit) and "worst20" in hit else float("nan"),
                "severe_clean": float((rest["worst20"] <= SEVERE_LOSS).mean()) if len(rest) and "worst20" in rest else float("nan"),
            }
        )
    result["flags"] = flags
    series = veto_series(labelled, target) if not labelled.empty else pd.DataFrame()
    result["dates"] = int(len(series))
    portfolios: dict[str, Any] = {}
    if not series.empty:
        for column in ("baseline", "jev_veto", "keyword_veto"):
            portfolios[column] = summarize_series(series[column], horizon, confidence=confidence).as_dict()
            portfolios[f"{column}_severe"] = float(series[f"{column}_severe"].mean())
        for a, b in (("jev_veto", "baseline"), ("keyword_veto", "baseline"), ("jev_veto", "keyword_veto")):
            portfolios[f"{a}_minus_{b}"] = summarize_series(series[a] - series[b], horizon, confidence=confidence).as_dict()
    result["portfolios"] = portfolios

    text_share = result["rows_with_text"] / max(result["rows"], 1)
    reasons: list[str] = []
    if text_share < 0.2:
        verdict = "INSUFFICIENT_DATA"
        reasons.append(f"설명 공시 본문이 있는 후보 {text_share:.0%}: DART 수집을 확인하세요")
    elif result["dates"] < MIN_DATES:
        verdict = "INSUFFICIENT_DATA"
        reasons.append(f"평가 신호일 {result['dates']}/{MIN_DATES}")
    else:
        jev_kw = portfolios["jev_veto_minus_keyword_veto"]
        jev_base = portfolios["jev_veto_minus_baseline"]
        kw_base = portfolios["keyword_veto_minus_baseline"]
        if jev_kw["low"] > 0 and jev_base["mean"] > 0:
            verdict = "ADDS_VALUE"
            reasons.append(f"Jev 제외 − 키워드 제외 {jev_kw['mean']:+.2%} (하한 {jev_kw['low']:+.2%})")
        elif kw_base["low"] > 0:
            verdict = "KEYWORDS_ENOUGH"
            reasons.append(f"키워드 제외만으로 H1 대비 {kw_base['mean']:+.2%} (하한 {kw_base['low']:+.2%})")
        else:
            verdict = "NO_VALUE"
            reasons.append(f"Jev − 키워드 하한 {jev_kw['low']:+.2%}, 키워드 − H1 하한 {kw_base['low']:+.2%}")
    result["verdict"] = verdict
    result["reasons"] = reasons
    return result


VERDICT_LABELS = {
    "INSUFFICIENT_DATA": "표본 부족",
    "ADDS_VALUE": "Jev 추가 가치 확인",
    "KEYWORDS_ENOUGH": "키워드 규칙으로 충분",
    "NO_VALUE": "효과 없음",
}


def _fmt(value: Any, pct: bool = True) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number * 100:+.2f}%" if pct else f"{number:.1%}"


def render(result: dict[str, Any], generated_at: datetime, horizon: int) -> str:
    lines = [
        "# 이익의 질 검증 (Jev · 가설 H3)",
        "",
        f"- 갱신: {generated_at.isoformat(timespec='seconds')} · `{EQ_VERSION}` · 모델 `{result.get('model')}` · 프롬프트 `{result.get('prompt_hash')}`",
        f"- 대상: 신호일마다 가설 H1 상위 {CANDIDATES_PER_DAY}개. 신호일까지 접수된 최근 {LOOKBACK_DAYS}일 안의 손익구조 변동·잠정실적 공시 본문(회사명·연도 가림)",
        f"- 규칙(사전등록): Jev가 이익 변화의 주원인을 일회성·영업외 항목으로 본(≥{FLAG_THRESHOLD}) 종목을 빼고 H1 상위 {PICKS}개. 키워드 규칙(일회성 단어 있고 영업 단어 없음)과 비교",
        f"- 표본: {result.get('rows', 0)}건 · 본문 있음 {result.get('rows_with_text', 0)}건 · 평가 신호일 {result.get('dates', 0)}일",
        "",
        f"## 판정: **{VERDICT_LABELS.get(result.get('verdict', ''), result.get('verdict', ''))}**",
        "",
        *[f"- {reason}" for reason in result.get("reasons", [])],
        "",
    ]
    if result.get("flags"):
        lines.extend(
            [
                "| 판단 | 일회성 판정 비율 | 건수 | 해당 초과수익 | 비해당 초과수익 | 해당 급락률 | 비해당 급락률 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["flags"]:
            lines.append(
                f"| {'Jev' if row['source'] == 'jev' else '키워드'} | {_fmt(row['flag_rate'], False)} | {row['flagged']} | "
                f"{_fmt(row['excess_flagged'])} | {_fmt(row['excess_clean'])} | {_fmt(row['severe_flagged'], False)} | {_fmt(row['severe_clean'], False)} |"
            )
        lines.append("")
    portfolios = result.get("portfolios") or {}
    if portfolios:
        labels = {
            "baseline": f"H1 상위 {PICKS}",
            "jev_veto": "Jev 일회성 제외",
            "keyword_veto": "키워드 일회성 제외",
            "jev_veto_minus_baseline": "Jev 제외 − H1",
            "keyword_veto_minus_baseline": "키워드 제외 − H1",
            "jev_veto_minus_keyword_veto": "Jev 제외 − 키워드 제외",
        }
        lines.extend(["| 방식 | 신호일 | 평균 초과수익 | 하한 | 상한 | 급락 비율 |", "|---|---:|---:|---:|---:|---:|"])
        for key, label in labels.items():
            item = portfolios.get(key, {})
            severe = portfolios.get(f"{key}_severe")
            lines.append(
                f"| {label} | {item.get('dates', 0)} | {_fmt(item.get('mean'))} | {_fmt(item.get('low'))} | {_fmt(item.get('high'))} | "
                f"{_fmt(severe, False) if severe is not None else ''} |"
            )
        lines.append("")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _clients(dart_client: DartClient | None, jev_key: str | None) -> tuple[DartClient | None, str]:
    if dart_client is None:
        key = os.getenv("DART_API_KEY", "")
        dart_client = DartClient(key) if key else None
    return dart_client, (jev_key if jev_key is not None else os.getenv("TYPESAFE_API_KEY", ""))


def run_earnings_quality(
    settings: Settings,
    *,
    panel: pd.DataFrame | None = None,
    max_calls: int = 2000,
    max_docs: int = 3000,
    workers: int | None = None,
    dart_client: DartClient | None = None,
    jev_key: str | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(ZoneInfo("Asia/Seoul"))
    horizon = settings.research.horizon
    if panel is None:
        panel = build_research_panel(settings, workers=workers)
    labelled = panel[pd.to_numeric(panel.get(f"excess{horizon}"), errors="coerce").notna()] if not panel.empty else panel
    candidates = h1_candidates(labelled)
    client, jev = _clients(dart_client, jev_key)
    if candidates.empty:
        result = {"version": EQ_VERSION, "verdict": "INSUFFICIENT_DATA", "reasons": ["H1 후보 없음(재무 데이터 확인)"], "rows": 0}
    else:
        reviewed = review_rows(candidates, settings, client=client, jev_key=jev, max_calls=max_calls, max_docs=max_docs, session=session)
        merged = candidates.merge(reviewed, on=["as_of", "symbol"], how="left")
        result = evaluate(merged, settings)
    result["generated_at"] = generated_at.isoformat(timespec="seconds")
    result["pit_version"] = PIT_VERSION
    result["availability_policy"] = "after_receipt_and_answer_capture"
    (settings.reports_dir / "earnings_quality.md").write_text(render(result, generated_at, horizon), encoding="utf-8")
    (settings.data_dir / "earnings_quality_verdict.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Earnings quality: verdict=%s dates=%s", result.get("verdict"), result.get("dates"))
    return result


def apply_earnings_quality(
    scored: pd.DataFrame,
    settings: Settings,
    *,
    dart_client: DartClient | None = None,
    jev_key: str | None = None,
    session: Any | None = None,
) -> pd.DataFrame:
    """Daily: flag today's H1 top 20. Reference and paper trading only."""

    result = scored.copy()
    for column in ("eq_one_off", "eq_recurring"):
        result[column] = np.nan
    result["eq_kw_veto"] = pd.NA
    if result.empty or "op_yield" not in result or result["op_yield"].isna().all():
        return result
    pool = result
    if "discovery_eligible" in result:
        pool = result[result["discovery_eligible"].fillna(False).astype(bool)]
    as_of = pd.to_datetime(pool["as_of"]).max()
    candidates = h1_candidates(pool.assign(as_of=as_of))
    if jev_key is None and not settings.jev.daily_calls:
        jev_key = ""  # keyword flags only; no paid calls
    client, jev = _clients(dart_client, jev_key)
    if candidates.empty or client is None:
        return result
    try:
        reviewed = review_rows(candidates, settings, client=client, jev_key=jev, max_calls=40, max_docs=40, session=session)
    except Exception as exc:  # never block the report
        LOGGER.warning("Earnings-quality review failed without blocking the report: %s", exc)
        return result
    lookup = reviewed.set_index("symbol")
    for index, row in result.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        if symbol in lookup.index:
            item = lookup.loc[symbol]
            result.at[index, "eq_one_off"] = item["eq_one_off"]
            result.at[index, "eq_recurring"] = item["eq_recurring"]
            result.at[index, "eq_kw_veto"] = bool(item["kw_veto"])
    return result


def eq_section(scored: pd.DataFrame, verdict: dict[str, Any] | None) -> tuple[str, str]:
    """Report block: today's H1 candidates whose latest profit explanation looks one-off."""

    import html as html_lib

    if "eq_one_off" not in scored or scored["eq_one_off"].isna().all():
        return "", ""
    label = VERDICT_LABELS.get((verdict or {}).get("verdict", ""), "검증 전")
    flagged = scored[(scored["eq_one_off"] >= FLAG_THRESHOLD) | (scored["eq_kw_veto"] == True)]  # noqa: E712
    lines = [
        f"{row.get('name', row['symbol'])} ({row['symbol']}): 일회성 이익 의심"
        + (f" · Jev {float(row['eq_one_off']):.2f}" if pd.notna(row["eq_one_off"]) else "")
        + (" · 키워드" if row.get("eq_kw_veto") is True else "")
        for _, row in flagged.iterrows()
    ] or ["H1 상위 후보 중 일회성 이익 의심 없음"]
    note = f"이익의 질 검증(H3): {label}. 판정 전에는 참고용이며 순위·매수 조건에 반영하지 않습니다."
    markdown = "\n\n## 이익의 질 점검 (H1 후보)\n- " + note + "\n" + "\n".join(f"- {line}" for line in lines) + "\n"
    html_block = (
        "<section class='panel'><h2>이익의 질 점검 (H1 후보)</h2><p>" + html_lib.escape(note) + "</p><ul>"
        + "".join(f"<li>{html_lib.escape(line)}</li>" for line in lines) + "</ul></section>"
    )
    return markdown, html_block
