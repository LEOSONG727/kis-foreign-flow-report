"""Historical replay: does Jev add anything the numbers it reads do not?

Jev only receives anonymous end-of-day numbers (no name, code, date or score),
so asking it about past signal dates cannot leak what happened next. That lets
us judge the daily Jev shadow in one run instead of waiting months for the
prospective ledger.

For every labelled signal date the same 12-name discovery shortlist used live
is sent through the identical request/parse path. Three things are measured:

1. Replicability: how much of Jev's risk answer is a plain linear function of
   the ten numbers it was shown. A near-perfect fit means code can replace it.
2. Incremental skill: within each day's shortlist, does Jev's risk still rank
   the next 20 sessions after removing those same ten numbers?
3. Use: does "top 3 by lowest Jev risk" beat "top 3 by discovery score" on the
   same dates, net of costs and versus the same-market equal-weight mean?

A pre-registered verdict decides whether the daily Jev calls keep running.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .config import Settings
from .jev_shadow import (
    LIMIT,
    MODEL,
    POLICY,
    PROMPT_HASH,
    QUESTIONS,
    REQUIRED,
    _number,
    _state,
    request_review,
    select_candidates,
)
from .research import build_research_panel, summarize_series
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)

REPLAY_VERSION = "jev-replay-v1"
ANSWERS = tuple(f"jev_{key}" for key in QUESTIONS)
RISK_COLUMNS = (*ANSWERS, "jev_max")
# The numbers Jev sees (units converted by code). Used as controls.
INPUT_COLUMNS = (*REQUIRED, "foreign_20_adtv")
LEDGER_COLUMNS = [
    "as_of",
    "symbol",
    "replay_version",
    "policy",
    "prompt_hash",
    "model",
    "status",
    "error",
    *ANSWERS,
    "input_tokens",
    "captured_at",
]
MIN_COMPLETE_DATES = 40
TOP_N = 3
REJECT_THRESHOLD = 0.65

VERDICT_LABELS = {
    "INSUFFICIENT_DATA": "표본 부족",
    "NO_VALUE": "추가 가치 없음",
    "ADDS_VALUE": "추가 가치 확인",
}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_shortlists(panel: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Same 12-name shortlist the live shadow would have sent, per labelled date."""

    if panel.empty:
        return panel.copy()
    labelled = panel[pd.to_numeric(panel.get(f"excess{horizon}"), errors="coerce").notna()]
    parts = [select_candidates(group, LIMIT) for _, group in labelled.groupby("as_of", sort=True)]
    parts = [part for part in parts if not part.empty]
    return pd.concat(parts, ignore_index=True) if parts else labelled.iloc[0:0].copy()


def _ledger_key(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(frame["as_of"]).dt.strftime("%Y-%m-%d")
        + "|"
        + frame["symbol"].astype(str).str.zfill(6)
        + "|"
        + frame.get("prompt_hash", pd.Series(PROMPT_HASH, index=frame.index)).astype(str)
    )


def run_replay_calls(
    shortlist: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    api_key: str,
    max_calls: int,
    session: Any | None = None,
    pause: Callable[[float], None] = time.sleep,
    spacing_seconds: float = 0.2,
) -> pd.DataFrame:
    """Call Jev for shortlist rows without a successful answer yet (resumable)."""

    if shortlist.empty:
        return ledger
    existing = ledger.copy() if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS)
    for column in LEDGER_COLUMNS:
        if column not in existing:
            existing[column] = pd.NA
    done = set(_ledger_key(existing[existing["status"].astype(str) == "OK"])) if not existing.empty else set()
    todo = shortlist[~_ledger_key(shortlist.assign(prompt_hash=PROMPT_HASH)).isin(done)]
    if todo.empty or max_calls <= 0:
        return existing[LEDGER_COLUMNS]
    http = session or requests.Session()
    rows: list[dict[str, Any]] = []
    calls = 0
    try:
        for _, row in todo.iterrows():
            if calls >= max_calls:
                LOGGER.info("Jev replay call budget reached (%d)", max_calls)
                break
            record: dict[str, Any] = {
                "as_of": pd.Timestamp(row["as_of"]).strftime("%Y-%m-%d"),
                "symbol": str(row["symbol"]).zfill(6),
                "replay_version": REPLAY_VERSION,
                "policy": POLICY,
                "prompt_hash": PROMPT_HASH,
                "model": MODEL,
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
            if any(_number(row.get(field)) is None for field in REQUIRED):
                record["status"] = "INSUFFICIENT_DATA"
                rows.append(record)
                continue
            status, values, error, auth_failed = request_review(http, api_key, _state(row), pause)
            calls += 1
            record["status"] = status
            record["error"] = error
            for column in ANSWERS:
                record[column] = values.get(column)
            record["input_tokens"] = values.get("jev_input_tokens")
            rows.append(record)
            if auth_failed:
                LOGGER.error("Jev replay stopped: authentication/validation failure (%s)", error)
                break
            if calls % 100 == 0:
                LOGGER.info("Jev replay progress: %d calls", calls)
            if spacing_seconds > 0:
                pause(spacing_seconds)
    finally:
        if session is None:
            http.close()
    if not rows:
        return existing[LEDGER_COLUMNS]
    incoming = pd.DataFrame(rows)
    for column in LEDGER_COLUMNS:
        if column not in incoming:
            incoming[column] = pd.NA
    parts = [frame[LEDGER_COLUMNS] for frame in (existing, incoming) if not frame.empty]
    combined = pd.concat(parts, ignore_index=True)
    combined["_ok"] = combined["status"].astype(str).eq("OK")
    combined["_key"] = _ledger_key(combined)
    # Keep one row per key, preferring a successful answer, else the latest try.
    combined = combined.sort_values(["_key", "_ok", "captured_at"]).drop_duplicates("_key", keep="last")
    return combined.drop(columns=["_ok", "_key"]).sort_values(["as_of", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def merge_replay(shortlist: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Shortlist rows joined with successful answers; only complete days kept."""

    if shortlist.empty or ledger.empty:
        return pd.DataFrame()
    answers = ledger[
        ledger["status"].astype(str).eq("OK") & ledger["prompt_hash"].astype(str).eq(PROMPT_HASH)
    ].copy()
    answers["as_of"] = pd.to_datetime(answers["as_of"])
    answers["symbol"] = answers["symbol"].astype(str).str.zfill(6)
    for column in ANSWERS:
        answers[column] = pd.to_numeric(answers[column], errors="coerce")
    base = shortlist.copy()
    base["as_of"] = pd.to_datetime(base["as_of"])
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    base["baseline_rank"] = base.groupby("as_of").cumcount() + 1
    merged = base.merge(answers[["as_of", "symbol", *ANSWERS]], on=["as_of", "symbol"], how="left")
    merged["jev_max"] = merged[list(ANSWERS)].max(axis=1, skipna=False)
    size = merged.groupby("as_of")["symbol"].transform("count")
    answered = merged.groupby("as_of")["jev_max"].transform(lambda values: values.notna().all())
    return merged[answered & (size >= TOP_N * 2)].reset_index(drop=True)


def _within_date_ranks(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    ranks = frame.groupby("as_of")[columns].rank(pct=True) - 0.5
    return ranks.fillna(0.0).to_numpy(dtype=float)


def _pooled_partial_corr(frame: pd.DataFrame, signal: str, target: str, controls: list[str]) -> float:
    """Pooled correlation of within-date ranks after removing the controls."""

    data = frame.dropna(subset=[signal, target])
    if len(data) < 30:
        return float("nan")
    x = _within_date_ranks(data, [signal])[:, 0]
    y = _within_date_ranks(data, [target])[:, 0]
    if controls:
        design = _within_date_ranks(data, controls)
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _date_block_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    block: int,
    samples: int,
    confidence: float,
    seed: int = 20260928,
) -> tuple[float, float, float]:
    dates = sorted(frame["as_of"].unique())
    point = statistic(frame)
    if len(dates) < 3 or not np.isfinite(point):
        return point, float("nan"), float("nan")
    groups = {date: group for date, group in frame.groupby("as_of")}
    rng = np.random.default_rng(seed)
    block = max(1, min(block, len(dates)))
    estimates: list[float] = []
    for draw in range(samples):
        chosen: list[pd.DataFrame] = []
        while len(chosen) < len(dates):
            start = int(rng.integers(len(dates)))
            for offset in range(block):
                # Relabel each draw so duplicated days stay separate groups.
                chosen.append(
                    groups[dates[(start + offset) % len(dates)]].assign(
                        as_of=pd.Timestamp("2000-01-01") + pd.Timedelta(days=len(chosen))
                    )
                )
        value = statistic(pd.concat(chosen[: len(dates)], ignore_index=True))
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return point, float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    return point, float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))


def _replica_basis(frame: pd.DataFrame) -> np.ndarray:
    """Flexible code-only view of the numbers Jev saw: rank, clipped raw, square."""

    columns = [np.ones(len(frame))]
    for name in INPUT_COLUMNS:
        raw = pd.to_numeric(frame[name], errors="coerce")
        low, high = raw.quantile(0.01), raw.quantile(0.99)
        clipped = raw.clip(low, high)
        spread = clipped.std() or 1.0
        standard = ((clipped - clipped.mean()) / spread).fillna(0.0).to_numpy(dtype=float)
        rank = (raw.rank(pct=True) - 0.5).fillna(0.0).to_numpy(dtype=float)
        columns.extend([standard, rank, standard**2, rank**2])
    return np.column_stack(columns)


def code_replica(frame: pd.DataFrame, signal: str = "jev_max", folds: int = 5) -> pd.Series:
    """Out-of-fold prediction of Jev's answer from its inputs alone.

    Folds are contiguous date blocks, so a replica never sees the same day it
    predicts. What the replica cannot reproduce is Jev's own judgment.
    """

    data = frame.dropna(subset=[signal])
    prediction = pd.Series(np.nan, index=frame.index)
    codes = pd.factorize(pd.to_datetime(data["as_of"]), sort=True)[0]
    if codes.max(initial=-1) + 1 < folds * 2:
        return prediction
    basis = _replica_basis(data)
    target = pd.to_numeric(data[signal], errors="coerce").to_numpy(dtype=float)
    for block in np.array_split(np.arange(codes.max() + 1), folds):
        test = np.isin(codes, block)
        train = ~test
        x = basis[train]
        gram = x.T @ x / len(x) + 1e-3 * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x.T @ target[train] / len(x))
        prediction.loc[data.index[test]] = basis[test] @ weights
    return prediction


def replicability_r2(frame: pd.DataFrame, signal: str = "jev_max") -> float:
    """Out-of-fold share of Jev's answer reproduced by code from its inputs."""

    replica = code_replica(frame, signal)
    valid = replica.notna() & frame[signal].notna()
    if valid.sum() < 30:
        return float("nan")
    y = pd.to_numeric(frame.loc[valid, signal], errors="coerce").to_numpy(dtype=float)
    fitted = replica[valid].to_numpy(dtype=float)
    total = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - fitted) ** 2).sum() / total) if total > 0 else float("nan")


def portfolio_series(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per-date top-3 excess for baseline, lowest-risk and reject-filter policies."""

    rows = []
    for as_of, group in frame.groupby("as_of", sort=True):
        group = group.dropna(subset=[target])
        if len(group) < TOP_N * 2:
            continue
        baseline = group.sort_values("baseline_rank").head(TOP_N)
        lowest = group.sort_values(["jev_max", "baseline_rank"]).head(TOP_N)
        kept = group[group["jev_max"] < REJECT_THRESHOLD].sort_values("baseline_rank").head(TOP_N)
        # Unfilled slots stay in cash (0 excess) so abstaining cannot win by itself.
        rows.append(
            {
                "as_of": as_of,
                "baseline": float(baseline[target].mean()),
                "jev_lowest_risk": float(lowest[target].mean()),
                "jev_reject_filter": float(kept[target].sum() / TOP_N),
            }
        )
    return pd.DataFrame(rows)


def evaluate_replay(merged: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    research = settings.research
    horizon = research.horizon
    target = f"excess{horizon}"
    kwargs = {"block": horizon, "samples": min(research.bootstrap_samples, 500), "confidence": research.confidence}
    dates = int(merged["as_of"].nunique()) if not merged.empty else 0
    result: dict[str, Any] = {
        "replay_version": REPLAY_VERSION,
        "prompt_hash": PROMPT_HASH,
        "model": MODEL,
        "complete_dates": dates,
        "rows": int(len(merged)),
        "signals": {},
        "portfolios": {},
    }
    if dates < 3:
        result["verdict"] = "INSUFFICIENT_DATA"
        result["reasons"] = [f"완결 신호일 {dates}/{MIN_COMPLETE_DATES}"]
        return result
    result["replicability_r2"] = replicability_r2(merged)
    result["answer_means"] = {column: float(merged[column].mean()) for column in RISK_COLUMNS}
    result["reject_share"] = float((merged["jev_max"] >= REJECT_THRESHOLD).mean())
    for signal in RISK_COLUMNS:
        entry: dict[str, Any] = {}
        for h in (5, horizon):
            entry[f"corr{h}"] = _date_block_bootstrap(
                merged, lambda frame, s=signal, t=f"excess{h}": _pooled_partial_corr(frame, s, t, []), **kwargs
            )
        # Jev's own judgment = answer minus what code reproduces from the same inputs.
        unique = merged.assign(_unique=pd.to_numeric(merged[signal], errors="coerce") - code_replica(merged, signal))
        entry["incremental"] = _date_block_bootstrap(
            unique, lambda frame: _pooled_partial_corr(frame, "_unique", target, []), **kwargs
        )
        entry["replicability_r2"] = replicability_r2(merged, signal)
        result["signals"][signal] = entry
    series = portfolio_series(merged, target)
    for column in ("baseline", "jev_lowest_risk", "jev_reject_filter"):
        result["portfolios"][column] = summarize_series(series[column], horizon, confidence=research.confidence).as_dict()
    for column in ("jev_lowest_risk", "jev_reject_filter"):
        result["portfolios"][f"{column}_minus_baseline"] = summarize_series(
            series[column] - series["baseline"], horizon, confidence=research.confidence
        ).as_dict()

    incremental = result["signals"]["jev_max"]["incremental"]
    delta = result["portfolios"]["jev_lowest_risk_minus_baseline"]
    reasons: list[str] = []
    if dates < MIN_COMPLETE_DATES:
        result["verdict"] = "INSUFFICIENT_DATA"
        reasons.append(f"완결 신호일 {dates}/{MIN_COMPLETE_DATES}")
    elif incremental[2] < 0 and delta["low"] > 0:
        # Risk must predict *lower* returns beyond its inputs, and using it must pay.
        result["verdict"] = "ADDS_VALUE"
        reasons.append(
            f"Jev 고유 판단-수익 상관 {incremental[0]:+.3f} (상한 {incremental[2]:+.3f}), "
            f"저위험 Top {TOP_N} − 기존 {delta['mean']:+.2%} (하한 {delta['low']:+.2%})"
        )
    else:
        result["verdict"] = "NO_VALUE"
        if not incremental[2] < 0:
            reasons.append(f"Jev 고유 판단-수익 상관 상한 {incremental[2]:+.3f} ≥ 0")
        if not delta["low"] > 0:
            reasons.append(f"저위험 Top {TOP_N} − 기존 하한 {delta['low']:+.2%} ≤ 0")
    result["reasons"] = reasons
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(value: Any, pct: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number * 100:+.2f}%" if pct else f"{number:+.3f}"


def render_replay_markdown(result: dict[str, Any], generated_at: datetime, horizon: int) -> str:
    verdict = result.get("verdict", "INSUFFICIENT_DATA")
    lines = [
        "# Jev 과거 재생 검증",
        "",
        f"- 갱신: {generated_at.isoformat(timespec='seconds')} · `{REPLAY_VERSION}` · 모델 `{result.get('model')}` · 프롬프트 `{result.get('prompt_hash')}`",
        f"- 표본: 완결 신호일 {result.get('complete_dates', 0)}일 · {result.get('rows', 0)}건 (일일 발견 상위 {LIMIT}개, 실전과 같은 입력·파싱)",
        "- Jev 입력에는 종목명·코드·날짜·점수가 없어 과거 날짜를 물어도 미래정보가 섞이지 않습니다.",
        f"- 수익률: 다음 거래일 시가 → D+{horizon} 종가, 비용 차감, 같은 시장 동일가중 평균 대비",
        "",
        f"## 판정: **{VERDICT_LABELS.get(verdict, verdict)}**",
        "",
        *[f"- {reason}" for reason in result.get("reasons", [])],
        "- 추가 가치 없음이면 일일 Jev 호출을 자동으로 멈춥니다(`data/jev_verdict.json`). 프롬프트·입력을 바꾸면 해시가 바뀌어 다시 검증합니다.",
        "",
    ]
    if not result.get("signals"):
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## 1. 입력 숫자로 재현되는 정도",
            "",
            f"- 최대 위험 답변의 R² {_fmt(result.get('replicability_r2'))}를 입력 숫자 {len(INPUT_COLUMNS)}개만 쓰는 코드 모델이 재현합니다(날짜 블록 교차적합).",
            f"- 평균 답변: " + " · ".join(f"`{key}` {value:.2f}" for key, value in result.get("answer_means", {}).items()),
            f"- REJECT(≥{REJECT_THRESHOLD}) 비율 {result.get('reject_share', float('nan')):.1%}",
            "",
            "## 2. 위험 답변과 이후 초과수익 (같은 날 후보 안에서)",
            "",
            "음수면 위험이 높다고 본 종목이 실제로 덜 올랐다는 뜻입니다. 고유 판단 열은 같은 입력으로 코드가 재현한 부분을 뺀 나머지의 상관입니다. 0에 가까우면 Jev 없이 코드 규칙으로 같은 효과를 낼 수 있습니다.",
            "",
            f"| 답변 | 코드 재현 R² | D+5 상관 | D+{horizon} 상관 | 구간 | 고유 판단 | 구간 |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
    )
    for signal, entry in result["signals"].items():
        raw = entry[f"corr{horizon}"]
        inc = entry["incremental"]
        lines.append(
            f"| {signal} | {_fmt(entry.get('replicability_r2'))} | {_fmt(entry['corr5'][0])} | {_fmt(raw[0])} | {_fmt(raw[1])} ~ {_fmt(raw[2])} | "
            f"{_fmt(inc[0])} | {_fmt(inc[1])} ~ {_fmt(inc[2])} |"
        )
    lines.extend(
        [
            "",
            f"## 3. Top {TOP_N} 비교 (같은 날, 빈 자리는 현금)",
            "",
            "| 방식 | 신호일 | 평균 초과수익 | 하한 | 상한 | 양수 비율 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "baseline": "기존 발견 순위",
        "jev_lowest_risk": "Jev 최저위험",
        "jev_reject_filter": "기존 순위 + Jev REJECT 제외",
        "jev_lowest_risk_minus_baseline": "Jev 최저위험 − 기존",
        "jev_reject_filter_minus_baseline": "REJECT 제외 − 기존",
    }
    for key, label in labels.items():
        item = result["portfolios"].get(key, {})
        lines.append(
            f"| {label} | {item.get('dates', 0)} | {_fmt(item.get('mean'), True)} | {_fmt(item.get('low'), True)} | "
            f"{_fmt(item.get('high'), True)} | {_fmt(item.get('positive_share'), True).replace('+', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_jev_verdict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def daily_jev_allowed(verdict: dict[str, Any] | None) -> bool:
    """Stop paying for daily calls only after this exact prompt was shown useless."""

    if not verdict:
        return True
    return not (
        verdict.get("verdict") == "NO_VALUE"
        and verdict.get("prompt_hash") == PROMPT_HASH
        and verdict.get("model") == MODEL
    )


def run_jev_replay(
    settings: Settings,
    *,
    max_calls: int = 1500,
    workers: int | None = None,
    api_key: str | None = None,
    session: Any | None = None,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(ZoneInfo("Asia/Seoul"))
    key = api_key if api_key is not None else os.getenv("TYPESAFE_API_KEY", "")
    horizon = settings.research.horizon
    if panel is None:
        panel = build_research_panel(settings, workers=workers)
    shortlist = replay_shortlists(panel, horizon)
    ledger_path = settings.data_dir / "jev_replay.csv"
    ledger = read_csv(ledger_path)
    if key:
        ledger = run_replay_calls(shortlist, ledger, api_key=key, max_calls=max_calls, session=session)
    else:
        LOGGER.warning("TYPESAFE_API_KEY is missing; evaluating stored replay answers only")
    write_csv(ledger if not ledger.empty else pd.DataFrame(columns=LEDGER_COLUMNS), ledger_path)
    merged = merge_replay(shortlist, ledger)
    result = evaluate_replay(merged, settings)
    result["generated_at"] = generated_at.isoformat(timespec="seconds")
    result["shortlist_rows"] = int(len(shortlist))
    (settings.reports_dir / "jev_replay.md").write_text(
        render_replay_markdown(result, generated_at, horizon), encoding="utf-8"
    )
    (settings.data_dir / "jev_verdict.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "Jev replay: verdict=%s complete_dates=%d rows=%d",
        result["verdict"],
        result["complete_dates"],
        result["rows"],
    )
    return result
