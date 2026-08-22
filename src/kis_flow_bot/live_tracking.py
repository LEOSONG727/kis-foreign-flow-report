from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .storage import read_csv, write_csv

HORIZONS = (5, 10, 20)
BUY_REVIEW = "오늘 신규매수 검토"
TRACKED_GRADES = {"A", "B", "C"}

RUN_COLUMNS = [
    "signal_date",
    "captured_at",
    "candidate_count",
    "model_version",
    "config_hash",
]

SNAPSHOT_COLUMNS = [
    "signal_date",
    "captured_at",
    "symbol",
    "name",
    "market",
    "close",
    "discovery_grade",
    "proposal_status",
    "proposal_priority",
    "discovery_score",
    "timing_score",
    "distribution_score",
    "stage",
    "campaign_regime",
    "watch_tier",
    "entry_readiness",
    "entry_gate_passed",
    "foreign_1_adtv",
    "foreign_3_adtv",
    "foreign_5_adtv",
    "foreign_20_adtv",
    "foreign_buy_days10",
    "sellback_ratio20",
    "price_return3",
    "price_return20",
    "rebound_from_low20",
    "data_fresh",
    "liquidity_eligible",
    "history_eligible",
    "model_version",
    "config_hash",
]

EPISODE_COLUMNS = [
    "episode_id",
    "symbol",
    "name",
    "market",
    "started_at",
    "last_seen_at",
    "ended_at",
    "closed_at",
    "start_grade",
    "start_proposal",
    "current_grade",
    "current_proposal",
    "first_buy_review_date",
    "model_version",
    "config_hash",
]

EVENT_COLUMNS = [
    "event_id",
    "episode_id",
    "event_date",
    "event_type",
    "symbol",
    "name",
    "market",
    "reference_close",
    "discovery_grade",
    "proposal_status",
    "discovery_score",
    "timing_score",
    "distribution_score",
    "stage",
    "campaign_regime",
    "foreign_3_adtv",
    "foreign_20_adtv",
    "foreign_buy_days10",
    "sellback_ratio20",
    "price_return20",
    "model_version",
    "config_hash",
]

OUTCOME_COLUMNS = [
    "event_id",
    "episode_id",
    "event_date",
    "event_type",
    "symbol",
    "name",
    "market",
    "discovery_grade",
    "proposal_status",
    "reference_close",
    "entry_date",
    "entry_open",
    "observed_sessions",
    "completed_horizon",
    "status",
    "event_outcome",
    "days_to_event",
    "max_upside20",
    "max_drawdown20",
    "return_5",
    "net_return_5",
    "signal_close_return_5",
    "market_return_5",
    "excess_return_5",
    "return_10",
    "net_return_10",
    "signal_close_return_10",
    "market_return_10",
    "excess_return_10",
    "return_20",
    "net_return_20",
    "signal_close_return_20",
    "market_return_20",
    "excess_return_20",
    "evaluated_through",
    "model_version",
    "config_hash",
]


@dataclass(frozen=True)
class LiveTrackingResult:
    snapshot_rows: int
    episodes: int
    events: int
    resolved_20: int
    report_path: Path


def _blank(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _with_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result[columns]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value) if value is not None and not pd.isna(value) else False


def tracking_config_hash(settings: Settings) -> str:
    """Fingerprint every setting that can change the recorded recommendation."""

    payload = {
        "screening": asdict(settings.screening),
        "model": asdict(settings.model),
        "brief": asdict(settings.brief),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_daily_snapshots(
    scored: pd.DataFrame,
    captured_at: datetime,
    model_version: str,
    config_hash: str,
) -> pd.DataFrame:
    """Build the immutable structured record of today's eligible A/B/C names."""

    if scored.empty:
        return _blank(SNAPSHOT_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, source in scored.iterrows():
        grade = _text(source.get("discovery_grade"))
        if grade not in TRACKED_GRADES or not _truthy(source.get("discovery_eligible", False)):
            continue
        signal_date = pd.to_datetime(source.get("as_of"), errors="coerce")
        if pd.isna(signal_date):
            continue
        row = {column: source.get(column, pd.NA) for column in SNAPSHOT_COLUMNS}
        row.update(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "captured_at": captured_at.isoformat(),
                "symbol": _text(source.get("symbol")).zfill(6),
                "name": _text(source.get("name")),
                "market": _text(source.get("market")),
                "discovery_grade": grade,
                "proposal_status": _text(source.get("proposal_status")),
                "model_version": model_version,
                "config_hash": config_hash,
            }
        )
        rows.append(row)
    if not rows:
        return _blank(SNAPSHOT_COLUMNS)
    return (
        pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
        .sort_values(["signal_date", "proposal_priority", "discovery_score", "symbol"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )


def append_official_snapshots(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Append one official candidate set per market date; never rewrite it on reruns."""

    current = _with_columns(existing, SNAPSHOT_COLUMNS) if not existing.empty else _blank(SNAPSHOT_COLUMNS)
    if incoming.empty:
        return current
    recorded_dates = set(pd.to_datetime(current["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d"))
    additions = incoming[~incoming["signal_date"].isin(recorded_dates)]
    if additions.empty:
        return current
    if current.empty:
        return (
            additions.drop_duplicates(["signal_date", "symbol"], keep="first")
            .sort_values(["signal_date", "symbol"])
            .reset_index(drop=True)[SNAPSHOT_COLUMNS]
        )
    return (
        pd.concat([current, additions], ignore_index=True)
        .drop_duplicates(["signal_date", "symbol"], keep="first")
        .sort_values(["signal_date", "symbol"])
        .reset_index(drop=True)[SNAPSHOT_COLUMNS]
    )


def _trading_dates(history: pd.DataFrame) -> list[pd.Timestamp]:
    if history.empty or "date" not in history:
        return []
    dates = pd.to_datetime(history["date"], errors="coerce").dropna().dt.normalize().unique()
    return sorted(pd.Timestamp(value) for value in dates)


def _session_count(
    dates: list[pd.Timestamp],
    after: str,
    before: str,
    include_end: bool,
) -> int:
    start = pd.Timestamp(after).normalize()
    end = pd.Timestamp(before).normalize()
    return sum(start < item <= end if include_end else start < item < end for item in dates)


def _event_from_snapshot(
    snapshot: pd.Series,
    episode_id: str,
    event_type: str,
) -> dict[str, Any]:
    event_date = _text(snapshot.get("signal_date"))
    event_id = f"{episode_id}:{event_type}:{event_date.replace('-', '')}"
    return {
        "event_id": event_id,
        "episode_id": episode_id,
        "event_date": event_date,
        "event_type": event_type,
        "symbol": _text(snapshot.get("symbol")).zfill(6),
        "name": _text(snapshot.get("name")),
        "market": _text(snapshot.get("market")),
        "reference_close": snapshot.get("close", pd.NA),
        "discovery_grade": _text(snapshot.get("discovery_grade")),
        "proposal_status": _text(snapshot.get("proposal_status")),
        "discovery_score": snapshot.get("discovery_score", pd.NA),
        "timing_score": snapshot.get("timing_score", pd.NA),
        "distribution_score": snapshot.get("distribution_score", pd.NA),
        "stage": _text(snapshot.get("stage")),
        "campaign_regime": _text(snapshot.get("campaign_regime")),
        "foreign_3_adtv": snapshot.get("foreign_3_adtv", pd.NA),
        "foreign_20_adtv": snapshot.get("foreign_20_adtv", pd.NA),
        "foreign_buy_days10": snapshot.get("foreign_buy_days10", pd.NA),
        "sellback_ratio20": snapshot.get("sellback_ratio20", pd.NA),
        "price_return20": snapshot.get("price_return20", pd.NA),
        "model_version": _text(snapshot.get("model_version")),
        "config_hash": _text(snapshot.get("config_hash")),
    }


def update_episode_ledger(
    snapshots: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    current_date: str,
    trading_dates: list[pd.Timestamp],
    gap_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update candidate lifecycles and create non-duplicated evaluation anchors."""

    episode_records = (
        _with_columns(episodes, EPISODE_COLUMNS).to_dict("records")
        if not episodes.empty
        else []
    )
    event_records = (
        _with_columns(events, EVENT_COLUMNS).to_dict("records")
        if not events.empty
        else []
    )
    event_ids = {_text(record.get("event_id")) for record in event_records}
    current = snapshots[snapshots["signal_date"].astype(str) == current_date].copy()
    current_symbols = set(current["symbol"].astype(str))

    def active_index(symbol: str) -> int | None:
        candidates = [
            index
            for index, record in enumerate(episode_records)
            if _text(record.get("symbol")).zfill(6) == symbol and not _text(record.get("closed_at"))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda index: _text(episode_records[index].get("started_at")))

    for _, snapshot in current.sort_values(["symbol"]).iterrows():
        symbol = _text(snapshot.get("symbol")).zfill(6)
        index = active_index(symbol)
        if index is not None:
            previous = episode_records[index]
            missing_before = _session_count(
                trading_dates,
                _text(previous.get("last_seen_at")),
                current_date,
                include_end=False,
            )
            if missing_before >= gap_sessions:
                previous["ended_at"] = _text(previous.get("last_seen_at"))
                previous["closed_at"] = current_date
                index = None

        proposal = _text(snapshot.get("proposal_status"))
        grade = _text(snapshot.get("discovery_grade"))
        if index is None:
            episode_id = f"{symbol}-{current_date.replace('-', '')}"
            episode_records.append(
                {
                    "episode_id": episode_id,
                    "symbol": symbol,
                    "name": _text(snapshot.get("name")),
                    "market": _text(snapshot.get("market")),
                    "started_at": current_date,
                    "last_seen_at": current_date,
                    "ended_at": "",
                    "closed_at": "",
                    "start_grade": grade,
                    "start_proposal": proposal,
                    "current_grade": grade,
                    "current_proposal": proposal,
                    "first_buy_review_date": current_date if proposal == BUY_REVIEW else "",
                    "model_version": _text(snapshot.get("model_version")),
                    "config_hash": _text(snapshot.get("config_hash")),
                }
            )
            event_type = "buy_review_entry" if proposal == BUY_REVIEW else "candidate_entry"
            event = _event_from_snapshot(snapshot, episode_id, event_type)
            if event["event_id"] not in event_ids:
                event_records.append(event)
                event_ids.add(event["event_id"])
            continue

        record = episode_records[index]
        record["last_seen_at"] = current_date
        record["current_grade"] = grade
        record["current_proposal"] = proposal
        if proposal == BUY_REVIEW and not _text(record.get("first_buy_review_date")):
            record["first_buy_review_date"] = current_date
            event = _event_from_snapshot(snapshot, _text(record.get("episode_id")), "buy_review_promotion")
            if event["event_id"] not in event_ids:
                event_records.append(event)
                event_ids.add(event["event_id"])

    for record in episode_records:
        symbol = _text(record.get("symbol")).zfill(6)
        if _text(record.get("closed_at")) or symbol in current_symbols:
            continue
        missed = _session_count(
            trading_dates,
            _text(record.get("last_seen_at")),
            current_date,
            include_end=True,
        )
        if missed >= gap_sessions:
            record["ended_at"] = _text(record.get("last_seen_at"))
            record["closed_at"] = current_date

    episode_frame = _with_columns(pd.DataFrame(episode_records), EPISODE_COLUMNS)
    event_frame = _with_columns(pd.DataFrame(event_records), EVENT_COLUMNS)
    if not episode_frame.empty:
        episode_frame = episode_frame.drop_duplicates("episode_id", keep="last").sort_values(
            ["started_at", "symbol"]
        ).reset_index(drop=True)
    if not event_frame.empty:
        event_frame = event_frame.drop_duplicates("event_id", keep="first").sort_values(
            ["event_date", "symbol", "event_type"]
        ).reset_index(drop=True)
    return episode_frame, event_frame


def _market_forward_return(
    history: pd.DataFrame,
    market: str,
    entry_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache: dict[tuple[str, str, str], float | None],
) -> float | None:
    key = (market, entry_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if key in cache:
        return cache[key]
    market_rows = history[history["market"].astype(str) == market]
    starts = market_rows[market_rows["date"] == entry_date][["symbol", "open"]].rename(
        columns={"open": "entry_open"}
    )
    ends = market_rows[market_rows["date"] == end_date][["symbol", "close"]].rename(
        columns={"close": "end_close"}
    )
    paired = starts.merge(ends, on="symbol", how="inner")
    paired = paired[(paired["entry_open"] > 0) & (paired["end_close"] > 0)]
    result = (
        float((paired["end_close"] / paired["entry_open"] - 1.0).median())
        if not paired.empty
        else None
    )
    cache[key] = result
    return result


def _evaluate_event(
    event: pd.Series,
    symbol_history: pd.DataFrame,
    market_history: pd.DataFrame,
    target: float,
    stop: float,
    roundtrip_cost: float,
    market_cache: dict[tuple[str, str, str], float | None],
) -> dict[str, Any]:
    symbol = _text(event.get("symbol")).zfill(6)
    event_date = pd.Timestamp(_text(event.get("event_date"))).normalize()
    reference_close = _number(event.get("reference_close"))
    future = symbol_history[symbol_history["date"] > event_date].copy()
    future = future.sort_values("date").drop_duplicates("date", keep="last").head(max(HORIZONS))
    evaluated_through = market_history["date"].max().strftime("%Y-%m-%d") if not market_history.empty else ""
    result: dict[str, Any] = {
        "event_id": _text(event.get("event_id")),
        "episode_id": _text(event.get("episode_id")),
        "event_date": event_date.strftime("%Y-%m-%d"),
        "event_type": _text(event.get("event_type")),
        "symbol": symbol,
        "name": _text(event.get("name")),
        "market": _text(event.get("market")),
        "discovery_grade": _text(event.get("discovery_grade")),
        "proposal_status": _text(event.get("proposal_status")),
        "reference_close": reference_close,
        "entry_date": "",
        "entry_open": pd.NA,
        "observed_sessions": len(future),
        "completed_horizon": max((h for h in HORIZONS if len(future) >= h), default=0),
        "status": "awaiting_entry",
        "event_outcome": "pending",
        "days_to_event": pd.NA,
        "max_upside20": pd.NA,
        "max_drawdown20": pd.NA,
        "evaluated_through": evaluated_through,
        "model_version": _text(event.get("model_version")),
        "config_hash": _text(event.get("config_hash")),
    }
    for horizon in HORIZONS:
        for prefix in ("return", "net_return", "signal_close_return", "market_return", "excess_return"):
            result[f"{prefix}_{horizon}"] = pd.NA
    if future.empty:
        return result

    entry_open = _number(future.iloc[0].get("open"))
    if not entry_open or entry_open <= 0:
        return result
    entry_date = pd.Timestamp(future.iloc[0]["date"]).normalize()
    result["entry_date"] = entry_date.strftime("%Y-%m-%d")
    result["entry_open"] = entry_open
    result["status"] = "open"

    valid_highs = pd.to_numeric(future["high"], errors="coerce").dropna()
    valid_lows = pd.to_numeric(future["low"], errors="coerce").dropna()
    if not valid_highs.empty:
        result["max_upside20"] = float(valid_highs.max() / entry_open - 1.0)
    if not valid_lows.empty:
        result["max_drawdown20"] = float(valid_lows.min() / entry_open - 1.0)

    outcome = "pending"
    days_to_event: int | None = None
    for day, (_, row) in enumerate(future.iterrows(), 1):
        high = _number(row.get("high"))
        low = _number(row.get("low"))
        hit_target = high is not None and high >= entry_open * (1.0 + target)
        hit_stop = low is not None and low <= entry_open * (1.0 - stop)
        if hit_target and hit_stop:
            outcome, days_to_event = "ambiguous", day
            break
        if hit_target:
            outcome, days_to_event = "target", day
            break
        if hit_stop:
            outcome, days_to_event = "stop", day
            break
    if outcome != "pending":
        result["status"] = "resolved"
        result["event_outcome"] = outcome
        result["days_to_event"] = days_to_event
    elif len(future) >= max(HORIZONS):
        result["status"] = "resolved"
        result["event_outcome"] = "horizon"

    for horizon in HORIZONS:
        if len(future) < horizon:
            continue
        end = future.iloc[horizon - 1]
        end_close = _number(end.get("close"))
        if not end_close or end_close <= 0:
            continue
        gross = end_close / entry_open - 1.0
        end_date = pd.Timestamp(end["date"]).normalize()
        market_return = _market_forward_return(
            market_history,
            _text(event.get("market")),
            entry_date,
            end_date,
            market_cache,
        )
        result[f"return_{horizon}"] = gross
        result[f"net_return_{horizon}"] = gross - roundtrip_cost
        result[f"signal_close_return_{horizon}"] = (
            end_close / reference_close - 1.0 if reference_close and reference_close > 0 else pd.NA
        )
        result[f"market_return_{horizon}"] = market_return if market_return is not None else pd.NA
        result[f"excess_return_{horizon}"] = (
            gross - market_return if market_return is not None else pd.NA
        )
    return result


def refresh_outcomes(
    events: pd.DataFrame,
    existing: pd.DataFrame,
    history: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Refresh forward outcomes without losing an older resolved result after retention pruning."""

    if events.empty:
        return _with_columns(existing, OUTCOME_COLUMNS) if not existing.empty else _blank(OUTCOME_COLUMNS)
    market_history = history.copy()
    if market_history.empty:
        market_history = pd.DataFrame(
            columns=["date", "symbol", "market", "open", "high", "low", "close"]
        )
    else:
        market_history["date"] = pd.to_datetime(market_history["date"], errors="coerce").dt.normalize()
        market_history["symbol"] = market_history["symbol"].astype("string").str.zfill(6)
        for column in ("open", "high", "low", "close"):
            market_history[column] = pd.to_numeric(market_history.get(column), errors="coerce")
        market_history = market_history.dropna(subset=["date", "symbol"])

    existing_map = {
        _text(row.get("event_id")): row
        for row in _with_columns(existing, OUTCOME_COLUMNS).to_dict("records")
    } if not existing.empty else {}
    histories_by_symbol = {
        str(symbol).zfill(6): group
        for symbol, group in market_history.groupby("symbol", sort=False)
    }
    empty_symbol_history = market_history.iloc[0:0]
    cache: dict[tuple[str, str, str], float | None] = {}
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        evaluated = _evaluate_event(
            event,
            histories_by_symbol.get(_text(event.get("symbol")).zfill(6), empty_symbol_history),
            market_history,
            settings.model.backtest_target_pct / 100.0,
            settings.model.backtest_stop_pct / 100.0,
            settings.model.backtest_roundtrip_cost_pct / 100.0,
            cache,
        )
        old = existing_map.get(evaluated["event_id"])
        old_sessions = int(_number(old.get("observed_sessions")) or 0) if old else 0
        new_sessions = int(_number(evaluated.get("observed_sessions")) or 0)
        rows.append(old if old and old_sessions > new_sessions else evaluated)
    result = _with_columns(pd.DataFrame(rows), OUTCOME_COLUMNS)
    return result.drop_duplicates("event_id", keep="last").sort_values(
        ["event_date", "symbol", "event_type"]
    ).reset_index(drop=True)


def _pct(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number * 100:+.2f}%"


def _event_label(value: str) -> str:
    return {
        "candidate_entry": "후보 최초 진입",
        "buy_review_entry": "신규매수 검토 진입",
        "buy_review_promotion": "신규매수 검토 승격",
    }.get(value, value)


def render_live_evaluation(
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    generated_at: datetime,
    roundtrip_cost_pct: float,
) -> str:
    resolved_20 = int(pd.to_numeric(outcomes.get("net_return_20"), errors="coerce").notna().sum()) if not outcomes.empty else 0
    lines = [
        "# 실전 추천 누적 평가",
        "",
        f"- 갱신: {generated_at.isoformat()}",
        f"- 후보 에피소드: {len(episodes):,}개 · 평가 이벤트: {len(events):,}개 · D+20 판정: {resolved_20:,}개",
        f"- 수익률 기준: 신호 다음 거래일 시가 진입 · 왕복비용 {roundtrip_cost_pct:.1f}% · 동일 시장 중앙값 대비 초과수익",
        "- 같은 종목의 연속 노출은 한 에피소드로 묶고, 가격 대기에서 신규매수 검토로 승격될 때만 별도 이벤트를 만듭니다.",
        "",
        "## 기간별 전체 성과",
        "",
        "| 기간 | 판정 수 | 비용 후 양수 비율 | 비용 후 중앙수익률 | 시장초과 중앙값 |",
        "|---|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        values = pd.to_numeric(outcomes.get(f"net_return_{horizon}"), errors="coerce") if not outcomes.empty else pd.Series(dtype=float)
        excess = pd.to_numeric(outcomes.get(f"excess_return_{horizon}"), errors="coerce") if not outcomes.empty else pd.Series(dtype=float)
        valid = values.dropna()
        lines.append(
            f"| D+{horizon} | {len(valid):,} | "
            f"{(valid.gt(0).mean() * 100):.1f}% | {_pct(valid.median())} | {_pct(excess.dropna().median())} |"
            if not valid.empty
            else f"| D+{horizon} | 0 | - | - | - |"
        )

    lines.extend(["", "## 등급·이벤트별 D+20", ""])
    d20 = pd.to_numeric(outcomes.get("net_return_20"), errors="coerce") if not outcomes.empty else pd.Series(dtype=float)
    if outcomes.empty or d20.notna().sum() == 0:
        lines.append("아직 D+20 판정 표본이 없습니다. 실제 신호를 소급 생성하지 않고 배포 이후부터 축적합니다.")
    else:
        working = outcomes.copy()
        working["net_return_20"] = pd.to_numeric(working["net_return_20"], errors="coerce")
        working["excess_return_20"] = pd.to_numeric(working["excess_return_20"], errors="coerce")
        lines.extend([
            "| 발견등급 | 이벤트 | 표본 | 양수 비율 | 비용 후 중앙수익률 | 시장초과 중앙값 |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for (grade, event_type), group in working.dropna(subset=["net_return_20"]).groupby(
            ["discovery_grade", "event_type"], dropna=False
        ):
            values = group["net_return_20"]
            lines.append(
                f"| {_text(grade)} | {_event_label(_text(event_type))} | {len(group):,} | "
                f"{values.gt(0).mean() * 100:.1f}% | {_pct(values.median())} | "
                f"{_pct(group['excess_return_20'].median())} |"
            )

    lines.extend(["", "## 최근 평가 이벤트", ""])
    if outcomes.empty:
        lines.append("아직 기록된 이벤트가 없습니다.")
    else:
        lines.extend([
            "| 신호일 | 종목 | 이벤트 | 등급 | D+5 | D+10 | D+20 | D+20 초과 | 상태 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for _, row in outcomes.sort_values(["event_date", "event_id"], ascending=[False, False]).head(20).iterrows():
            lines.append(
                f"| {_text(row.get('event_date'))} | {_text(row.get('name'))}({_text(row.get('symbol'))}) | "
                f"{_event_label(_text(row.get('event_type')))} | {_text(row.get('discovery_grade'))} | "
                f"{_pct(row.get('net_return_5'))} | {_pct(row.get('net_return_10'))} | "
                f"{_pct(row.get('net_return_20'))} | {_pct(row.get('excess_return_20'))} | "
                f"{_text(row.get('status'))} |"
            )
    lines.extend([
        "",
        "> 이 파일은 모델을 자동 변경하지 않습니다. 충분한 실전 표본이 쌓인 뒤 등급·게이트 조정의 근거로만 사용합니다.",
        "",
    ])
    return "\n".join(lines)


def update_live_tracking(
    scored: pd.DataFrame,
    history: pd.DataFrame,
    settings: Settings,
    generated_at: datetime,
) -> LiveTrackingResult | None:
    if not settings.tracking.enabled:
        return None

    report_dates = pd.to_datetime(scored.get("as_of"), errors="coerce").dropna()
    if report_dates.empty:
        return None
    current_date = report_dates.max().strftime("%Y-%m-%d")
    config_hash = tracking_config_hash(settings)
    incoming = build_daily_snapshots(
        scored,
        generated_at,
        settings.tracking.model_version,
        config_hash,
    )
    runs_path = settings.data_dir / "signal_runs.csv"
    snapshots_path = settings.data_dir / "signal_snapshots.csv"
    episodes_path = settings.data_dir / "signal_episodes.csv"
    events_path = settings.data_dir / "signal_events.csv"
    outcomes_path = settings.data_dir / "signal_outcomes.csv"
    report_path = settings.reports_dir / "live_evaluation.md"

    runs = read_csv(runs_path)
    runs = _with_columns(runs, RUN_COLUMNS) if not runs.empty else _blank(RUN_COLUMNS)
    recorded_dates = set(runs["signal_date"].astype(str))
    if current_date not in recorded_dates:
        runs = pd.concat(
            [
                runs,
                pd.DataFrame(
                    [
                        {
                            "signal_date": current_date,
                            "captured_at": generated_at.isoformat(),
                            "candidate_count": len(incoming),
                            "model_version": settings.tracking.model_version,
                            "config_hash": config_hash,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )[RUN_COLUMNS]
        snapshots = append_official_snapshots(read_csv(snapshots_path), incoming)
    else:
        snapshots = _with_columns(read_csv(snapshots_path), SNAPSHOT_COLUMNS)
    episodes, events = update_episode_ledger(
        snapshots,
        read_csv(episodes_path),
        read_csv(events_path),
        current_date,
        _trading_dates(history),
        settings.tracking.episode_gap_sessions,
    )
    outcomes = refresh_outcomes(events, read_csv(outcomes_path), history, settings)
    report = render_live_evaluation(
        episodes,
        events,
        outcomes,
        generated_at,
        settings.model.backtest_roundtrip_cost_pct,
    )

    write_csv(runs.sort_values("signal_date").reset_index(drop=True), runs_path)
    write_csv(snapshots, snapshots_path)
    write_csv(episodes, episodes_path)
    write_csv(events, events_path)
    write_csv(outcomes, outcomes_path)
    report_path.write_text(report, encoding="utf-8")

    resolved_20 = int(pd.to_numeric(outcomes.get("net_return_20"), errors="coerce").notna().sum())
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## 실전 추천 누적 평가\n\n"
                f"- 에피소드 {len(episodes):,}개 · 평가 이벤트 {len(events):,}개 · D+20 판정 {resolved_20:,}개\n"
            )
    return LiveTrackingResult(
        snapshot_rows=len(snapshots),
        episodes=len(episodes),
        events=len(events),
        resolved_20=resolved_20,
        report_path=report_path,
    )
