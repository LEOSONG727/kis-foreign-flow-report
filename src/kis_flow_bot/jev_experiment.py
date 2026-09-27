"""Versioned, prospective comparisons on identical eligible shortlists."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .jev_shadow import OUTPUTS, select_candidates
from .storage import read_csv, write_csv

VERSION = "jev-challenge-v2"
COLUMNS = [
    "signal_date", "symbol", "name", "experiment_version", "cohort_id", "cohort_size",
    "baseline_rank", "ml_rank", "jev_rank", "signal_close", "entry_date", "entry_open",
    "return_5", "return_20", "net_return_5", "net_return_20", "worst_entry_loss_20",
    "roundtrip_cost_pct", "late_capture", *OUTPUTS,
]


def _new_rows(scored, cost=.3):
    if scored.empty or "jev_status" not in scored:
        return pd.DataFrame(columns=COLUMNS)
    frame = select_candidates(scored).copy()
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    frame["signal_date"] = pd.to_datetime(frame["as_of"]).dt.strftime("%Y-%m-%d")
    if frame["signal_date"].nunique() != 1:
        return pd.DataFrame(columns=COLUMNS)
    frame["baseline_rank"] = range(1, len(frame) + 1)
    frame["ml_rank"] = pd.to_numeric(frame.get("ml_shadow_rank", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    frame["jev_rank"] = float("nan")
    complete = frame["jev_status"].eq("OK").all()
    if complete:
        risk = frame[["jev_fragile_flow", "jev_price_trap", "jev_chase_risk"]].astype(float).max(axis=1)
        # Worst failure mode cannot be hidden by averaging two benign answers.
        ranked = frame.assign(_risk=risk).sort_values(["_risk", "baseline_rank"], kind="stable")
        frame["jev_rank"] = pd.Series(range(1, len(frame) + 1), index=ranked.index)
    frame["experiment_version"] = VERSION
    identity = "|".join([VERSION, str(cost), *sorted(frame.get("jev_policy", pd.Series(dtype=str)).dropna().unique()),
                         *sorted(frame.get("jev_prompt_hash", pd.Series(dtype=str)).dropna().unique()),
                         *sorted(frame.get("jev_model", pd.Series(dtype=str)).dropna().unique())])
    frame["cohort_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
    frame["cohort_size"] = len(frame)
    frame["roundtrip_cost_pct"] = cost
    frame["signal_close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame["late_capture"] = False
    for column in COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[COLUMNS]


def update_jev_experiment(scored, history, path, *, roundtrip_cost_pct=.3):
    incoming = _new_rows(scored, roundtrip_cost_pct)
    existing = read_csv(path)
    if existing.empty:
        ledger = incoming
    else:
        # Freeze whole daily cohorts. A rerun cannot add replacement candidates.
        keys = ["signal_date", "experiment_version"]
        known = pd.MultiIndex.from_frame(existing[keys].astype(str))
        novel = ~pd.MultiIndex.from_frame(incoming[keys].astype(str)).isin(known)
        ledger = pd.concat([existing, incoming.loc[novel]], ignore_index=True) if novel.any() else existing.copy()
    if ledger.empty:
        return ledger
    for column in COLUMNS:
        if column not in ledger:
            ledger[column] = pd.NA
    if not history.empty:
        prices = history.copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["symbol"] = prices["symbol"].astype("string").str.zfill(6)
        prices = prices.dropna(subset=["date"]).drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
        groups = {symbol: rows for symbol, rows in prices.groupby("symbol")}
        for index, row in ledger[ledger["return_20"].isna() & ledger["experiment_version"].eq(VERSION)].iterrows():
            symbol = str(row["symbol"]).zfill(6)
            if symbol not in groups:
                continue
            series = groups[symbol]
            future = series[series["date"] > pd.Timestamp(row["signal_date"])].head(20)
            if future.empty:
                continue
            captured = pd.to_datetime(row["jev_captured_at"], utc=True, errors="coerce")
            entry_time = (future["date"].iloc[0].normalize() + pd.Timedelta(hours=9)).tz_localize("Asia/Seoul")
            late = pd.isna(captured) or captured >= entry_time
            ledger.at[index, "late_capture"] = bool(late)
            if late:
                continue
            opening = pd.to_numeric(future["open"].iloc[0], errors="coerce")
            if pd.isna(opening) or opening <= 0:
                continue
            ledger.at[index, "entry_date"] = future["date"].iloc[0].strftime("%Y-%m-%d")
            ledger.at[index, "entry_open"] = opening
            cost = float(row["roundtrip_cost_pct"]) / 100
            for horizon in (5, 20):
                if len(future) >= horizon:
                    close = pd.to_numeric(future["close"].iloc[horizon - 1], errors="coerce")
                    if pd.notna(close) and close > 0:
                        value = close / opening - 1
                        ledger.at[index, f"return_{horizon}"] = value
                        ledger.at[index, f"net_return_{horizon}"] = value - cost
            if len(future) >= 20:
                lows = pd.to_numeric(future["low"], errors="coerce")
                if lows.notna().all() and lows.gt(0).all():
                    ledger.at[index, "worst_entry_loss_20"] = min(0., (lows / opening - 1).min())
    ledger["symbol"] = ledger["symbol"].astype("string").str.zfill(6)
    ledger = ledger.sort_values(["signal_date", "baseline_rank", "symbol"]).reset_index(drop=True)
    write_csv(ledger[COLUMNS], path)
    return ledger[COLUMNS]


def write_jev_comparison(ledger, path, top_n=3):
    lines = ["# Jev 비교 · 탐색적 실험", "", "동일 후보군·동일 신호일만 비교. 다음 거래일 시가 진입, 20번째 거래일 종가 청산.",
             "수익률은 기록된 왕복 비용 차감. 진입가 대비 최악 손실은 포트폴리오 MDD가 아닙니다.",
             "서로 겹치는 보유 기간은 독립 표본이 아니며 자동 승격하지 않습니다.", ""]
    if ledger.empty:
        lines.append("아직 기록 없음")
    else:
        current = ledger[ledger["experiment_version"].eq(VERSION)]
        for cohort, group in current.groupby("cohort_id"):
            all_days = group["signal_date"].nunique()
            paired = []
            for day, rows in group.groupby("signal_date"):
                valid = (len(rows) == int(rows["cohort_size"].iloc[0]) and rows["symbol"].nunique() == len(rows)
                         and rows["jev_status"].eq("OK").all() and rows["net_return_20"].notna().all()
                         and rows["worst_entry_loss_20"].notna().all())
                if not valid:
                    continue
                portfolios = {}
                for label, column in (("기존 발견", "baseline_rank"), ("ML", "ml_rank"), ("Jev", "jev_rank")):
                    top = rows[pd.to_numeric(rows[column], errors="coerce").between(1, top_n)]
                    if len(top) != top_n:
                        break
                    portfolios[label] = (pd.to_numeric(top["net_return_20"]).mean(), pd.to_numeric(top["worst_entry_loss_20"]).mean())
                if len(portfolios) == 3:
                    paired.append(portfolios)
            lines.extend([f"## 실험 {cohort}", f"전체 {all_days}일 / 세 방식 모두 평가 가능한 공통 날짜 {len(paired)}일", ""])
            if len(paired) < 20:
                lines.append("표본 축적 중. 공통 날짜 20개 이상부터 탐색적 수치 표시.")
                continue
            lines.extend(["| 방식 | 비용 후 20일 수익 | 기존 대비 차이 | 진입가 대비 최악 손실 |", "|---|---:|---:|---:|"])
            baseline = sum(day["기존 발견"][0] for day in paired) / len(paired)
            for label in ("기존 발견", "ML", "Jev"):
                ret = sum(day[label][0] for day in paired) / len(paired)
                loss = sum(day[label][1] for day in paired) / len(paired)
                lines.append(f"| {label} | {ret:+.2%} | {ret-baseline:+.2%}p | {loss:.2%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
