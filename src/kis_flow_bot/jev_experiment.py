"""Prospective Jev vs quant/ML comparison on the same daily shortlist."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .storage import read_csv, write_csv

VERSION = "jev-challenge-v1"
COLUMNS = [
    "signal_date", "symbol", "name", "experiment_version", "baseline_rank",
    "ml_rank", "jev_rank", "jev_review", "jev_fragile_flow",
    "jev_price_trap", "jev_chase_risk", "signal_close", "entry_date",
    "entry_open", "return_5", "return_20", "max_drawdown_20",
]


def _new_rows(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "jev_review" not in scored:
        return pd.DataFrame(columns=COLUMNS)
    frame = scored.head(20).copy()
    if frame["jev_review"].isna().any():
        # A missing API judgment must not silently change a contender list.
        return pd.DataFrame(columns=COLUMNS)
    frame["baseline_rank"] = range(1, len(frame) + 1)
    if "ml_shadow_rank" in frame:
        frame["ml_rank"] = pd.to_numeric(frame["ml_shadow_rank"], errors="coerce")
    else:
        frame["ml_rank"] = pd.NA
    frame["_review_order"] = frame["jev_review"].map(
        {"DEEP_REVIEW": 0, "WATCH": 1, "REJECT": 2}
    )
    risk_columns = ["jev_fragile_flow", "jev_price_trap", "jev_chase_risk"]
    frame["_risk"] = frame[risk_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    ranked = frame.sort_values(["_review_order", "_risk", "baseline_rank"])
    frame["jev_rank"] = pd.Series(range(1, len(frame) + 1), index=ranked.index)
    frame["signal_date"] = pd.to_datetime(frame["as_of"]).dt.strftime("%Y-%m-%d")
    frame["experiment_version"] = VERSION
    frame["signal_close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    for col in ("entry_date", "entry_open", "return_5", "return_20", "max_drawdown_20"):
        frame[col] = pd.NA
    return frame.rename(columns={"symbol": "symbol"})[COLUMNS]


def update_jev_experiment(scored: pd.DataFrame, history: pd.DataFrame, path: Path) -> pd.DataFrame:
    incoming = _new_rows(scored)
    existing = read_csv(path)
    if existing.empty:
        ledger = incoming
    elif incoming.empty:
        ledger = existing
    else:
        key = ["signal_date", "symbol", "experiment_version"]
        known = pd.MultiIndex.from_frame(existing[key].astype(str))
        novel = ~pd.MultiIndex.from_frame(incoming[key].astype(str)).isin(known)
        ledger = pd.concat([existing, incoming.loc[novel]], ignore_index=True)
    if ledger.empty or history.empty:
        return ledger
    prices = history.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["symbol"] = prices["symbol"].astype("string").str.zfill(6)
    prices = prices.dropna(subset=["date"]).sort_values(["symbol", "date"])
    ledger["symbol"] = ledger["symbol"].astype("string").str.zfill(6)
    for index, row in ledger[ledger["return_20"].isna()].iterrows():
        day = pd.to_datetime(row["signal_date"], errors="coerce")
        future = prices[(prices["symbol"] == row["symbol"]) & (prices["date"] > day)].head(20)
        if future.empty:
            continue
        opening = pd.to_numeric(future["open"].iloc[0], errors="coerce")
        if pd.isna(opening) or opening <= 0:
            continue
        ledger.at[index, "entry_date"] = future["date"].iloc[0].strftime("%Y-%m-%d")
        ledger.at[index, "entry_open"] = opening
        for horizon in (5, 20):
            if len(future) >= horizon:
                closing = pd.to_numeric(future["close"].iloc[horizon - 1], errors="coerce")
                if pd.notna(closing):
                    ledger.at[index, f"return_{horizon}"] = closing / opening - 1
        if len(future) >= 20:
            lows = pd.to_numeric(future["low"], errors="coerce").dropna()
            if not lows.empty:
                ledger.at[index, "max_drawdown_20"] = (lows / opening - 1).min()
    ledger = ledger.sort_values(["signal_date", "baseline_rank", "symbol"]).reset_index(drop=True)
    write_csv(ledger[COLUMNS], path)
    return ledger[COLUMNS]


def write_jev_comparison(ledger: pd.DataFrame, path: Path, top_n: int = 3) -> None:
    """Summarize paired daily portfolios only after their 20-session outcome matures."""
    lines = [
        "# Jev Shadow 비교",
        "",
        "동일한 일별 후보군에서 기존 발견 순위, ML 순위, Jev 위험 진단 순위의 상위 "
        f"{top_n}개를 비교합니다. 신호 다음 거래일 시가 진입, 20번째 거래일 종가 청산이며 "
        "수수료와 세금은 포함되지 않습니다.",
        "",
    ]
    if ledger.empty:
        lines.append("아직 Jev 응답을 모두 받은 일별 후보군이 없습니다.")
    else:
        full = ledger.groupby("signal_date").filter(
            lambda day: day["return_20"].notna().all()
        )
        dates = full["signal_date"].nunique()
        lines.append(f"20거래일 결과가 확정된 신호일: {dates}일")
        lines.append("서로 가까운 신호일은 보유 기간이 겹치므로 독립 표본으로 해석하지 않습니다.")
        if dates < 20:
            lines.append("20개 신호일 이상 쌓인 뒤 비교 수치를 표시합니다.")
        else:
            lines.extend(["", "| 순위 | 일별 상위 수익률 평균 | 일별 상위 최대 하락폭 평균 |", "|---|---:|---:|"])
            for label, column in (("기존 발견", "baseline_rank"), ("ML", "ml_rank"), ("Jev 진단", "jev_rank")):
                daily = []
                for _, day in full.groupby("signal_date"):
                    top = day[pd.to_numeric(day[column], errors="coerce") <= top_n]
                    if len(top) == top_n:
                        daily.append((
                            pd.to_numeric(top["return_20"]).mean(),
                            pd.to_numeric(top["max_drawdown_20"]).mean(),
                        ))
                if daily:
                    means = pd.DataFrame(daily, columns=["return", "drawdown"]).mean()
                    lines.append(f"| {label} ({len(daily)}일) | {means['return']:+.2%} | {means['drawdown']:+.2%} |")
            lines.append("")
            lines.append("Jev가 성과를 개선했다는 결론은 별도 기간 검증과 거래비용 반영 후 판단합니다.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
