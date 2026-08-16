from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .features import add_cross_sectional_residuals, build_base_features
from .scoring import score_features
from .storage import read_csv, write_csv


@dataclass(frozen=True)
class PathResult:
    outcome: str
    days_to_event: int | None
    forward_return: float
    max_upside: float
    max_drawdown: float


def evaluate_forward_path(
    entry_price: float,
    future_prices: pd.Series,
    target: float,
    stop: float,
) -> PathResult:
    returns = pd.to_numeric(future_prices, errors="coerce").dropna() / entry_price - 1.0
    outcome = "censored"
    days_to_event = None
    for day, value in enumerate(returns.tolist(), 1):
        if value >= target:
            outcome, days_to_event = "target", day
            break
        if value <= -stop:
            outcome, days_to_event = "stop", day
            break
    return PathResult(
        outcome=outcome,
        days_to_event=days_to_event,
        forward_return=float(returns.iloc[-1]) if not returns.empty else 0.0,
        max_upside=float(returns.max()) if not returns.empty else 0.0,
        max_drawdown=float(returns.min()) if not returns.empty else 0.0,
    )


def build_calibration(results: pd.DataFrame) -> dict:
    bands = [(0.0, 64.0, "<64"), (64.0, 72.0, "64-71"), (72.0, 80.0, "72-79"), (80.0, 101.0, "80+")]
    resolved = results[results["outcome"].isin(["target", "stop"])].copy() if not results.empty else pd.DataFrame()
    summaries = []
    for low, high, label in bands:
        group = resolved[(resolved["score"] >= low) & (resolved["score"] < high)] if not resolved.empty else pd.DataFrame()
        wins = int((group["outcome"] == "target").sum()) if not group.empty else 0
        count = len(group)
        summaries.append(
            {
                "label": label,
                "low": low,
                "high": high,
                "samples": count,
                "wins": wins,
                "win_rate": wins / count if count else None,
                "calibrated_probability": (wins + 2) / (count + 4) if count else None,
            }
        )
    actionable_samples = sum(item["samples"] for item in summaries if item["low"] >= 72.0)
    active = len(resolved) >= 300 and actionable_samples >= 100
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "active": active,
        "resolved_samples": len(resolved),
        "minimum_resolved_samples": 300,
        "minimum_actionable_samples": 100,
        "bands": summaries,
    }


def _write_backtest_report(settings: Settings, results: pd.DataFrame, calibration: dict) -> None:
    report_path = settings.reports_dir / "backtest_latest.md"
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        report_path.write_text(
            "# 워크포워드 백테스트\n\n장기 이력과 미래 평가 구간이 아직 부족합니다. 전체시장 장기 이력 초기화 후 다시 실행하세요.\n",
            encoding="utf-8",
        )
        return
    rows = []
    for item in calibration["bands"]:
        rate = "-" if item["win_rate"] is None else f"{item['win_rate'] * 100:.1f}%"
        rows.append(f"| {item['label']} | {item['samples']} | {item['wins']} | {rate} |")
    text = (
        "# 워크포워드 백테스트\n\n"
        f"- 캠페인 신호: {len(results):,}건\n"
        f"- 목표/손절 판정 완료: {calibration['resolved_samples']:,}건\n"
        f"- 확률 보정 활성화: {'예' if calibration['active'] else '아니오'}\n\n"
        "| 점수대 | 판정표본 | 목표 선도달 | 승률 |\n|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n현재 유니버스와 재무 마스터를 사용하는 한계가 있어 생존편향·재무시점 편향을 별도로 유의해야 합니다.\n"
    )
    report_path.write_text(text, encoding="utf-8")


def run_backtest(settings: Settings) -> pd.DataFrame:
    history = read_csv(settings.data_dir / "history.csv")
    universe = read_csv(settings.data_dir / "universe.csv")
    snapshots = read_csv(settings.data_dir / "holdings.csv")
    opinions = read_csv(settings.data_dir / "opinions.csv")
    horizon = settings.model.backtest_horizon_days
    if history.empty or universe.empty:
        results = pd.DataFrame()
    else:
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history["symbol"] = history["symbol"].astype("string").str.zfill(6)
        dates = sorted(history["date"].dropna().unique())
        signal_rows: list[dict] = []
        last_signal_index: dict[str, int] = {}
        if len(dates) >= 60 + horizon:
            for date_index in range(59, len(dates) - horizon, 5):
                as_of = pd.Timestamp(dates[date_index])
                history_view = history[history["date"] <= as_of]
                snapshot_view = snapshots
                if not snapshots.empty and "date" in snapshots:
                    snapshot_view = snapshots[pd.to_datetime(snapshots["date"], errors="coerce") <= as_of]
                opinion_view = opinions
                if not opinions.empty and "date" in opinions:
                    opinion_view = opinions[pd.to_datetime(opinions["date"], errors="coerce") <= as_of]
                features = build_base_features(history_view, universe, snapshot_view, opinion_view)
                features = add_cross_sectional_residuals(features)
                scored = score_features(features, settings.model)
                candidates = scored[
                    (scored["score"] >= settings.model.watch_score)
                    & scored["stage"].isin(["매집·눌림", "초기매집", "매집휴식"])
                ]
                for _, row in candidates.iterrows():
                    symbol = str(row["symbol"]).zfill(6)
                    if date_index - last_signal_index.get(symbol, -horizon) < horizon:
                        continue
                    future = history[(history["symbol"] == symbol) & (history["date"] > as_of)].sort_values("date").head(horizon)
                    if len(future) < horizon:
                        continue
                    path = evaluate_forward_path(
                        float(row["close"]),
                        future["close"],
                        settings.model.backtest_target_pct / 100.0,
                        settings.model.backtest_stop_pct / 100.0,
                    )
                    signal_rows.append(
                        {
                            "as_of": as_of,
                            "symbol": symbol,
                            "name": row.get("name", symbol),
                            "score": row["score"],
                            "stage": row["stage"],
                            "entry_price": row["close"],
                            "outcome": path.outcome,
                            "days_to_event": path.days_to_event,
                            "forward_return": path.forward_return,
                            "max_upside": path.max_upside,
                            "max_drawdown": path.max_drawdown,
                        }
                    )
                    last_signal_index[symbol] = date_index
        results = pd.DataFrame(signal_rows)

    calibration = build_calibration(results)
    write_csv(results, settings.reports_dir / "backtest_latest.csv")
    (settings.data_dir / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_backtest_report(settings, results, calibration)
    return results
