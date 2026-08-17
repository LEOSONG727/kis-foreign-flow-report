from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
    realized_return: float
    max_upside: float
    max_drawdown: float
    ambiguous_same_day: bool = False


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
        realized_return=(
            target if outcome == "target"
            else -stop if outcome == "stop"
            else float(returns.iloc[-1]) if not returns.empty else 0.0
        ),
        max_upside=float(returns.max()) if not returns.empty else 0.0,
        max_drawdown=float(returns.min()) if not returns.empty else 0.0,
    )


def evaluate_forward_ohlc(
    entry_price: float,
    future: pd.DataFrame,
    target: float,
    stop: float,
    roundtrip_cost: float = 0.0,
) -> PathResult:
    """Evaluate an executable next-open entry using daily high/low conservatively."""
    path = future.copy()
    for column in ("high", "low", "close"):
        path[column] = pd.to_numeric(path[column], errors="coerce")
    path = path.dropna(subset=["high", "low", "close"])
    if entry_price <= 0 or path.empty:
        return PathResult("censored", None, 0.0, 0.0, 0.0, 0.0)

    outcome = "censored"
    days_to_event = None
    ambiguous = False
    for day, (_, candle) in enumerate(path.iterrows(), 1):
        hit_target = float(candle["high"]) / entry_price - 1.0 >= target
        hit_stop = float(candle["low"]) / entry_price - 1.0 <= -stop
        if hit_target and hit_stop:
            # Daily bars do not reveal which level traded first. Treating the
            # ambiguous case as a stop avoids overstating strategy performance.
            outcome, days_to_event, ambiguous = "stop", day, True
            break
        if hit_stop:
            outcome, days_to_event = "stop", day
            break
        if hit_target:
            outcome, days_to_event = "target", day
            break

    close_returns = path["close"] / entry_price - 1.0 - roundtrip_cost
    high_returns = path["high"] / entry_price - 1.0 - roundtrip_cost
    low_returns = path["low"] / entry_price - 1.0 - roundtrip_cost
    return PathResult(
        outcome=outcome,
        days_to_event=days_to_event,
        forward_return=float(close_returns.iloc[-1]),
        realized_return=(
            target - roundtrip_cost if outcome == "target"
            else -stop - roundtrip_cost if outcome == "stop"
            else float(close_returns.iloc[-1])
        ),
        max_upside=float(high_returns.max()),
        max_drawdown=float(low_returns.min()),
        ambiguous_same_day=ambiguous,
    )


def build_calibration(
    results: pd.DataFrame,
    target: float = 0.08,
    stop: float = 0.06,
    roundtrip_cost: float = 0.003,
) -> dict:
    """Activate probabilities only after economic and temporal holdout checks pass."""
    bands = [(0.0, 64.0, "<64"), (64.0, 68.0, "64-67"), (68.0, 72.0, "68-71"), (72.0, 101.0, "72+")]
    resolved = results[results["outcome"].isin(["target", "stop"])].copy() if not results.empty else pd.DataFrame()
    score_column = "entry_score" if "entry_score" in resolved else "score"
    summaries = []
    for low, high, label in bands:
        group = resolved[(resolved[score_column] >= low) & (resolved[score_column] < high)] if not resolved.empty else pd.DataFrame()
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
    breakeven_win_rate = (stop + roundtrip_cost) / (target + stop)
    wins = int((resolved["outcome"] == "target").sum()) if not resolved.empty else 0
    overall_win_rate = wins / len(resolved) if len(resolved) else 0.0
    if "realized_return" in results:
        overall_mean_return = float(pd.to_numeric(results["realized_return"], errors="coerce").mean())
    else:
        overall_mean_return = float("nan")

    holdout = pd.DataFrame()
    if not results.empty and "as_of" in results:
        ordered_dates = sorted(pd.to_datetime(results["as_of"], errors="coerce").dropna().unique())
        if ordered_dates:
            holdout_start = pd.Timestamp(ordered_dates[max(0, int(len(ordered_dates) * 2 / 3))])
            holdout = results[pd.to_datetime(results["as_of"], errors="coerce") >= holdout_start].copy()
    holdout_resolved = holdout[holdout["outcome"].isin(["target", "stop"])] if not holdout.empty else pd.DataFrame()
    holdout_wins = int((holdout_resolved["outcome"] == "target").sum()) if not holdout_resolved.empty else 0
    holdout_win_rate = holdout_wins / len(holdout_resolved) if len(holdout_resolved) else 0.0
    holdout_mean_return = (
        float(pd.to_numeric(holdout["realized_return"], errors="coerce").mean())
        if not holdout.empty and "realized_return" in holdout else float("nan")
    )
    sampled_rates = [item["win_rate"] for item in summaries if item["samples"] >= 30]
    roughly_monotonic = all(
        later + 0.03 >= earlier for earlier, later in zip(sampled_rates, sampled_rates[1:])
    )
    checks = {
        "resolved_samples": len(resolved) >= 300,
        "holdout_samples": len(holdout_resolved) >= 75,
        "overall_positive_expectancy": overall_mean_return > 0,
        "holdout_positive_expectancy": holdout_mean_return > 0,
        "overall_above_breakeven": overall_win_rate >= breakeven_win_rate,
        "holdout_above_breakeven": holdout_win_rate >= breakeven_win_rate,
        "score_ordering": roughly_monotonic and len(sampled_rates) >= 2,
    }
    active = all(checks.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quality_gate_version": 2,
        "active": active,
        "resolved_samples": len(resolved),
        "minimum_resolved_samples": 300,
        "holdout_resolved_samples": len(holdout_resolved),
        "minimum_holdout_samples": 75,
        "breakeven_win_rate": breakeven_win_rate,
        "overall_win_rate": overall_win_rate,
        "holdout_win_rate": holdout_win_rate,
        "overall_mean_realized_return": overall_mean_return,
        "holdout_mean_realized_return": holdout_mean_return,
        "activation_checks": checks,
        "bands": summaries,
    }


def summarize_variants(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in results.groupby("variant", sort=False):
        resolved = group[group["outcome"].isin(["target", "stop"])]
        wins = int((group["outcome"] == "target").sum())
        stops = int((group["outcome"] == "stop").sum())
        rows.append(
            {
                "variant": variant,
                "production": bool(group["production"].iloc[0]),
                "entry_min_return3_pct": float(group["entry_min_return3_pct"].iloc[0]),
                "entry_max_return20_pct": float(group["entry_max_return20_pct"].iloc[0]),
                "entry_max_rebound20_pct": float(group["entry_max_rebound20_pct"].iloc[0]),
                "requires_reacceleration": bool(group["requires_reacceleration"].iloc[0]),
                "signals": len(group),
                "resolved": len(resolved),
                "targets": wins,
                "stops": stops,
                "resolved_win_rate": wins / len(resolved) if len(resolved) else np.nan,
                "target_rate_all": wins / len(group),
                "positive_20d_rate": float((group["forward_return"] > 0).mean()),
                "mean_20d_return": float(group["forward_return"].mean()),
                "median_20d_return": float(group["forward_return"].median()),
                "mean_strategy_return": float(group["realized_return"].mean()),
                "median_strategy_return": float(group["realized_return"].median()),
                "mean_max_upside": float(group["max_upside"].mean()),
                "mean_max_drawdown": float(group["max_drawdown"].mean()),
                "ambiguous_days": int(group["ambiguous_same_day"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def _write_backtest_report(
    settings: Settings,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    calibration: dict,
) -> None:
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
    comparison = []
    for _, item in summary.iterrows():
        gate = (
            f"3일 {item['entry_min_return3_pct']:+.0f}% / 20일 +{item['entry_max_return20_pct']:.0f}% / "
            f"저점 +{item['entry_max_rebound20_pct']:.0f}%"
        )
        comparison.append(
            f"| {item['variant']}{' (현재)' if item['production'] else ''} | {gate} | {'예' if item['requires_reacceleration'] else '아니오'} | "
            f"{int(item['signals'])} | {int(item['resolved'])} | {item['resolved_win_rate'] * 100:.1f}% | "
            f"{item['positive_20d_rate'] * 100:.1f}% | {item['mean_strategy_return'] * 100:+.2f}% | "
            f"{item['mean_20d_return'] * 100:+.2f}% | "
            f"{item['mean_max_drawdown'] * 100:+.2f}% |"
        )
    text = (
        "# 워크포워드 백테스트\n\n"
        "- 체결 가정: 신호 다음 거래일 시가 진입\n"
        f"- 평가: 이후 {settings.model.backtest_horizon_days}거래일, 목표 +{settings.model.backtest_target_pct:.0f}% / 손절 -{settings.model.backtest_stop_pct:.0f}%\n"
        f"- 왕복 비용: {settings.model.backtest_roundtrip_cost_pct:.1f}% 차감\n"
        "- 같은 날 목표·손절 동시 도달: 보수적으로 손절 처리\n"
        f"- 전체 임계값별 신호: {len(results):,}건\n"
        f"- 목표/손절 판정 완료: {calibration['resolved_samples']:,}건\n"
        f"- 손익분기 승률: {calibration['breakeven_win_rate'] * 100:.1f}%\n"
        f"- 홀드아웃 승률/실현수익: {calibration['holdout_win_rate'] * 100:.1f}% / {calibration['holdout_mean_realized_return'] * 100:+.2f}%\n"
        f"- 확률 보정 활성화: {'예' if calibration['active'] else '아니오'}\n\n"
        "## V1 진입정책 비교\n\n"
        "| 변형 | 3일 하한/20일 상한/저점반등 상한 | 재가속 필수 | 신호 | 판정 | 목표 선도달 | 20일 수익 양수 | 평균 전략수익 | 평균 20일 수익 | 평균 최대낙폭 |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(comparison)
        + "\n\n## 현재 모델 점수대 확인\n\n"
        "| 점수대 | 판정표본 | 목표 선도달 | 승률 |\n|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n현재 유니버스와 재무 마스터를 사용하므로 생존편향·재무시점 편향이 남아 있습니다. "
        "따라서 절대 승률보다 임계값 간 상대 비교를 우선 사용해야 합니다.\n"
    )
    report_path.write_text(text, encoding="utf-8")


def _policy_variants(settings: Settings) -> list[tuple[str, float, float, float, bool, bool]]:
    model = settings.model
    return [
        ("legacy_v0", -99.0, 99.0, 99.0, False, False),
        (
            "price_gate_only",
            model.entry_min_return3_pct,
            model.entry_max_return20_pct,
            model.entry_max_rebound20_pct,
            False,
            False,
        ),
        ("v1_relaxed", -3.0, 6.0, 8.0, True, False),
        (
            "v1_balanced",
            model.entry_min_return3_pct,
            model.entry_max_return20_pct,
            model.entry_max_rebound20_pct,
            model.persistent_entry_requires_reacceleration,
            True,
        ),
    ]


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
        last_signal_index: dict[tuple[str, str], int] = {}
        variants = _policy_variants(settings)
        if len(dates) >= 60 + horizon:
            for date_index in range(59, len(dates) - horizon):
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
                for variant, min3, max20, max_rebound20, require_reacceleration, production in variants:
                    variant_model = replace(
                        settings.model,
                        entry_min_return3_pct=min3,
                        entry_max_return20_pct=max20,
                        entry_max_rebound20_pct=max_rebound20,
                        persistent_entry_requires_reacceleration=require_reacceleration,
                    )
                    scored = score_features(features, variant_model)
                    candidates = scored[
                        scored["action"].isin(["강한 신규매수", "초기 분할매수"])
                        & (scored["entry_score"] >= variant_model.entry_buy_score)
                        & (scored["data_points"] >= 60)
                    ]
                    for _, row in candidates.iterrows():
                        symbol = str(row["symbol"]).zfill(6)
                        signal_key = (variant, symbol)
                        if date_index - last_signal_index.get(signal_key, -horizon) < horizon:
                            continue
                        future = history[
                            (history["symbol"] == symbol) & (history["date"] > as_of)
                        ].sort_values("date").head(horizon)
                        if len(future) < horizon:
                            continue
                        next_open = pd.to_numeric(future["open"].iloc[0], errors="coerce")
                        entry_price = float(next_open) if pd.notna(next_open) and next_open > 0 else float(future["close"].iloc[0])
                        path = evaluate_forward_ohlc(
                            entry_price,
                            future,
                            variant_model.backtest_target_pct / 100.0,
                            variant_model.backtest_stop_pct / 100.0,
                            variant_model.backtest_roundtrip_cost_pct / 100.0,
                        )
                        signal_rows.append(
                            {
                                "variant": variant,
                                "production": production,
                                "entry_min_return3_pct": min3,
                                "entry_max_return20_pct": max20,
                                "entry_max_rebound20_pct": max_rebound20,
                                "requires_reacceleration": require_reacceleration,
                                "as_of": as_of,
                                "entry_date": future["date"].iloc[0],
                                "symbol": symbol,
                                "name": row.get("name", symbol),
                                "score": row["score"],
                                "entry_score": row["entry_score"],
                                "campaign_score": row["campaign_score"],
                                "stage": row["stage"],
                                "campaign_regime": row["campaign_regime"],
                                "action": row["action"],
                                "signal_close": row["close"],
                                "entry_price": entry_price,
                                "price_return3": row.get("price_return3", 0.0),
                                "price_return10": row.get("price_return10", 0.0),
                                "price_return20": row.get("price_return20", 0.0),
                                "rebound_from_low20": row.get("rebound_from_low20", 0.0),
                                "foreign_net20": row.get("foreign_net20", 0.0),
                                "outcome": path.outcome,
                                "days_to_event": path.days_to_event,
                                "forward_return": path.forward_return,
                                "realized_return": path.realized_return,
                                "max_upside": path.max_upside,
                                "max_drawdown": path.max_drawdown,
                                "ambiguous_same_day": path.ambiguous_same_day,
                            }
                        )
                        last_signal_index[signal_key] = date_index
        results = pd.DataFrame(signal_rows)

    production_results = results[results["production"]].copy() if not results.empty else results
    calibration = build_calibration(
        production_results,
        settings.model.backtest_target_pct / 100.0,
        settings.model.backtest_stop_pct / 100.0,
        settings.model.backtest_roundtrip_cost_pct / 100.0,
    )
    summary = summarize_variants(results)
    write_csv(results, settings.reports_dir / "backtest_latest.csv")
    write_csv(summary, settings.reports_dir / "backtest_thresholds.csv")
    (settings.data_dir / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_backtest_report(settings, results, summary, calibration)
    return results
