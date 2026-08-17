from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig, Settings
from .features import add_cross_sectional_residuals, build_base_features
from .scoring import score_features
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)
_GATE_SWEEP_CTX: dict = {}


@dataclass(frozen=True)
class PathResult:
    outcome: str
    days_to_event: int | None
    forward_return: float
    realized_return: float
    max_upside: float
    max_drawdown: float
    ambiguous_same_day: bool = False


@dataclass(frozen=True)
class ScaledStrategyResult:
    return_rate: float
    invested_krw: float
    proceeds_krw: float
    tranches: int
    gap_half_sold: bool
    exit_reason: str


def simulate_scaled_strategy(
    context: pd.DataFrame,
    signal_date: pd.Timestamp,
    model: ModelConfig,
) -> ScaledStrategyResult:
    """Approximate the user's executable scale-in / gap-profit strategy.

    Orders generated from a daily close are filled at the next session open.
    The only same-session action is a gap-up half sale of an existing position.
    """
    frame = context.copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "foreign_net_amount"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    future_index = frame.index[frame["date"] > pd.Timestamp(signal_date)].tolist()
    if not future_index:
        return ScaledStrategyResult(0.0, 0.0, 0.0, 0, False, "미체결")

    side_cost = model.backtest_roundtrip_cost_pct / 200.0
    trigger = model.add_trigger_pct / 100.0
    gap_threshold = model.gap_take_profit_pct / 100.0
    trailing = model.strategy_trailing_stop_pct / 100.0
    shares = 0
    invested = 0.0
    proceeds = 0.0
    tranches = 0
    last_buy = 0.0
    last_buy_day = -10_000
    pending_add = True
    pending_exit = False
    gap_half_sold = False
    trailing_high = 0.0
    exit_reason = "평가기간 종료"

    for future_day, idx in enumerate(future_index):
        row = frame.loc[idx]
        open_price = float(row["open"])
        if pending_exit and shares > 0:
            proceeds += shares * open_price * (1.0 - side_cost)
            shares = 0
            exit_reason = "수급 신호 무효"
            break
        if pending_add and tranches < model.max_tranches:
            order_krw = model.initial_order_krw if tranches == 0 else model.add_order_krw
            quantity = int(order_krw // (open_price * (1.0 + side_cost)))
            if quantity > 0:
                invested += quantity * open_price * (1.0 + side_cost)
                shares += quantity
                tranches += 1
                last_buy = open_price
                last_buy_day = future_day
            pending_add = False

        if shares <= 0:
            break

        # A gap on the initial entry is not a profit. Apply the half-sale only
        # after the position existed before today's open.
        if future_day > 0 and not gap_half_sold and idx > 0:
            previous_close = float(frame.loc[idx - 1, "close"])
            gap = open_price / previous_close - 1.0 if previous_close > 0 else 0.0
            if gap >= gap_threshold:
                sell_quantity = max(1, shares // 2)
                proceeds += sell_quantity * open_price * (1.0 - side_cost)
                shares -= sell_quantity
                gap_half_sold = True
                trailing_high = open_price
                exit_reason = "갭상승 절반 익절"

        if shares <= 0:
            break
        if gap_half_sold:
            stop_price = trailing_high * (1.0 - trailing)
            if trailing_high > 0 and float(row["low"]) <= stop_price:
                proceeds += shares * stop_price * (1.0 - side_cost)
                shares = 0
                exit_reason = "절반 익절 후 추적손절"
                break
            trailing_high = max(trailing_high, float(row["high"]))

        close_slice = frame.loc[:idx].tail(20)
        flow20 = float(close_slice["foreign_net_amount"].fillna(0.0).sum())
        positive_flow = float(close_slice.loc[close_slice["foreign_net_amount"] > 0, "foreign_net_amount"].sum())
        negative_flow = abs(float(close_slice.loc[close_slice["foreign_net_amount"] < 0, "foreign_net_amount"].sum()))
        sellback = negative_flow / positive_flow if positive_flow > 0 else 9.0
        flow3 = float(close_slice.tail(3)["foreign_net_amount"].fillna(0.0).sum())
        flow5 = float(close_slice.tail(5)["foreign_net_amount"].fillna(0.0).sum())
        base_close = float(close_slice["close"].iloc[0]) if not close_slice.empty else float(row["close"])
        price20 = float(row["close"]) / base_close - 1.0 if base_close > 0 else 0.0
        close3 = close_slice.tail(4)
        price3 = (
            float(close3["close"].iloc[-1]) / float(close3["close"].iloc[0]) - 1.0
            if len(close3) >= 2 and float(close3["close"].iloc[0]) > 0 else 0.0
        )
        prior_close = float(frame.loc[idx - 1, "close"]) if idx > 0 else float(row["close"])
        price1 = float(row["close"]) / prior_close - 1.0 if prior_close > 0 else 0.0
        signal_alive = (
            flow20 > 0
            and sellback < 1.0
            and not (flow3 <= 0 and flow5 <= 0)
            and price20 > model.structural_break_return20_pct / 100.0
        )
        if not signal_alive:
            pending_exit = True
        else:
            add_confirmed = (
                not model.strict_add_confirmation
                or (
                    flow3 > 0
                    and flow5 > 0
                    and sellback <= 0.80
                    and price3 >= model.add_min_return3_pct / 100.0
                    and price1 >= model.add_min_return1_pct / 100.0
                    and price20 >= model.add_min_return20_pct / 100.0
                    and future_day - last_buy_day >= model.add_cooldown_days
                )
            )
        if signal_alive and (
            tranches < model.max_tranches
            and last_buy > 0
            and float(row["close"]) <= last_buy * (1.0 - trigger)
            and add_confirmed
        ):
            pending_add = True

    if shares > 0:
        final_price = float(frame.loc[future_index[-1], "close"])
        proceeds += shares * final_price * (1.0 - side_cost)
    return ScaledStrategyResult(
        return_rate=(proceeds / invested - 1.0) if invested > 0 else 0.0,
        invested_krw=invested,
        proceeds_krw=proceeds,
        tranches=tranches,
        gap_half_sold=gap_half_sold,
        exit_reason=exit_reason,
    )


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


def market_forward_benchmarks(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    end_date: pd.Timestamp,
    roundtrip_cost: float = 0.0,
) -> dict[str, float]:
    """Median next-open to horizon-close return by market for an excess-return check."""
    future = history[
        (history["date"] > pd.Timestamp(as_of))
        & (history["date"] <= pd.Timestamp(end_date))
    ].sort_values(["symbol", "date"])
    if future.empty:
        return {}
    rows = []
    for symbol, group in future.groupby("symbol", sort=False):
        entry = pd.to_numeric(group["open"].iloc[0], errors="coerce")
        close = pd.to_numeric(group["close"].iloc[-1], errors="coerce")
        if pd.isna(entry) or pd.isna(close) or float(entry) <= 0:
            continue
        rows.append(
            {
                "market": str(group["market"].iloc[-1]),
                "return": float(close) / float(entry) - 1.0 - roundtrip_cost,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    return frame.groupby("market")["return"].median().astype(float).to_dict()


def build_calibration(
    results: pd.DataFrame,
    target: float = 0.08,
    stop: float = 0.06,
    roundtrip_cost: float = 0.003,
) -> dict:
    """Activate probabilities only after economic and temporal holdout checks pass."""
    bands = [(0.0, 64.0, "<64"), (64.0, 68.0, "64-67"), (68.0, 72.0, "68-71"), (72.0, 101.0, "72+")]
    resolved = results[results["outcome"].isin(["target", "stop"])].copy() if not results.empty else pd.DataFrame()
    score_column = (
        "signal_score" if "signal_score" in resolved
        else "entry_score" if "entry_score" in resolved
        else "score"
    )
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
    overall_mean_excess = (
        float(pd.to_numeric(results["excess_forward_return"], errors="coerce").mean())
        if not results.empty and "excess_forward_return" in results else float("nan")
    )
    holdout_mean_excess = (
        float(pd.to_numeric(holdout["excess_forward_return"], errors="coerce").mean())
        if not holdout.empty and "excess_forward_return" in holdout else float("nan")
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
        "overall_positive_excess_return": overall_mean_excess > 0,
        "holdout_positive_excess_return": holdout_mean_excess > 0,
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
        "overall_mean_excess_return": overall_mean_excess,
        "holdout_mean_excess_return": holdout_mean_excess,
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
                "timing_buy_score": float(group.get("timing_buy_score", pd.Series([0.0])).iloc[0]),
                "strict_add_confirmation": bool(
                    group.get("strict_add_confirmation", pd.Series([False])).iloc[0]
                ),
                "signals": len(group),
                "resolved": len(resolved),
                "targets": wins,
                "stops": stops,
                "resolved_win_rate": wins / len(resolved) if len(resolved) else np.nan,
                "target_rate_all": wins / len(group),
                "positive_20d_rate": float((group["forward_return"] > 0).mean()),
                "mean_20d_return": float(group["forward_return"].mean()),
                "mean_excess_20d_return": float(
                    pd.to_numeric(
                        group.get("excess_forward_return", pd.Series(np.nan, index=group.index)),
                        errors="coerce",
                    ).mean()
                ),
                "positive_excess_20d_rate": float(
                    (
                        pd.to_numeric(
                            group.get("excess_forward_return", pd.Series(np.nan, index=group.index)),
                            errors="coerce",
                        ) > 0
                    ).mean()
                ),
                "median_20d_return": float(group["forward_return"].median()),
                "mean_strategy_return": float(group["realized_return"].mean()),
                "median_strategy_return": float(group["realized_return"].median()),
                "mean_scaled_strategy_return": float(
                    pd.to_numeric(group.get("scaled_strategy_return", group["realized_return"]), errors="coerce").mean()
                ),
                "positive_scaled_strategy_rate": float(
                    (pd.to_numeric(group.get("scaled_strategy_return", group["realized_return"]), errors="coerce") > 0).mean()
                ),
                "mean_tranches": float(
                    pd.to_numeric(group.get("strategy_tranches", pd.Series(1.0, index=group.index)), errors="coerce").mean()
                ),
                "mean_max_upside": float(group["max_upside"].mean()),
                "mean_max_drawdown": float(group["max_drawdown"].mean()),
                "ambiguous_days": int(group["ambiguous_same_day"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def summarize_score_bands(results: pd.DataFrame) -> pd.DataFrame:
    """Show whether stronger discovery/timing scores actually improve outcomes."""
    if results.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    definitions = [
        ("discovery_v2", "discovery_score", [65.0, 70.0, 75.0, 80.0, 101.0]),
        ("flow_price_v2", "signal_score", [0.0, 60.0, 65.0, 70.0, 75.0, 80.0, 101.0]),
    ]
    for variant, score_column, edges in definitions:
        group = results[results["variant"] == variant].copy()
        if group.empty or score_column not in group:
            continue
        for low, high in zip(edges, edges[1:]):
            band = group[(group[score_column] >= low) & (group[score_column] < high)]
            if band.empty:
                continue
            scaled = pd.to_numeric(
                band.get("scaled_strategy_return", band["realized_return"]), errors="coerce"
            )
            rows.append(
                {
                    "variant": variant,
                    "score_name": score_column,
                    "band": f"{low:.0f}-{high - 1:.0f}",
                    "low": low,
                    "high": high,
                    "signals": len(band),
                    "positive_20d_rate": float((band["forward_return"] > 0).mean()),
                    "mean_20d_return": float(band["forward_return"].mean()),
                    "mean_excess_20d_return": float(
                        pd.to_numeric(
                            band.get("excess_forward_return", pd.Series(np.nan, index=band.index)),
                            errors="coerce",
                        ).mean()
                    ),
                    "mean_scaled_strategy_return": float(scaled.mean()),
                    "positive_scaled_strategy_rate": float((scaled > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_backtest_report(
    settings: Settings,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    score_bands: pd.DataFrame,
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
            f"| {item['variant']}{' (현재)' if item['production'] else ''} | {gate} | {item['timing_buy_score']:.0f} | {'예' if item['requires_reacceleration'] else '아니오'} | {'예' if item['strict_add_confirmation'] else '아니오'} | "
            f"{int(item['signals'])} | {int(item['resolved'])} | {item['resolved_win_rate'] * 100:.1f}% | "
            f"{item['positive_20d_rate'] * 100:.1f}% | {item['mean_scaled_strategy_return'] * 100:+.2f}% | "
            f"{item['positive_scaled_strategy_rate'] * 100:.1f}% | {item['mean_tranches']:.1f} | "
            f"{item['mean_20d_return'] * 100:+.2f}% | {item['mean_excess_20d_return'] * 100:+.2f}% | "
            f"{item['mean_max_drawdown'] * 100:+.2f}% |"
        )
    band_rows = []
    for _, item in score_bands.iterrows():
        score_label = "발견" if item["score_name"] == "discovery_score" else "발견+타이밍"
        band_rows.append(
            f"| {item['variant']} | {score_label} {item['band']} | {int(item['signals'])} | "
            f"{item['positive_20d_rate'] * 100:.1f}% | {item['mean_20d_return'] * 100:+.2f}% | "
            f"{item['mean_excess_20d_return'] * 100:+.2f}% | {item['mean_scaled_strategy_return'] * 100:+.2f}% |"
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
        f"- 전체/홀드아웃 시장초과수익: {calibration['overall_mean_excess_return'] * 100:+.2f}% / {calibration['holdout_mean_excess_return'] * 100:+.2f}%\n"
        f"- 확률 보정 활성화: {'예' if calibration['active'] else '아니오'}\n\n"
        "## V2 진입정책 비교\n\n"
        "| 변형 | 3일 하한/20일 상한/저점반등 상한 | 타이밍 하한 | 재가속 필수 | 추가확인 | 신호 | 판정 | 목표 선도달 | 20일 수익 양수 | 분할전략 평균 | 분할전략 양수 | 평균 차수 | 평균 20일 수익 | 시장초과 | 평균 최대낙폭 |\n"
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(comparison)
        + "\n\n## 현재 모델 점수대 확인\n\n"
        "| 점수대 | 판정표본 | 목표 선도달 | 승률 |\n|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n## 발견·진입 점수대 성과\n\n"
        "| 대상 | 점수대 | 신호 | 20일 양수 | 평균 20일 | 시장초과 | 분할전략 평균 |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
        + ("\n".join(band_rows) if band_rows else "| - | 표본 부족 | 0 | - | - | - | - |")
        + "\n\n현재 유니버스와 재무 마스터를 사용하므로 생존편향·재무시점 편향이 남아 있습니다. "
        "따라서 절대 승률보다 임계값 간 상대 비교를 우선 사용해야 합니다. 분할전략은 1천만원 최초 진입, "
        "마지막 매수가 대비 하락 시 500만원 추가, 갭 5% 이상 절반 익절 후 8% 추적손절을 근사합니다.\n"
    )
    report_path.write_text(text, encoding="utf-8")


def _policy_variants(settings: Settings) -> list[dict]:
    model = settings.model
    return [
        {
            "variant": "discovery_v2", "mode": "discovery", "min3": -99.0,
            "max20": 99.0, "max_rebound20": 99.0, "reacceleration": False,
            "discovery": model.discovery_candidate_score, "timing": 0.0,
            "strict_add": True, "production": False,
        },
        {
            "variant": "legacy_v0", "mode": "entry", "min3": -99.0,
            "max20": 99.0, "max_rebound20": 99.0, "reacceleration": False,
            "discovery": 0.0, "timing": 0.0, "strict_add": False, "production": False,
        },
        {
            "variant": "price_gate_only", "mode": "entry", "min3": model.entry_min_return3_pct,
            "max20": model.entry_max_return20_pct, "max_rebound20": model.entry_max_rebound20_pct,
            "reacceleration": False, "discovery": 0.0, "timing": 0.0,
            "strict_add": False, "production": False,
        },
        {
            "variant": "v1_relaxed", "mode": "entry", "min3": -3.0,
            "max20": 6.0, "max_rebound20": 8.0, "reacceleration": True,
            "discovery": 55.0, "timing": 55.0, "strict_add": False, "production": False,
        },
        {
            "variant": "v1_balanced", "mode": "entry", "min3": -3.0,
            "max20": 3.0, "max_rebound20": 5.0, "reacceleration": True,
            "discovery": 60.0, "timing": 62.0, "strict_add": False, "production": False,
        },
        {
            "variant": "flow_price_v2_old_add", "mode": "entry", "min3": model.entry_min_return3_pct,
            "max20": model.entry_max_return20_pct, "max_rebound20": model.entry_max_rebound20_pct,
            "reacceleration": model.persistent_entry_requires_reacceleration,
            "discovery": model.discovery_candidate_score, "timing": model.timing_buy_score,
            "strict_add": False, "production": False,
        },
        {
            "variant": "flow_price_v2", "mode": "entry", "min3": model.entry_min_return3_pct,
            "max20": model.entry_max_return20_pct, "max_rebound20": model.entry_max_rebound20_pct,
            "reacceleration": model.persistent_entry_requires_reacceleration,
            "discovery": model.discovery_candidate_score, "timing": model.timing_buy_score,
            "strict_add": True, "production": True,
        },
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
                market_benchmarks = market_forward_benchmarks(
                    history,
                    as_of,
                    pd.Timestamp(dates[date_index + horizon]),
                    settings.model.backtest_roundtrip_cost_pct / 100.0,
                )
                snapshot_view = snapshots
                if not snapshots.empty and "date" in snapshots:
                    snapshot_view = snapshots[pd.to_datetime(snapshots["date"], errors="coerce") <= as_of]
                opinion_view = opinions
                if not opinions.empty and "date" in opinions:
                    opinion_view = opinions[pd.to_datetime(opinions["date"], errors="coerce") <= as_of]
                features = build_base_features(history_view, universe, snapshot_view, opinion_view)
                features = add_cross_sectional_residuals(features)
                for policy in variants:
                    variant = str(policy["variant"])
                    min3 = float(policy["min3"])
                    max20 = float(policy["max20"])
                    max_rebound20 = float(policy["max_rebound20"])
                    require_reacceleration = bool(policy["reacceleration"])
                    production = bool(policy["production"])
                    variant_model = replace(
                        settings.model,
                        entry_min_return3_pct=min3,
                        entry_max_return20_pct=max20,
                        entry_max_rebound20_pct=max_rebound20,
                        persistent_entry_requires_reacceleration=require_reacceleration,
                        discovery_candidate_score=float(policy["discovery"]),
                        timing_buy_score=float(policy["timing"]),
                        strict_add_confirmation=bool(policy["strict_add"]),
                    )
                    scored = score_features(features, variant_model)
                    if policy["mode"] == "discovery":
                        candidates = scored[
                            (scored["discovery_score"] >= variant_model.discovery_candidate_score)
                            & (~scored["stage"].isin([
                                "가격상승", "수익실현", "분산·이탈", "이벤트확인",
                            ]))
                            & (~scored["campaign_regime"].isin([
                                "분산·이탈", "이벤트·구조확인", "이상대량·확인",
                                "기관주도·외인확인", "상승진행", "장기매집·휴식",
                            ]))
                            & (scored["data_points"] >= 60)
                        ]
                    else:
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
                        market = str(row.get("market", ""))
                        market_forward = float(
                            market_benchmarks.get(
                                market,
                                np.median(list(market_benchmarks.values())) if market_benchmarks else 0.0,
                            )
                        )
                        context = history[
                            (history["symbol"] == symbol)
                            & (history["date"] <= future["date"].iloc[-1])
                        ].sort_values("date")
                        scaled = simulate_scaled_strategy(context, as_of, variant_model)
                        signal_rows.append(
                            {
                                "variant": variant,
                                "production": production,
                                "entry_min_return3_pct": min3,
                                "entry_max_return20_pct": max20,
                                "entry_max_rebound20_pct": max_rebound20,
                                "requires_reacceleration": require_reacceleration,
                                "timing_buy_score": variant_model.timing_buy_score,
                                "strict_add_confirmation": variant_model.strict_add_confirmation,
                                "as_of": as_of,
                                "entry_date": future["date"].iloc[0],
                                "symbol": symbol,
                                "name": row.get("name", symbol),
                                "market": market,
                                "score": row["score"],
                                "entry_score": row["entry_score"],
                                "campaign_score": row["campaign_score"],
                                "discovery_score": row.get("discovery_score", 0.0),
                                "timing_score": row.get("timing_score", 0.0),
                                "distribution_score": row.get("distribution_score", 0.0),
                                "signal_score": row.get("signal_score", 0.0),
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
                                "market_forward_return": market_forward,
                                "excess_forward_return": path.forward_return - market_forward,
                                "realized_return": path.realized_return,
                                "max_upside": path.max_upside,
                                "max_drawdown": path.max_drawdown,
                                "ambiguous_same_day": path.ambiguous_same_day,
                                "scaled_strategy_return": scaled.return_rate,
                                "strategy_invested_krw": scaled.invested_krw,
                                "strategy_proceeds_krw": scaled.proceeds_krw,
                                "strategy_tranches": scaled.tranches,
                                "strategy_gap_half_sold": scaled.gap_half_sold,
                                "strategy_exit_reason": scaled.exit_reason,
                            }
                        )
                        last_signal_index[signal_key] = date_index
        results = pd.DataFrame(signal_rows)

    production_results = results[results["production"]].copy() if not results.empty else results
    if not production_results.empty and "scaled_strategy_return" in production_results:
        production_results["realized_return"] = production_results["scaled_strategy_return"]
    calibration = build_calibration(
        production_results,
        settings.model.backtest_target_pct / 100.0,
        settings.model.backtest_stop_pct / 100.0,
        settings.model.backtest_roundtrip_cost_pct / 100.0,
    )
    summary = summarize_variants(results)
    score_bands = summarize_score_bands(results)
    write_csv(results, settings.reports_dir / "backtest_latest.csv")
    write_csv(summary, settings.reports_dir / "backtest_thresholds.csv")
    write_csv(score_bands, settings.reports_dir / "backtest_score_bands.csv")
    (settings.data_dir / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_backtest_report(settings, results, summary, score_bands, calibration)
    return results


GATE_SWEEP_POLICIES = (
    {
        "variant": "current",
        "label": "현재 즉시매수",
        "min3": -3.0,
        "max20": 3.0,
        "max_rebound20": 5.0,
        "baseline": True,
    },
    {
        "variant": "plus_one",
        "label": "한 칸 완화",
        "min3": -3.0,
        "max20": 4.0,
        "max_rebound20": 6.0,
        "baseline": False,
    },
    {
        "variant": "plus_two",
        "label": "두 칸 완화",
        "min3": -3.0,
        "max20": 5.0,
        "max_rebound20": 7.0,
        "baseline": False,
    },
    {
        "variant": "watch_keep_3d",
        "label": "관심후보 가격만",
        "min3": -3.0,
        "max20": 6.0,
        "max_rebound20": 8.0,
        "baseline": False,
    },
    {
        "variant": "watch",
        "label": "관심후보 전체",
        "min3": -4.0,
        "max20": 6.0,
        "max_rebound20": 8.0,
        "baseline": False,
    },
)


def passes_price_gates(
    price_return3: float,
    price_return20: float,
    rebound_from_low20: float,
    min3: float,
    max20: float,
    max_rebound20: float,
) -> bool:
    return (
        float(price_return3) >= min3 / 100.0
        and float(price_return20) < max20 / 100.0
        and float(rebound_from_low20) < max_rebound20 / 100.0
    )


def apply_signal_cooldown(
    events: pd.DataFrame,
    horizon_days: int,
    date_ranks: dict[pd.Timestamp, int] | None = None,
) -> pd.DataFrame:
    if events.empty:
        return events
    frame = events.sort_values(["symbol", "as_of"]).copy()
    ranks = date_ranks or {
        pd.Timestamp(date): index for index, date in enumerate(sorted(frame["as_of"].unique()))
    }
    kept: list[int] = []
    last_index: dict[str, int] = {}
    for position, row in frame.iterrows():
        symbol = str(row["symbol"])
        rank = ranks.get(pd.Timestamp(row["as_of"]))
        if rank is None:
            continue
        previous = last_index.get(symbol)
        if previous is not None and rank - previous < horizon_days:
            continue
        kept.append(position)
        last_index[symbol] = rank
    return frame.loc[kept].reset_index(drop=True)


def _summarize_event_subset(variant: str, policy: dict, group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "variant": variant,
            "label": policy.get("label", variant),
            "min3": policy["min3"],
            "max20": policy["max20"],
            "max_rebound20": policy["max_rebound20"],
            "signals": 0,
            "resolved": 0,
            "resolved_win_rate": float("nan"),
            "positive_20d_rate": float("nan"),
            "mean_20d_return": float("nan"),
            "mean_excess_20d_return": float("nan"),
            "mean_scaled_strategy_return": float("nan"),
            "positive_scaled_strategy_rate": float("nan"),
            "mean_max_drawdown": float("nan"),
            "mean_tranches": float("nan"),
        }
    resolved = group[group["outcome"].isin(["target", "stop"])]
    wins = int((group["outcome"] == "target").sum())
    scaled = pd.to_numeric(group["scaled_strategy_return"], errors="coerce")
    return {
        "variant": variant,
        "label": policy.get("label", variant),
        "min3": policy["min3"],
        "max20": policy["max20"],
        "max_rebound20": policy["max_rebound20"],
        "signals": len(group),
        "resolved": len(resolved),
        "resolved_win_rate": wins / len(resolved) if len(resolved) else float("nan"),
        "positive_20d_rate": float((group["forward_return"] > 0).mean()),
        "mean_20d_return": float(group["forward_return"].mean()),
        "mean_excess_20d_return": float(pd.to_numeric(group["excess_forward_return"], errors="coerce").mean()),
        "mean_scaled_strategy_return": float(scaled.mean()),
        "positive_scaled_strategy_rate": float((scaled > 0).mean()),
        "mean_max_drawdown": float(group["max_drawdown"].mean()),
        "mean_tranches": float(pd.to_numeric(group["strategy_tranches"], errors="coerce").mean()),
    }


def recommend_gate_policy(
    summaries: pd.DataFrame,
    minimum_incremental_signals: int = 15,
) -> dict:
    """Pick the widest gate that does not clearly hurt incremental trades."""
    if summaries.empty:
        return {"chosen": "current", "reasons": ["비교 표본이 없습니다."], "applied": False}
    by_name = summaries.set_index("variant").to_dict("index")
    if "current" not in by_name:
        return {"chosen": "current", "reasons": ["기준 게이트 결과가 없습니다."], "applied": False}
    base = by_name["current"]
    chosen = "current"
    reasons = [
        (
            f"기준 현재 게이트 신호 {int(base['signals'])}건, 분할전략 "
            f"{base['mean_scaled_strategy_return'] * 100:+.2f}%, 시장초과 "
            f"{base['mean_excess_20d_return'] * 100:+.2f}%"
        )
    ]
    for name in ("plus_one", "plus_two", "watch_keep_3d", "watch"):
        row = by_name.get(name)
        incremental = by_name.get(f"{name}_incremental")
        if row is None or incremental is None:
            continue
        if int(incremental["signals"]) < minimum_incremental_signals:
            reasons.append(
                f"{row['label']}: 추가 신호 {int(incremental['signals'])}건으로 "
                f"{minimum_incremental_signals}건 미만이라 채택하지 않음"
            )
            continue
        overall_ok = (
            row["mean_scaled_strategy_return"]
            >= base["mean_scaled_strategy_return"] - 0.003
        )
        incremental_ok = (
            incremental["mean_scaled_strategy_return"]
            >= base["mean_scaled_strategy_return"] - 0.005
        )
        drawdown_ok = incremental["mean_max_drawdown"] >= base["mean_max_drawdown"] - 0.02
        excess_ok = incremental["mean_excess_20d_return"] >= min(
            0.0, base["mean_excess_20d_return"] - 0.01
        )
        if overall_ok and incremental_ok and drawdown_ok and excess_ok:
            chosen = name
            reasons.append(
                f"{row['label']}: 채택. 추가 신호 분할전략 "
                f"{incremental['mean_scaled_strategy_return'] * 100:+.2f}%, "
                f"낙폭 {incremental['mean_max_drawdown'] * 100:+.2f}%"
            )
        else:
            reasons.append(
                f"{row['label']}: 거절. 추가 신호 분할전략 "
                f"{incremental['mean_scaled_strategy_return'] * 100:+.2f}% / "
                f"낙폭 {incremental['mean_max_drawdown'] * 100:+.2f}% / "
                f"시장초과 {incremental['mean_excess_20d_return'] * 100:+.2f}%"
            )
    return {"chosen": chosen, "reasons": reasons, "applied": chosen != "current"}


def _init_gate_sweep_worker(payload: dict) -> None:
    _GATE_SWEEP_CTX.clear()
    _GATE_SWEEP_CTX.update(payload)


def _gate_sweep_one_date(date_index: int) -> tuple[int, list[dict]]:
    """Score the widest executable gate for one as-of date."""
    ctx = _GATE_SWEEP_CTX
    history = ctx["history"]
    universe = ctx["universe"]
    snapshots = ctx["snapshots"]
    opinions = ctx["opinions"]
    dates = ctx["dates"]
    horizon = int(ctx["horizon"])
    widest_model = ctx["widest_model"]
    eval_model = ctx["eval_model"]
    as_of = pd.Timestamp(dates[date_index])
    history_view = history[history["date"] <= as_of]
    market_benchmarks = market_forward_benchmarks(
        history,
        as_of,
        pd.Timestamp(dates[date_index + horizon]),
        eval_model.backtest_roundtrip_cost_pct / 100.0,
    )
    snapshot_view = snapshots
    if not snapshots.empty and "date" in snapshots:
        snapshot_view = snapshots[snapshots["date"] <= as_of]
    opinion_view = opinions
    if not opinions.empty and "date" in opinions:
        opinion_view = opinions[opinions["date"] <= as_of]
    features = build_base_features(history_view, universe, snapshot_view, opinion_view)
    features = add_cross_sectional_residuals(features)
    scored = score_features(features, widest_model)
    candidates = scored[
        scored["action"].isin(["강한 신규매수", "초기 분할매수"])
        & (scored["entry_score"] >= eval_model.entry_buy_score)
        & (scored["data_points"] >= 60)
    ]
    events: list[dict] = []
    for _, row in candidates.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        future = history[
            (history["symbol"] == symbol) & (history["date"] > as_of)
        ].sort_values("date").head(horizon)
        if len(future) < horizon:
            continue
        next_open = pd.to_numeric(future["open"].iloc[0], errors="coerce")
        entry_price = (
            float(next_open)
            if pd.notna(next_open) and next_open > 0
            else float(future["close"].iloc[0])
        )
        path = evaluate_forward_ohlc(
            entry_price,
            future,
            eval_model.backtest_target_pct / 100.0,
            eval_model.backtest_stop_pct / 100.0,
            eval_model.backtest_roundtrip_cost_pct / 100.0,
        )
        market = str(row.get("market", ""))
        market_forward = float(
            market_benchmarks.get(
                market,
                np.median(list(market_benchmarks.values())) if market_benchmarks else 0.0,
            )
        )
        context = history[
            (history["symbol"] == symbol)
            & (history["date"] <= future["date"].iloc[-1])
        ].sort_values("date")
        scaled = simulate_scaled_strategy(context, as_of, eval_model)
        events.append(
            {
                "as_of": as_of,
                "entry_date": future["date"].iloc[0],
                "symbol": symbol,
                "name": row.get("name", symbol),
                "market": market,
                "discovery_score": row.get("discovery_score", 0.0),
                "entry_score": row.get("entry_score", 0.0),
                "campaign_regime": row.get("campaign_regime", ""),
                "stage": row.get("stage", ""),
                "price_return3": float(row.get("price_return3", 0.0)),
                "price_return20": float(row.get("price_return20", 0.0)),
                "rebound_from_low20": float(row.get("rebound_from_low20", 0.0)),
                "outcome": path.outcome,
                "forward_return": path.forward_return,
                "excess_forward_return": path.forward_return - market_forward,
                "realized_return": path.realized_return,
                "max_drawdown": path.max_drawdown,
                "scaled_strategy_return": scaled.return_rate,
                "strategy_tranches": scaled.tranches,
            }
        )
    return date_index, events


def run_gate_sweep(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score once, then compare nested price-gate policies for executable buys."""
    history = read_csv(settings.data_dir / "history.csv")
    universe = read_csv(settings.data_dir / "universe.csv")
    snapshots = read_csv(settings.data_dir / "holdings.csv")
    opinions = read_csv(settings.data_dir / "opinions.csv")
    horizon = settings.model.backtest_horizon_days
    empty = pd.DataFrame()
    if history.empty or universe.empty:
        decision = recommend_gate_policy(empty)
        _write_gate_sweep_report(settings, empty, empty, decision)
        return empty, empty, decision

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["symbol"] = history["symbol"].astype("string").str.zfill(6)
    if not snapshots.empty and "date" in snapshots:
        snapshots = snapshots.copy()
        snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
        snapshots["symbol"] = snapshots["symbol"].astype("string").str.zfill(6)
    if not opinions.empty and "date" in opinions:
        opinions = opinions.copy()
        opinions["date"] = pd.to_datetime(opinions["date"], errors="coerce")
        opinions["symbol"] = opinions["symbol"].astype("string").str.zfill(6)
    dates = sorted(history["date"].dropna().unique())
    widest = GATE_SWEEP_POLICIES[-1]
    widest_model = replace(
        settings.model,
        entry_min_return3_pct=widest["min3"],
        entry_max_return20_pct=widest["max20"],
        entry_max_rebound20_pct=widest["max_rebound20"],
    )
    events: list[dict] = []
    if len(dates) >= 60 + horizon:
        date_indexes = list(range(59, len(dates) - horizon))
        payload = {
            "history": history,
            "universe": universe,
            "snapshots": snapshots,
            "opinions": opinions,
            "dates": dates,
            "horizon": horizon,
            "widest_model": widest_model,
            "eval_model": settings.model,
        }
        workers = max(1, min(6, (os.cpu_count() or 2) - 1))
        LOGGER.info("gate-sweep start: %s dates, %s workers", len(date_indexes), workers)
        if workers == 1:
            _init_gate_sweep_worker(payload)
            for step, date_index in enumerate(date_indexes, start=1):
                _, date_events = _gate_sweep_one_date(date_index)
                events.extend(date_events)
                if step == 1 or step % 5 == 0 or step == len(date_indexes):
                    LOGGER.info(
                        "gate-sweep %s/%s dates, events=%s",
                        step,
                        len(date_indexes),
                        len(events),
                    )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_gate_sweep_worker,
                initargs=(payload,),
            ) as pool:
                futures = [pool.submit(_gate_sweep_one_date, date_index) for date_index in date_indexes]
                done = 0
                for future in as_completed(futures):
                    _, date_events = future.result()
                    events.extend(date_events)
                    done += 1
                    if done == 1 or done % 5 == 0 or done == len(date_indexes):
                        LOGGER.info(
                            "gate-sweep %s/%s dates, events=%s",
                            done,
                            len(date_indexes),
                            len(events),
                        )
    events_frame = pd.DataFrame(events)
    summaries: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    baseline_keys: set[tuple] = set()
    date_ranks = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    for policy in GATE_SWEEP_POLICIES:
        if events_frame.empty:
            subset = events_frame
        else:
            mask = events_frame.apply(
                lambda row, item=policy: passes_price_gates(
                    row["price_return3"],
                    row["price_return20"],
                    row["rebound_from_low20"],
                    item["min3"],
                    item["max20"],
                    item["max_rebound20"],
                ),
                axis=1,
            )
            subset = apply_signal_cooldown(events_frame[mask], horizon, date_ranks)
        tagged = subset.copy()
        if not tagged.empty:
            tagged["variant"] = policy["variant"]
            selected_rows.append(tagged)
        summaries.append(_summarize_event_subset(policy["variant"], policy, subset))
        if policy["variant"] == "current" and not subset.empty:
            baseline_keys = set(zip(subset["symbol"], subset["as_of"]))
        elif not subset.empty:
            incremental = subset[
                ~subset.apply(lambda row: (row["symbol"], row["as_of"]) in baseline_keys, axis=1)
            ]
            incremental_tagged = incremental.copy()
            if not incremental_tagged.empty:
                incremental_tagged["variant"] = f"{policy['variant']}_incremental"
                selected_rows.append(incremental_tagged)
            summaries.append(
                _summarize_event_subset(
                    f"{policy['variant']}_incremental",
                    {**policy, "label": f"{policy['label']} 추가분만"},
                    incremental,
                )
            )
    summary = pd.DataFrame(summaries)
    decision = recommend_gate_policy(summary)
    detail = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    write_csv(detail, settings.reports_dir / "gate_sweep_latest.csv")
    write_csv(summary, settings.reports_dir / "gate_sweep_summary.csv")
    _write_gate_sweep_report(settings, detail, summary, decision)
    LOGGER.info(
        "gate-sweep done: chosen=%s applied=%s events=%s",
        decision.get("chosen"),
        decision.get("applied"),
        0 if events_frame.empty else len(events_frame),
    )
    for reason in decision.get("reasons", []):
        LOGGER.info("gate-sweep reason: %s", reason)
    return detail, summary, decision


def _write_gate_sweep_report(
    settings: Settings,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    decision: dict,
) -> None:
    path = settings.reports_dir / "gate_sweep.md"
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        path.write_text(
            "# 게이트 완화 검증\n\n비교할 워크포워드 신호가 없습니다.\n",
            encoding="utf-8",
        )
        return
    def pct_cell(value: float) -> str:
        numeric = float(value) if value is not None else float("nan")
        return "-" if not np.isfinite(numeric) else f"{numeric * 100:.1f}%"

    def signed_cell(value: float) -> str:
        numeric = float(value) if value is not None else float("nan")
        return "-" if not np.isfinite(numeric) else f"{numeric * 100:+.2f}%"

    rows = []
    for _, item in summary.iterrows():
        rows.append(
            f"| {item['label']} | 3일 {item['min3']:+.0f}% / 20일 +{item['max20']:.0f}% / "
            f"저점 +{item['max_rebound20']:.0f}% | {int(item['signals'])} | {int(item['resolved'])} | "
            f"{pct_cell(item['resolved_win_rate'])} | {pct_cell(item['positive_20d_rate'])} | "
            f"{signed_cell(item['mean_scaled_strategy_return'])} | "
            f"{pct_cell(item['positive_scaled_strategy_rate'])} | "
            f"{signed_cell(item['mean_20d_return'])} | {signed_cell(item['mean_excess_20d_return'])} | "
            f"{signed_cell(item['mean_max_drawdown'])} |"
        )
    chosen = str(decision.get("chosen", "current"))
    reasons = "\n".join(f"- {reason}" for reason in decision.get("reasons", []))
    text = (
        "# 게이트 완화 검증\n\n"
        "즉시매수 가격 게이트만 바꾸고, 발견점수·국면·분산·위험흡수 조건은 그대로 둡니다. "
        "점수는 가장 넓은 게이트로 한 번만 계산한 뒤 하위 게이트를 부분집합으로 비교합니다. "
        "추가분은 현재 게이트를 통과하지 못한 신호만 따로 본 결과입니다.\n\n"
        f"- 체결 가정: 신호 다음 거래일 시가, {settings.model.backtest_horizon_days}거래일, "
        f"목표 +{settings.model.backtest_target_pct:.0f}% / 손절 -{settings.model.backtest_stop_pct:.0f}%, "
        f"왕복 {settings.model.backtest_roundtrip_cost_pct:.1f}%\n"
        f"- 권고안: **{chosen}**\n"
        f"- 즉시매수 게이트 변경: {'예' if decision.get('applied') else '아니오'}\n\n"
        "## 비교표\n\n"
        "| 정책 | 게이트 | 신호 | 판정 | 목표 선도달 | 20일 양수 | 분할전략 평균 | 분할전략 양수 | 평균 20일 | 시장초과 | 평균 최대낙폭 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n## 판정 이유\n\n"
        + reasons
        + "\n\n추가 신호가 기준보다 뚜렷이 나빠지면 화면의 관심후보만 유지하고 즉시매수는 바꾸지 않습니다. "
        "표본이 작으므로 절대 승률보다 현재 대비 상대 비교를 사용합니다.\n"
    )
    path.write_text(text, encoding="utf-8")
