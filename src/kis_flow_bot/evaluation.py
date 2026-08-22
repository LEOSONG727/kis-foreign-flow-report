"""Multi-horizon and multi-exit evaluation helpers.

These functions do not change discovery scoring or price gates.
They only measure outcomes under agreed evaluation policies:

- horizons: 20 / 40 / 60 sessions
- exits: plain hold, flow-invalidation only, current scaled strategy
- market regime attribution over the forward window
- bootstrap confidence intervals for summary tables
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig


EVAL_HORIZONS: tuple[int, ...] = (20, 40, 60)
STRATEGY_MODES: tuple[str, ...] = ("hold", "flow_invalidation", "scaled")


@dataclass(frozen=True)
class ExitResult:
    return_rate: float
    invested_krw: float
    proceeds_krw: float
    tranches: int
    gap_half_sold: bool
    exit_reason: str


@dataclass(frozen=True)
class HorizonPath:
    horizon: int
    available: bool
    forward_return: float
    excess_return: float
    max_upside: float
    max_drawdown: float
    market_return: float


@dataclass(frozen=True)
class StrategyPath:
    mode: str
    horizon: int
    available: bool
    return_rate: float
    exit_reason: str
    tranches: int


def classify_market_regime(market_return: float, up_pct: float = 0.03, down_pct: float = -0.03) -> str:
    """Attribute a completed path to an up / sideways / down market window."""
    if market_return >= up_pct:
        return "상승"
    if market_return <= down_pct:
        return "하락"
    return "횡보"


def _numeric_frame(context: pd.DataFrame) -> pd.DataFrame:
    frame = context.copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "foreign_net_amount"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    return frame.dropna(subset=["date", "open", "close"])


def forward_slice(
    context: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    frame = _numeric_frame(context)
    future = frame[frame["date"] > pd.Timestamp(signal_date)].head(horizon)
    return future.reset_index(drop=True)


def evaluate_hold_path(
    context: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizon: int,
    roundtrip_cost_pct: float = 0.3,
    market_return: float | None = None,
) -> HorizonPath:
    """Next-open entry, hold through horizon close, deduct round-trip cost."""
    future = forward_slice(context, signal_date, horizon)
    if len(future) < horizon:
        return HorizonPath(horizon, False, 0.0, 0.0, 0.0, 0.0, 0.0)
    entry = float(future.iloc[0]["open"])
    if entry <= 0:
        return HorizonPath(horizon, False, 0.0, 0.0, 0.0, 0.0, 0.0)
    cost = roundtrip_cost_pct / 100.0
    end_close = float(future.iloc[-1]["close"])
    high = float(pd.to_numeric(future["high"], errors="coerce").max())
    low = float(pd.to_numeric(future["low"], errors="coerce").min())
    forward = end_close / entry - 1.0 - cost
    market = 0.0 if market_return is None else float(market_return)
    return HorizonPath(
        horizon=horizon,
        available=True,
        forward_return=forward,
        excess_return=forward - market,
        max_upside=high / entry - 1.0 - cost,
        max_drawdown=low / entry - 1.0 - cost,
        market_return=market,
    )


def simulate_flow_invalidation(
    context: pd.DataFrame,
    signal_date: pd.Timestamp,
    model: ModelConfig,
    horizon: int,
) -> ExitResult:
    """Buy once at next open; exit next open only when the live flow thesis dies."""
    frame = _numeric_frame(context)
    future_index = frame.index[frame["date"] > pd.Timestamp(signal_date)].tolist()
    if not future_index:
        return ExitResult(0.0, 0.0, 0.0, 0, False, "미체결")
    future_index = future_index[:horizon]
    side_cost = model.backtest_roundtrip_cost_pct / 200.0
    shares = 0
    invested = 0.0
    proceeds = 0.0
    pending_exit = False
    exit_reason = "평가기간 종료"
    for future_day, idx in enumerate(future_index):
        row = frame.loc[idx]
        open_price = float(row["open"])
        if pending_exit and shares > 0:
            proceeds += shares * open_price * (1.0 - side_cost)
            shares = 0
            exit_reason = "수급 신호 무효"
            break
        if future_day == 0:
            quantity = int(model.initial_order_krw // (open_price * (1.0 + side_cost)))
            if quantity <= 0:
                return ExitResult(0.0, 0.0, 0.0, 0, False, "미체결")
            invested = quantity * open_price * (1.0 + side_cost)
            shares = quantity
        if shares <= 0:
            break
        close_slice = frame.loc[:idx].tail(20)
        flow20 = float(close_slice["foreign_net_amount"].fillna(0.0).sum())
        positive_flow = float(
            close_slice.loc[close_slice["foreign_net_amount"] > 0, "foreign_net_amount"].sum()
        )
        negative_flow = abs(
            float(close_slice.loc[close_slice["foreign_net_amount"] < 0, "foreign_net_amount"].sum())
        )
        sellback = negative_flow / positive_flow if positive_flow > 0 else 9.0
        flow3 = float(close_slice.tail(3)["foreign_net_amount"].fillna(0.0).sum())
        flow5 = float(close_slice.tail(5)["foreign_net_amount"].fillna(0.0).sum())
        base_close = float(close_slice["close"].iloc[0]) if not close_slice.empty else float(row["close"])
        price20 = float(row["close"]) / base_close - 1.0 if base_close > 0 else 0.0
        signal_alive = (
            flow20 > 0
            and sellback < 1.0
            and not (flow3 <= 0 and flow5 <= 0)
            and price20 > model.structural_break_return20_pct / 100.0
        )
        if not signal_alive:
            pending_exit = True
    if shares > 0:
        final_price = float(frame.loc[future_index[-1], "close"])
        proceeds += shares * final_price * (1.0 - side_cost)
    return ExitResult(
        return_rate=(proceeds / invested - 1.0) if invested > 0 else 0.0,
        invested_krw=invested,
        proceeds_krw=proceeds,
        tranches=1 if invested > 0 else 0,
        gap_half_sold=False,
        exit_reason=exit_reason,
    )


def simulate_scaled_to_horizon(
    context: pd.DataFrame,
    signal_date: pd.Timestamp,
    model: ModelConfig,
    horizon: int,
) -> ExitResult:
    """Current scaled strategy, but stop measuring after `horizon` sessions."""
    from .backtest import simulate_scaled_strategy

    frame = _numeric_frame(context)
    cutoff_dates = frame.loc[frame["date"] > pd.Timestamp(signal_date), "date"].head(horizon)
    if cutoff_dates.empty:
        return ExitResult(0.0, 0.0, 0.0, 0, False, "미체결")
    limited = frame[frame["date"] <= cutoff_dates.iloc[-1]].copy()
    scaled = simulate_scaled_strategy(limited, signal_date, model)
    return ExitResult(
        return_rate=scaled.return_rate,
        invested_krw=scaled.invested_krw,
        proceeds_krw=scaled.proceeds_krw,
        tranches=scaled.tranches,
        gap_half_sold=scaled.gap_half_sold,
        exit_reason=scaled.exit_reason,
    )


def evaluate_strategy_modes(
    context: pd.DataFrame,
    signal_date: pd.Timestamp,
    model: ModelConfig,
    horizon: int,
) -> list[StrategyPath]:
    hold = evaluate_hold_path(
        context,
        signal_date,
        horizon,
        model.backtest_roundtrip_cost_pct,
    )
    flow = simulate_flow_invalidation(context, signal_date, model, horizon)
    scaled = simulate_scaled_to_horizon(context, signal_date, model, horizon)
    available = hold.available
    return [
        StrategyPath("hold", horizon, available, hold.forward_return, "평가기간 종료", 1 if available else 0),
        StrategyPath(
            "flow_invalidation",
            horizon,
            available and flow.tranches > 0,
            flow.return_rate,
            flow.exit_reason,
            flow.tranches,
        ),
        StrategyPath(
            "scaled",
            horizon,
            available and scaled.tranches > 0,
            scaled.return_rate,
            scaled.exit_reason,
            scaled.tranches,
        ),
    ]


def bootstrap_mean_ci(
    values: Iterable[float],
    samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return mean and percentile CI. Empty input yields NaNs."""
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return float("nan"), float("nan"), float("nan")
    mean = float(series.mean())
    if len(series) < 3:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(series.to_numpy(), size=(samples, len(series)), replace=True)
    boots = draws.mean(axis=1)
    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1 - alpha / 2))
    return mean, low, high


def summarize_strategy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-horizon strategy rows into a report-ready table."""
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    group_cols = [c for c in ("variant", "horizon", "strategy", "regime") if c in frame]
    if not group_cols:
        group_cols = ["horizon", "strategy"] if "strategy" in frame else ["horizon"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        returns = pd.to_numeric(group.get("return_rate", group.get("forward_return")), errors="coerce")
        excess = pd.to_numeric(group.get("excess_return"), errors="coerce") if "excess_return" in group else pd.Series(dtype=float)
        mean, low, high = bootstrap_mean_ci(returns.dropna().tolist())
        row = {name: value for name, value in zip(group_cols, keys)}
        row.update(
            {
                "signals": int(len(group)),
                "mean_return": mean,
                "ci_low": low,
                "ci_high": high,
                "positive_rate": float((returns > 0).mean()) if returns.notna().any() else float("nan"),
                "mean_excess": float(excess.mean()) if excess.notna().any() else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def render_evaluation_markdown(
    summary: pd.DataFrame,
    title: str = "다기간·다전략 평가",
) -> str:
    lines = [
        f"# {title}",
        "",
        "메인 발견점수·75점·가격 게이트는 바꾸지 않습니다.",
        "이 표는 평가 장치만 확장한 결과입니다.",
        "",
        "- 보유: 다음 거래일 시가 진입 후 기간 종료까지 보유",
        "- 수급무효: 외인 매집 신호가 깨질 때만 다음 시가 청산",
        "- 분할전략: 현재 추가매수·갭 익절·추적손절",
        "",
        "| 변형 | 기간 | 전략 | 국면 | 신호 | 평균수익 | 95% CI | 양수비율 | 시장초과 |",
        "|---|---:|---|---|---:|---:|---|---:|---:|",
    ]
    if summary.empty:
        lines.append("| - | - | - | - | 0 | - | - | - | - |")
        lines.append("")
        return "\n".join(lines)

    def pct(value: float) -> str:
        return "-" if value is None or not np.isfinite(value) else f"{value * 100:+.2f}%"

    def band(low: float, high: float) -> str:
        if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
            return "-"
        return f"[{low * 100:+.2f}%, {high * 100:+.2f}%]"

    for _, row in summary.iterrows():
        lines.append(
            f"| {row.get('variant', '-')} | D+{int(row.get('horizon', 0))} | "
            f"{row.get('strategy', '-')} | {row.get('regime', '전체')} | {int(row.get('signals', 0))} | "
            f"{pct(float(row.get('mean_return', float('nan'))))} | "
            f"{band(float(row.get('ci_low', float('nan'))), float(row.get('ci_high', float('nan'))))} | "
            f"{pct(float(row.get('positive_rate', float('nan')))).replace('+', '')} | "
            f"{pct(float(row.get('mean_excess', float('nan'))))} |"
        )
    lines.append("")
    lines.append(
        "실전 D+40/D+60 표본은 각각 약 8주·12주 후부터 쌓입니다. "
        "당장은 과거 이력 백테스트로 구조를 검증하고, 라이브는 같은 스키마로 누적합니다."
    )
    lines.append("")
    return "\n".join(lines)
