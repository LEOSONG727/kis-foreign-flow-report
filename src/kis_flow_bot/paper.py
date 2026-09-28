"""Paper trading: one portfolio engine for live forward tests and history.

Rules (pre-registered, identical for every strategy):
- Picks recorded after the close of day D are bought at the open of D+1.
- Up to ``slots`` equal-sized positions: each buy targets equity / slots at
  the time of the order. A name already held is not bought again; when all
  slots are full, new picks are skipped (no forced replacement).
- Each position is sold at the open after it has been held ``hold_sessions``
  closes. Costs are charged on both sides.
- A missing open (suspension) delays a sell and cancels a buy. A missing close
  is marked at the last known close.

Live mode replays an append-only signal ledger. Each day's picks are frozen
the first time they are recorded, so the forward record cannot be rewritten
after prices are known.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)

BUY_REVIEW = "오늘 신규매수 검토"
STRATEGY_LABELS = {
    "official": "현행 규칙 (오늘 신규매수 검토)",
    "discovery_top5": "발견점수 상위 5",
    "learned_top5": "학습 모델 상위 5",
    "filing_veto_top5": "발견 상위 5 · 공시 희석·악재 제외",
}
SIGNAL_COLUMNS = ["as_of", "strategy", "rank", "symbol", "name", "captured_at"]


@dataclass(frozen=True)
class PaperConfig:
    slots: int = 10
    hold_sessions: int = 20
    cost_per_side: float = 0.0015
    initial_equity: float = 100_000_000.0
    picks_per_day: int = 5


def paper_config(settings: Settings) -> PaperConfig:
    raw = getattr(settings, "paper", None)
    base = PaperConfig(cost_per_side=settings.model.backtest_roundtrip_cost_pct / 200.0)
    if raw is None:
        return base
    return PaperConfig(
        slots=int(raw.slots),
        hold_sessions=int(raw.hold_sessions),
        cost_per_side=base.cost_per_side,
        initial_equity=float(raw.initial_equity),
        picks_per_day=int(raw.picks_per_day),
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def simulate(
    signals: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    config: PaperConfig,
    start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy in ``signals`` through the same portfolio rules."""

    nav_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    if signals.empty or closes.empty:
        return pd.DataFrame(), pd.DataFrame()
    signals = signals.copy()
    signals["as_of"] = pd.to_datetime(signals["as_of"])
    signals["symbol"] = signals["symbol"].astype(str).str.zfill(6)
    dates = [pd.Timestamp(d) for d in closes.index]
    first = pd.Timestamp(start) if start is not None else signals["as_of"].min()
    dates = [d for d in dates if d >= first]
    last_close = closes.ffill()
    for strategy, group in signals.groupby("strategy", sort=True):
        picks_by_day = {
            day: rows.sort_values("rank")["symbol"].tolist()
            for day, rows in group.groupby("as_of")
        }
        cash = config.initial_equity
        positions: dict[str, dict[str, Any]] = {}
        pending_buys: list[tuple[str, float, pd.Timestamp]] = []
        pending_sells: list[str] = []
        for day in dates:
            opens_today = opens.loc[day] if day in opens.index else pd.Series(dtype=float)
            # 1) fills at today's open
            still_pending = []
            for symbol in pending_sells:
                price = opens_today.get(symbol, np.nan)
                if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
                    still_pending.append(symbol)
                    continue
                position = positions.pop(symbol)
                proceeds = position["shares"] * price * (1.0 - config.cost_per_side)
                cash += proceeds
                trade_rows.append(
                    {
                        "strategy": strategy,
                        "symbol": symbol,
                        "signal_date": position["signal_date"],
                        "entry_date": position["entry_date"],
                        "exit_date": day,
                        "entry_price": position["entry_price"],
                        "exit_price": float(price),
                        "return": proceeds / position["cost"] - 1.0,
                        "sessions": position["held"],
                    }
                )
            pending_sells = still_pending
            for symbol, target, signal_day in pending_buys:
                price = opens_today.get(symbol, np.nan)
                if symbol in positions or not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
                    continue
                shares = math.floor(min(target, cash) / (price * (1.0 + config.cost_per_side)))
                if shares <= 0:
                    continue
                cost = shares * price * (1.0 + config.cost_per_side)
                cash -= cost
                positions[symbol] = {
                    "shares": shares,
                    "entry_price": float(price),
                    "entry_date": day,
                    "signal_date": signal_day,
                    "cost": cost,
                    "held": 0,
                }
            pending_buys = []
            # 2) mark at the close
            closes_today = last_close.loc[day]
            value = 0.0
            for symbol, position in positions.items():
                position["held"] += 1
                mark = closes_today.get(symbol, np.nan)
                if not (isinstance(mark, (int, float)) and math.isfinite(mark)):
                    mark = position["entry_price"]
                value += position["shares"] * float(mark)
            equity = cash + value
            nav_rows.append(
                {
                    "date": day,
                    "strategy": strategy,
                    "equity": equity,
                    "cash": cash,
                    "positions": len(positions),
                    "exposure": value / equity if equity > 0 else 0.0,
                }
            )
            # 3) orders for the next open
            for symbol, position in positions.items():
                if position["held"] >= config.hold_sessions and symbol not in pending_sells:
                    pending_sells.append(symbol)
            free = config.slots - (len(positions) - len(pending_sells))
            for symbol in picks_by_day.get(day, [])[: config.picks_per_day]:
                if free <= 0:
                    break
                if symbol in positions:
                    continue
                pending_buys.append((symbol, equity / config.slots, day))
                free -= 1
    return pd.DataFrame(nav_rows), pd.DataFrame(trade_rows)


def benchmark_nav(closes: pd.DataFrame, start: pd.Timestamp, initial: float) -> pd.DataFrame:
    """Equal-weight daily-rebalanced index of every stored stock."""

    frame = closes[closes.index >= pd.Timestamp(start)]
    if len(frame) < 2:
        return pd.DataFrame(columns=["date", "strategy", "equity"])
    returns = frame.pct_change(fill_method=None).clip(-0.3, 0.3).mean(axis=1).fillna(0.0)
    equity = initial * (1.0 + returns).cumprod()
    return pd.DataFrame({"date": frame.index, "strategy": "benchmark", "equity": equity.to_numpy()})


def summarize(nav: pd.DataFrame, trades: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if nav.empty:
        return pd.DataFrame()
    bench = benchmark.set_index("date")["equity"] if not benchmark.empty else pd.Series(dtype=float)
    for strategy, group in nav.groupby("strategy"):
        series = group.set_index("date")["equity"]
        start_equity = float(series.iloc[0])
        total = float(series.iloc[-1] / start_equity - 1.0)
        drawdown = float((series / series.cummax() - 1.0).min())
        span = bench.reindex(series.index).dropna()
        bench_total = float(span.iloc[-1] / span.iloc[0] - 1.0) if len(span) >= 2 else float("nan")
        closed = trades[trades["strategy"] == strategy] if not trades.empty else trades
        rows.append(
            {
                "strategy": strategy,
                "days": int(len(series)),
                "return": total,
                "benchmark": bench_total,
                "excess": total - bench_total if math.isfinite(bench_total) else float("nan"),
                "max_drawdown": drawdown,
                "avg_exposure": float(group["exposure"].mean()),
                "closed_trades": int(len(closed)),
                "win_rate": float((closed["return"] > 0).mean()) if len(closed) else float("nan"),
                "avg_trade": float(closed["return"].mean()) if len(closed) else float("nan"),
                "open_positions": int(group["positions"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def live_signals(scored: pd.DataFrame, picks: int = 5) -> pd.DataFrame:
    """Today's picks per strategy, from the fully annotated daily frame."""

    if scored.empty or "as_of" not in scored:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    eligible = scored
    if "discovery_eligible" in scored:
        eligible = scored[scored["discovery_eligible"].fillna(False).astype(bool)]
    as_of = pd.to_datetime(eligible["as_of"]).max() if not eligible.empty else pd.NaT
    ordered = eligible.sort_values(["discovery_score", "timing_score", "entry_score"], ascending=False)
    chosen: dict[str, pd.DataFrame] = {}
    if "proposal_status" in ordered:
        chosen["official"] = ordered[ordered["proposal_status"].astype(str) == BUY_REVIEW].head(picks)
    chosen["discovery_top5"] = ordered.head(picks)
    if "learned_rank" in ordered and ordered["learned_rank"].notna().any():
        chosen["learned_top5"] = ordered.dropna(subset=["learned_rank"]).sort_values("learned_rank").head(picks)
    if "kw_dilution" in ordered:
        risky = ordered["kw_dilution"].fillna(False).astype(bool) | ordered["kw_adverse"].fillna(False).astype(bool)
        chosen["filing_veto_top5"] = ordered[~risky].head(picks)
    rows = []
    for strategy, frame in chosen.items():
        for rank, (_, row) in enumerate(frame.iterrows(), 1):
            rows.append(
                {
                    "as_of": as_of.strftime("%Y-%m-%d"),
                    "strategy": strategy,
                    "rank": rank,
                    "symbol": str(row["symbol"]).zfill(6),
                    "name": str(row.get("name", "")),
                }
            )
        if frame.empty:
            # An explicit "no pick" row keeps cash days in the frozen record.
            rows.append({"as_of": as_of.strftime("%Y-%m-%d"), "strategy": strategy, "rank": 0, "symbol": "", "name": ""})
    return pd.DataFrame(rows, columns=[c for c in SIGNAL_COLUMNS if c != "captured_at"])


def append_signals(ledger: pd.DataFrame, today: pd.DataFrame, captured_at: str) -> pd.DataFrame:
    """First capture per (date, strategy) wins; reruns cannot change it."""

    if today.empty:
        return ledger
    today = today.assign(captured_at=captured_at)
    if ledger.empty:
        return today[SIGNAL_COLUMNS]
    known = set(zip(ledger["as_of"].astype(str), ledger["strategy"].astype(str)))
    fresh = today[[key not in known for key in zip(today["as_of"].astype(str), today["strategy"].astype(str))]]
    return pd.concat([ledger[SIGNAL_COLUMNS], fresh[SIGNAL_COLUMNS]], ignore_index=True)


def _price_pivots(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = history.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
    for column in ("open", "close"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    opens = prices.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    closes = prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    return opens, closes


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(number) else f"{number * 100:+.2f}%"


def render_summary_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| 전략 | 거래일 | 수익률 | 동일가중 지수 | 초과 | 최대낙폭 | 평균 투자비중 | 청산 거래 | 승률 | 거래당 평균 | 보유 중 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        label = STRATEGY_LABELS.get(row["strategy"], row["strategy"])
        win = "-" if not math.isfinite(row["win_rate"]) else f"{row['win_rate']:.0%}"
        lines.append(
            f"| {label} | {row['days']} | {_pct(row['return'])} | {_pct(row['benchmark'])} | {_pct(row['excess'])} | "
            f"{_pct(row['max_drawdown'])} | {row['avg_exposure']:.0%} | {row['closed_trades']} | {win} | "
            f"{_pct(row['avg_trade'])} | {row['open_positions']} |"
        )
    return lines


def update_paper_trading(
    scored: pd.DataFrame,
    history: pd.DataFrame,
    settings: Settings,
    generated_at: datetime,
) -> pd.DataFrame:
    """Freeze today's picks, replay the whole forward record, write the report."""

    config = paper_config(settings)
    start = pd.Timestamp(getattr(getattr(settings, "paper", None), "start_date", "") or "1900-01-01")
    ledger_path = settings.data_dir / "paper_signals.csv"
    ledger = read_csv(ledger_path)
    today = live_signals(scored, config.picks_per_day)
    if not today.empty and pd.Timestamp(today["as_of"].iloc[0]) >= start:
        ledger = append_signals(ledger, today, generated_at.isoformat(timespec="seconds"))
        write_csv(ledger, ledger_path)
    if ledger.empty:
        return pd.DataFrame()
    opens, closes = _price_pivots(history)
    picks = ledger[ledger["symbol"].astype(str).str.strip().ne("") & ledger["symbol"].notna()]
    first = pd.to_datetime(ledger["as_of"]).min()
    nav, trades = simulate(picks.assign(symbol=picks["symbol"].astype(str)), opens, closes, config, start=first)
    strategies = sorted(ledger["strategy"].unique())
    if not nav.empty:
        # Strategies that never picked still exist, as cash.
        missing = [s for s in strategies if s not in set(nav["strategy"])]
        days = sorted(nav["date"].unique())
        for strategy in missing:
            nav = pd.concat(
                [nav, pd.DataFrame({"date": days, "strategy": strategy, "equity": config.initial_equity,
                                    "cash": config.initial_equity, "positions": 0, "exposure": 0.0})],
                ignore_index=True,
            )
    bench = benchmark_nav(closes, first, config.initial_equity)
    summary = summarize(nav, trades, bench)
    write_csv(nav, settings.data_dir / "paper_nav.csv")
    write_csv(trades, settings.data_dir / "paper_trades.csv")
    lines = [
        "# 모의 운용 (실전 기록)",
        "",
        f"- 갱신: {generated_at.isoformat(timespec='seconds')} · 시작: {first.date()} · 신호 기록 {ledger['as_of'].nunique()}일",
        f"- 규칙: 신호 다음 날 시가 매수, 최대 {config.slots}종목 동일 비중, {config.hold_sessions}거래일 보유 후 다음 시가 매도, "
        f"편도 비용 {config.cost_per_side:.2%}, 시작 자금 {config.initial_equity / 1e8:.0f}억 원(가상)",
        "- 매일 추천은 처음 기록된 순간 고정됩니다. 가격을 본 뒤 고칠 수 없습니다.",
        "- 비교 지수는 저장된 전 종목의 동일가중 일간 수익률입니다.",
        "",
    ]
    lines.extend(render_summary_table(summary) if not summary.empty else ["아직 체결된 거래가 없습니다."])
    lines.extend(["", "20거래일 보유 기준이라 최소 1~2개월은 지나야 의미 있는 비교가 됩니다. 이 기록은 모델을 자동 변경하지 않습니다.", ""])
    (settings.reports_dir / "paper_trading.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def paper_section(summary: pd.DataFrame | None) -> tuple[str, str]:
    import html as html_lib

    if summary is None or summary.empty:
        return "", ""
    lines = []
    for _, row in summary.iterrows():
        label = STRATEGY_LABELS.get(row["strategy"], row["strategy"])
        lines.append(f"{label}: {_pct(row['return'])} (지수 대비 {_pct(row['excess'])}, 보유 {row['open_positions']}종목)")
    markdown = "\n\n## 모의 운용\n" + "\n".join(f"- {line}" for line in lines) + "\n"
    html_block = (
        "<section class='panel'><h2>모의 운용</h2><ul>"
        + "".join(f"<li>{html_lib.escape(line)}</li>" for line in lines)
        + "</ul><p>가상 자금 기록입니다. 자세한 내용은 reports/paper_trading.md</p></section>"
    )
    return markdown, html_block


def portfolio_backtest(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    settings: Settings,
    generated_at: datetime,
) -> pd.DataFrame:
    """Same engine on past signals: discovery top 5 vs walk-forward learned top 5."""

    config = paper_config(settings)
    signals = []
    if not panel.empty:
        ordered = panel.sort_values(["as_of", "discovery_score", "timing_score"], ascending=[True, False, False])
        top = ordered.groupby("as_of").head(config.picks_per_day).copy()
        top["rank"] = top.groupby("as_of").cumcount() + 1
        signals.append(top.assign(strategy="discovery_top5")[["as_of", "strategy", "rank", "symbol"]])
    if predictions is not None and not predictions.empty:
        ordered = predictions.sort_values(["as_of", "challenger_score"], ascending=[True, False])
        top = ordered.groupby("as_of").head(config.picks_per_day).copy()
        top["rank"] = top.groupby("as_of").cumcount() + 1
        signals.append(top.assign(strategy="learned_top5")[["as_of", "strategy", "rank", "symbol"]])
    if not signals:
        return pd.DataFrame()
    frame = pd.concat(signals, ignore_index=True)
    # Compare both on the dates where both exist (learned needs a training warm-up).
    common_start = frame.groupby("strategy")["as_of"].min().max()
    frame = frame[frame["as_of"] >= common_start]
    opens, closes = _price_pivots(history)
    nav, trades = simulate(frame, opens, closes, config, start=common_start)
    bench = benchmark_nav(closes, common_start, config.initial_equity)
    summary = summarize(nav, trades, bench)
    lines = [
        "# 포트폴리오 백테스트",
        "",
        f"- 갱신: {generated_at.isoformat(timespec='seconds')} · 기간: {pd.Timestamp(common_start).date()} 이후 저장 이력 끝까지",
        f"- 모의 운용과 같은 규칙: 다음 날 시가 매수, 최대 {config.slots}종목, {config.hold_sessions}거래일 보유, 편도 {config.cost_per_side:.2%}",
        "- 발견점수는 이 기간 데이터로 규칙을 조정했으므로 표본 내 결과입니다. 학습 모델은 워크포워드(그날까지 확정된 결과로만 학습)입니다.",
        "- 동시 보유·현금·비용을 반영한 계좌 수익률이라, 종목별 평균보다 실제 운용에 가깝습니다.",
        "",
    ]
    lines.extend(render_summary_table(summary) if not summary.empty else ["거래 없음"])
    lines.append("")
    (settings.reports_dir / "portfolio_backtest.md").write_text("\n".join(lines), encoding="utf-8")
    write_csv(summary, settings.reports_dir / "portfolio_backtest.csv")
    return summary
