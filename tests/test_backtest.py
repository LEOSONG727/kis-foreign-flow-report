from __future__ import annotations

import pandas as pd

from kis_flow_bot.backtest import (
    apply_signal_cooldown,
    build_calibration,
    evaluate_forward_ohlc,
    evaluate_forward_path,
    market_forward_benchmarks,
    passes_price_gates,
    recommend_gate_policy,
    simulate_scaled_strategy,
    split_time_holdout,
    summarize_screen_bucket,
    summarize_variants,
)
from kis_flow_bot.config import ModelConfig


def test_time_holdout_splits_by_date_not_by_shuffle():
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(
                ["2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23", "2026-03-30"]
            ),
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            "excess_forward_return": [0.0, 0.01, 0.0, 0.02, 0.01],
            "outcome": ["horizon"] * 5,
            "max_drawdown": [-0.03] * 5,
        }
    )
    train, holdout, cut = split_time_holdout(frame, 0.40, horizon=5,
        trading_dates=pd.bdate_range("2026-03-02", "2026-04-10"))
    assert cut is not None
    assert train["as_of"].max() < holdout["as_of"].min()
    assert len(train) + len(holdout) == 4  # The boundary label is purged.
    assert train["label_end_date"].max() < holdout["as_of"].min()


def test_screen_bucket_summary_handles_empty_frame():
    stats = summarize_screen_bucket(pd.DataFrame())
    assert stats["signals"] == 0


def test_target_before_stop_is_a_win():
    result = evaluate_forward_path(100.0, pd.Series([102.0, 109.0, 94.0]), 0.08, 0.06)
    assert result.outcome == "target"
    assert result.days_to_event == 2


def test_stop_before_target_is_a_loss():
    result = evaluate_forward_path(100.0, pd.Series([93.0, 110.0]), 0.08, 0.06)
    assert result.outcome == "stop"
    assert result.days_to_event == 1


def test_calibration_stays_inactive_with_small_sample():
    results = pd.DataFrame([{"score": 75.0, "outcome": "target"}] * 10)
    calibration = build_calibration(results)
    assert not calibration["active"]
    assert calibration["resolved_samples"] == 10


def test_calibration_rejects_large_but_loss_making_sample():
    dates = pd.bdate_range("2025-01-02", periods=320)
    results = pd.DataFrame(
        [
            {
                "as_of": dates[index],
                "entry_score": 74.0,
                "outcome": "target" if index % 3 == 0 else "stop",
                "realized_return": 0.077 if index % 3 == 0 else -0.063,
            }
            for index in range(320)
        ]
    )
    calibration = build_calibration(results)
    assert not calibration["active"]
    assert calibration["quality_gate_version"] == 3
    assert not calibration["activation_checks"]["overall_positive_expectancy"]


def test_backtest_enters_next_open_and_uses_intraday_path_conservatively():
    future = pd.DataFrame(
        [
            {"open": 100.0, "high": 109.0, "low": 93.0, "close": 105.0},
            {"open": 105.0, "high": 110.0, "low": 104.0, "close": 109.0},
        ]
    )
    result = evaluate_forward_ohlc(100.0, future, target=0.08, stop=0.06, roundtrip_cost=0.003)
    assert result.outcome == "stop"
    assert result.days_to_event == 1
    assert result.ambiguous_same_day
    assert round(result.forward_return, 3) == 0.087


def test_market_benchmark_uses_same_market_mean_and_next_open():
    history = pd.DataFrame(
        [
            {"date": "2026-01-02", "symbol": "A", "market": "KOSPI", "open": 100, "close": 100},
            {"date": "2026-01-05", "symbol": "A", "market": "KOSPI", "open": 100, "close": 105},
            {"date": "2026-01-06", "symbol": "A", "market": "KOSPI", "open": 105, "close": 110},
            {"date": "2026-01-02", "symbol": "B", "market": "KOSPI", "open": 200, "close": 200},
            {"date": "2026-01-05", "symbol": "B", "market": "KOSPI", "open": 200, "close": 210},
            {"date": "2026-01-06", "symbol": "B", "market": "KOSPI", "open": 210, "close": 240},
            {"date": "2026-01-05", "symbol": "C", "market": "KOSDAQ", "open": 50, "close": 55},
            {"date": "2026-01-06", "symbol": "C", "market": "KOSDAQ", "open": 55, "close": 60},
        ]
    )
    history["date"] = pd.to_datetime(history["date"])
    result = market_forward_benchmarks(
        history,
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-06"),
        roundtrip_cost=0.003,
    )
    assert round(result["KOSPI"], 3) == 0.147
    assert "KOSDAQ" not in result  # C was not in the observable signal-date basket.


def test_variant_summary_reports_signal_quality():
    results = pd.DataFrame(
        [
            {"variant": "v1_balanced", "production": True, "entry_min_return3_pct": -3.0,
             "entry_max_return20_pct": 3.0, "entry_max_rebound20_pct": 5.0, "requires_reacceleration": True,
             "outcome": "target", "forward_return": 0.10, "realized_return": 0.077, "max_upside": 0.12, "max_drawdown": -0.03,
             "ambiguous_same_day": False},
            {"variant": "v1_balanced", "production": True, "entry_min_return3_pct": -3.0,
             "entry_max_return20_pct": 3.0, "entry_max_rebound20_pct": 5.0, "requires_reacceleration": True,
             "outcome": "stop", "forward_return": -0.07, "realized_return": -0.063, "max_upside": 0.02, "max_drawdown": -0.08,
             "ambiguous_same_day": True},
        ]
    )
    row = summarize_variants(results).iloc[0]
    assert row["signals"] == 2
    assert row["resolved_win_rate"] == 0.5
    assert row["ambiguous_days"] == 1


def _strategy_context(future_rows: list[dict]) -> tuple[pd.DataFrame, pd.Timestamp]:
    dates = pd.bdate_range("2026-01-02", periods=20 + len(future_rows))
    rows = [
        {
            "date": date, "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "foreign_net_amount": 10_000_000.0,
        }
        for date in dates[:20]
    ]
    for date, values in zip(dates[20:], future_rows):
        rows.append({"date": date, "foreign_net_amount": 10_000_000.0, **values})
    return pd.DataFrame(rows), dates[19]


def test_scaled_strategy_adds_only_on_next_open_while_flow_is_alive():
    context, signal_date = _strategy_context(
        [
            {"open": 100.0, "high": 101.0, "low": 95.0, "close": 96.0},
            {"open": 95.0, "high": 101.0, "low": 94.0, "close": 100.0},
        ]
    )
    result = simulate_scaled_strategy(
        context, signal_date, ModelConfig(strict_add_confirmation=False)
    )
    assert result.tranches == 2
    assert result.invested_krw > 10_000_000
    assert result.return_rate > 0


def test_strict_add_waits_until_flow_and_price_stabilize():
    context, signal_date = _strategy_context(
        [
            {"open": 100.0, "high": 101.0, "low": 94.0, "close": 95.0},
            {"open": 95.0, "high": 96.0, "low": 92.0, "close": 93.0},
            {"open": 94.0, "high": 96.0, "low": 93.0, "close": 95.0},
            {"open": 95.0, "high": 98.0, "low": 94.0, "close": 97.0},
            {"open": 97.0, "high": 99.0, "low": 96.0, "close": 98.0},
        ]
    )
    result = simulate_scaled_strategy(context, signal_date, ModelConfig())
    assert result.tranches == 2
    assert result.invested_krw > 10_000_000


def test_price_gates_keep_watch_names_out_of_current_buy():
    assert passes_price_gates(0.0, 0.045, 0.06, -3.0, 3.0, 5.0) is False
    assert passes_price_gates(0.0, 0.045, 0.06, -4.0, 6.0, 8.0) is True


def test_signal_cooldown_uses_full_calendar_ranks():
    events = pd.DataFrame(
        [
            {"symbol": "000010", "as_of": pd.Timestamp("2026-03-02")},
            {"symbol": "000010", "as_of": pd.Timestamp("2026-03-04")},
            {"symbol": "000010", "as_of": pd.Timestamp("2026-04-01")},
        ]
    )
    ranks = {
        pd.Timestamp("2026-03-02"): 0,
        pd.Timestamp("2026-03-04"): 2,
        pd.Timestamp("2026-04-01"): 22,
    }
    kept = apply_signal_cooldown(events, horizon_days=20, date_ranks=ranks)
    assert list(kept["as_of"].dt.strftime("%Y-%m-%d")) == ["2026-03-02", "2026-04-01"]


def test_gate_policy_accepts_wider_gates_when_incremental_is_similar():
    summaries = pd.DataFrame(
        [
            {
                "variant": "current", "label": "현재",
                "mean_scaled_strategy_return": 0.012, "mean_excess_20d_return": 0.04,
                "mean_max_drawdown": -0.10, "signals": 120,
            },
            {
                "variant": "watch", "label": "관심후보 전체",
                "mean_scaled_strategy_return": 0.013, "mean_excess_20d_return": 0.041,
                "mean_max_drawdown": -0.105, "signals": 160,
            },
            {
                "variant": "watch_incremental", "label": "관심후보 전체 추가분만",
                "mean_scaled_strategy_return": 0.011, "mean_excess_20d_return": 0.03,
                "mean_max_drawdown": -0.11, "signals": 40,
            },
        ]
    )
    decision = recommend_gate_policy(summaries)
    assert decision["chosen"] == "watch"
    assert decision["applied"] is True


def test_gate_policy_rejects_wider_gates_when_incremental_is_worse():
    summaries = pd.DataFrame(
        [
            {
                "variant": "current", "label": "현재",
                "mean_scaled_strategy_return": 0.012, "mean_excess_20d_return": 0.04,
                "mean_max_drawdown": -0.10, "signals": 120,
            },
            {
                "variant": "watch", "label": "관심후보 전체",
                "mean_scaled_strategy_return": 0.004, "mean_excess_20d_return": 0.01,
                "mean_max_drawdown": -0.14, "signals": 200,
            },
            {
                "variant": "watch_incremental", "label": "관심후보 전체 추가분만",
                "mean_scaled_strategy_return": -0.02, "mean_excess_20d_return": -0.01,
                "mean_max_drawdown": -0.16, "signals": 80,
            },
        ]
    )
    decision = recommend_gate_policy(summaries)
    assert decision["chosen"] == "current"
    assert decision["applied"] is False


def test_scaled_strategy_takes_gap_half_and_trails_the_rest():
    context, signal_date = _strategy_context(
        [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"open": 106.0, "high": 108.0, "low": 104.0, "close": 107.0},
            {"open": 105.0, "high": 106.0, "low": 98.0, "close": 100.0},
        ]
    )
    result = simulate_scaled_strategy(context, signal_date, ModelConfig())
    assert result.gap_half_sold
    assert result.exit_reason == "절반 익절 후 추적손절"
    assert result.return_rate > 0
