from __future__ import annotations

import pandas as pd

from kis_flow_bot.config import ModelConfig
from kis_flow_bot.evaluation import (
    EVAL_HORIZONS,
    bootstrap_mean_ci,
    classify_market_regime,
    evaluate_hold_path,
    evaluate_strategy_modes,
    forward_slice,
    render_evaluation_markdown,
    simulate_flow_invalidation,
    summarize_strategy_frame,
)


def _context(n_future: int = 60) -> tuple[pd.DataFrame, pd.Timestamp]:
    dates = pd.bdate_range("2026-01-02", periods=20 + n_future)
    rows = [
        {
            "date": date,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "foreign_net_amount": 10_000_000.0,
        }
        for date in dates[:20]
    ]
    for offset, date in enumerate(dates[20:], start=1):
        rows.append(
            {
                "date": date,
                "open": 100.0 + offset * 0.1,
                "high": 101.0 + offset * 0.1,
                "low": 99.0 + offset * 0.1,
                "close": 100.0 + offset * 0.2,
                "foreign_net_amount": 10_000_000.0,
            }
        )
    return pd.DataFrame(rows), dates[19]


def test_eval_horizons_include_40_and_60():
    assert EVAL_HORIZONS == (20, 40, 60)


def test_classify_market_regime():
    assert classify_market_regime(0.05) == "상승"
    assert classify_market_regime(-0.05) == "하락"
    assert classify_market_regime(0.0) == "횡보"


def test_hold_path_requires_full_horizon():
    context, signal_date = _context(30)
    short = evaluate_hold_path(context, signal_date, 40)
    assert short.available is False
    full_ctx, signal_date = _context(60)
    long = evaluate_hold_path(full_ctx, signal_date, 40)
    assert long.available is True
    assert long.forward_return != 0.0


def test_flow_invalidation_exits_when_flow_dies():
    context, signal_date = _context(20)
    # Wipe foreign flow after entry window so thesis dies.
    future = context["date"] > signal_date
    context = context.copy()
    context.loc[future, "foreign_net_amount"] = -50_000_000.0
    result = simulate_flow_invalidation(context, signal_date, ModelConfig(), horizon=20)
    assert result.tranches == 1
    assert result.exit_reason in {"수급 신호 무효", "평가기간 종료"}


def test_strategy_modes_return_three_paths():
    context, signal_date = _context(20)
    paths = evaluate_strategy_modes(context, signal_date, ModelConfig(), horizon=20)
    assert [p.mode for p in paths] == ["hold", "flow_invalidation", "scaled"]
    assert all(p.available for p in paths)


def test_bootstrap_ci_and_summary_markdown():
    mean, low, high = bootstrap_mean_ci([0.01, 0.02, -0.01, 0.03, 0.0] * 5)
    assert mean == mean  # not NaN
    frame = pd.DataFrame(
        [
            {
                "variant": "flow_price_v2",
                "horizon": 20,
                "strategy": "hold",
                "regime": "횡보",
                "return_rate": 0.01,
                "excess_return": 0.02,
            },
            {
                "variant": "flow_price_v2",
                "horizon": 20,
                "strategy": "hold",
                "regime": "횡보",
                "return_rate": 0.03,
                "excess_return": 0.01,
            },
        ]
    )
    summary = summarize_strategy_frame(frame)
    assert not summary.empty
    text = render_evaluation_markdown(summary)
    assert "다기간·다전략 평가" in text
    assert "D+20" in text


def test_forward_slice_length():
    context, signal_date = _context(25)
    slice20 = forward_slice(context, signal_date, 20)
    assert len(slice20) == 20
