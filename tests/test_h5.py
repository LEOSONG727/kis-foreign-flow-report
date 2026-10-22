from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kis_flow_bot.config import load_settings
from kis_flow_bot.hypotheses import (
    H5_FLOW_COLUMN,
    H5_REGISTERED_ON,
    attach_h5_flow,
    h5_cap_scale,
    h5_flow,
    h5_select,
    h5_window,
)
from kis_flow_bot.paper import live_signals, update_paper_trading
from kis_flow_bot.research import evaluate_h5, h5_series

ROOT = Path(__file__).resolve().parents[1]


def _day(as_of: str, n: int = 60) -> pd.DataFrame:
    # Flow rises with i; price return falls with i inside each flow block.
    i = np.arange(n)
    return pd.DataFrame(
        {
            "as_of": as_of,
            "symbol": [f"{k:06d}" for k in i],
            "name": [f"s{k}" for k in i],
            H5_FLOW_COLUMN: i.astype(float),
            "price_return20": ((i * 7) % n) / n - 0.5,
        }
    )


def test_flow_is_share_of_market_cap_and_missing_cap_is_missing():
    flow = h5_flow(pd.Series([10.0, 10.0, 10.0]), pd.Series([1000.0, 0.0, np.nan]))
    assert flow.iloc[0] == 0.01
    assert flow.iloc[1:].isna().all()


def test_cap_scale_carries_master_shares_to_past_closes():
    universe = pd.DataFrame({"symbol": ["1", "000002"], "market_cap_eok": [10.0, 20.0]})
    closes = pd.DataFrame({"000001": [50.0, 100.0], "000002": [200.0, np.nan]})
    scale = h5_cap_scale(universe, closes)
    assert scale["000001"] == 10.0 * 1e8 / 100.0
    assert scale["000002"] == 20.0 * 1e8 / 200.0  # last *known* close
    frame = pd.DataFrame({"symbol": ["000001"], "close": [50.0], "foreign_net60": [5e8]})
    out = attach_h5_flow(frame, scale)
    assert abs(out[H5_FLOW_COLUMN].iloc[0] - 5e8 / (10.0 * 1e8 * 0.5)) < 1e-12


def test_select_keeps_top_flow_quintile_and_opposite_price_thirds():
    frame = _day("2026-10-20")
    flow_rank = frame[H5_FLOW_COLUMN].rank(pct=True)
    price_rank = frame["price_return20"].rank(pct=True)
    confirm = h5_select(frame, "confirm", top_k=10)
    lag = h5_select(frame, "lag", top_k=10)
    assert len(confirm) and len(lag)
    assert set(confirm["symbol"]).isdisjoint(lag["symbol"])
    assert (flow_rank[confirm.index] >= 0.80).all() and (price_rank[confirm.index] >= 2 / 3).all()
    assert (flow_rank[lag.index] >= 0.80).all() and (price_rank[lag.index] < 1 / 3).all()
    # Ordered by flow, strongest first.
    assert list(confirm[H5_FLOW_COLUMN]) == sorted(confirm[H5_FLOW_COLUMN], reverse=True)


def test_select_ranks_within_each_date():
    frame = pd.concat([_day("2026-10-20"), _day("2026-10-21").assign(**{H5_FLOW_COLUMN: lambda f: f[H5_FLOW_COLUMN] * 1e6})])
    picks = h5_select(frame, "confirm", top_k=3)
    assert picks.groupby("as_of").size().to_dict() == {"2026-10-20": 3, "2026-10-21": 3}


def test_window_counts_registration_day_as_discovery():
    dates = pd.Series(pd.to_datetime([H5_REGISTERED_ON, "2026-10-20"]))
    assert list(h5_window(dates)) == ["discovery", "confirmation"]


def _labelled(start: str, days: int, confirm_edge: float) -> pd.DataFrame:
    rows = []
    for k, day in enumerate(pd.bdate_range(start, periods=days)):
        frame = _day(day)
        rng = np.random.default_rng(k)
        price_rank = frame["price_return20"].rank(pct=True)
        noise = rng.normal(0.0, 0.01, len(frame))
        frame["excess20"] = np.where(price_rank >= 2 / 3, confirm_edge, -confirm_edge) + noise
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_h5_needs_forward_dates_and_both_lower_bounds():
    settings = load_settings(ROOT)
    only_discovery = _labelled("2026-04-01", 80, 0.03)
    result = evaluate_h5(only_discovery, settings)
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert result["discovery"]["spread"]["mean"] > 0.05

    forward = _labelled("2026-10-20", 70, 0.03)
    assert evaluate_h5(forward, settings)["verdict"] == "CONFIRMED"
    assert evaluate_h5(_labelled("2026-10-20", 70, -0.03), settings)["verdict"] == "REVERSED"


def test_h5_series_reports_both_baskets_per_date():
    series = h5_series(_labelled("2026-10-20", 3, 0.02), "excess20")
    assert len(series) == 3
    assert (series["spread"] == series["confirm"] - series["lag"]).all()


def test_live_signals_add_h5_pair_only_when_flow_is_known():
    scored = _day("2026-10-20").assign(discovery_eligible=True, discovery_score=50.0, timing_score=50.0, entry_score=50.0)
    by = live_signals(scored, picks=2).groupby("strategy")["symbol"].apply(list).to_dict()
    assert len(by["h5_confirm_top5"]) == 2 and len(by["h5_lag_top5"]) == 2
    without = live_signals(scored.drop(columns=[H5_FLOW_COLUMN]), picks=2)
    assert not without["strategy"].str.startswith("h5").any()


def _history(days: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-01", periods=days)
    rows = []
    for symbol, drift in (("000001", 0.01), ("000002", 0.0)):
        path = 100.0 * (1.0 + drift) ** np.arange(days)
        for day, price in zip(dates, path):
            rows.append({"date": day, "symbol": symbol, "open": price, "close": price})
    return pd.DataFrame(rows)


def test_strategy_added_later_is_measured_from_its_own_start(tmp_path):
    (tmp_path / "config.toml").write_text(
        (ROOT / "config.toml").read_text(encoding="utf-8").replace('start_date = "2026-09-29"', 'start_date = "2026-01-01"'),
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    history = _history()
    dates = sorted(history["date"].unique())
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    ledger = pd.DataFrame(
        [
            {"as_of": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"), "strategy": "discovery_top5", "rank": 1,
             "symbol": "000002", "name": "flat", "captured_at": "t0"},
            {"as_of": pd.Timestamp(dates[10]).strftime("%Y-%m-%d"), "strategy": "h5_confirm_top5", "rank": 1,
             "symbol": "000001", "name": "up", "captured_at": "t1"},
        ]
    )
    ledger.to_csv(data / "paper_signals.csv", index=False)
    pd.DataFrame({"symbol": ["000001", "000002"], "market_cap_eok": [100.0, 100.0]}).to_csv(data / "universe.csv", index=False)
    scored = pd.DataFrame(
        {"as_of": [pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")] * 2, "symbol": ["000001", "000002"],
         "discovery_eligible": True, "discovery_score": [90, 80], "timing_score": 50, "entry_score": 50}
    )
    summary = update_paper_trading(scored, history, settings, datetime(2026, 11, 1)).set_index("strategy")
    assert summary.loc["h5_confirm_top5", "start"] == pd.Timestamp(dates[10]).strftime("%Y-%m-%d")
    assert summary.loc["discovery_top5", "start"] == pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    # Its own span: fewer days than the original strategy, and no idle cash before it existed.
    assert summary.loc["h5_confirm_top5", "days"] < summary.loc["discovery_top5", "days"]
    assert summary.loc["h5_confirm_top5", "return"] > 0
    report = (tmp_path / "reports" / "paper_trading.md").read_text(encoding="utf-8")
    assert "부터 기록" in report
