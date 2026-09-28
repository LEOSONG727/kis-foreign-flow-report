from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kis_flow_bot.config import load_settings
from kis_flow_bot.paper import (
    PaperConfig,
    append_signals,
    live_signals,
    simulate,
    update_paper_trading,
)

ROOT = Path(__file__).resolve().parents[1]


def _prices(days=30, symbols=("000001", "000002"), drift=0.01):
    dates = pd.bdate_range("2026-10-01", periods=days)
    opens = pd.DataFrame(index=dates)
    closes = pd.DataFrame(index=dates)
    for i, symbol in enumerate(symbols):
        path = 100.0 * (1.0 + drift * (i + 1)) ** np.arange(days)
        opens[symbol] = path
        closes[symbol] = path * 1.001
    return opens, closes


def test_buys_next_open_holds_then_sells_with_costs():
    opens, closes = _prices()
    day0 = opens.index[0]
    signals = pd.DataFrame([{"as_of": day0, "strategy": "s", "rank": 1, "symbol": "000001"}])
    config = PaperConfig(slots=1, hold_sessions=5, cost_per_side=0.001, initial_equity=1_000_000)
    nav, trades = simulate(signals, opens, closes, config)
    trade = trades.iloc[0]
    assert trade["entry_date"] == opens.index[1]  # next open, never the signal day
    assert trade["exit_date"] == opens.index[6]  # after 5 closes
    gross = opens.iloc[6]["000001"] / opens.iloc[1]["000001"] - 1.0
    assert trade["return"] < gross  # costs were charged
    assert abs(trade["return"] - ((1 + gross) * 0.999 / 1.001 - 1)) < 1e-9


def test_full_slots_skip_new_picks_and_missing_open_cancels_buy():
    opens, closes = _prices()
    opens.loc[opens.index[1], "000002"] = np.nan  # suspended at the fill open
    day0 = opens.index[0]
    signals = pd.DataFrame(
        [
            {"as_of": day0, "strategy": "s", "rank": 1, "symbol": "000002"},
            {"as_of": day0, "strategy": "s", "rank": 2, "symbol": "000001"},
        ]
    )
    nav, trades = simulate(signals, opens, closes, PaperConfig(slots=1, hold_sessions=3))
    assert nav.iloc[1]["positions"] == 0  # the only slot's order was cancelled


def test_signals_are_frozen_on_first_capture():
    first = pd.DataFrame([{"as_of": "2026-10-01", "strategy": "s", "rank": 1, "symbol": "000001", "name": "a"}])
    rerun = first.assign(symbol="000002")
    ledger = append_signals(pd.DataFrame(), first, "t1")
    ledger = append_signals(ledger, rerun, "t2")
    assert list(ledger["symbol"]) == ["000001"]


def test_live_signals_cover_each_strategy_including_empty_official():
    scored = pd.DataFrame(
        {
            "as_of": ["2026-10-01"] * 6,
            "symbol": [f"{i:06d}" for i in range(6)],
            "name": list("abcdef"),
            "discovery_eligible": True,
            "discovery_score": [90, 80, 70, 60, 50, 40],
            "timing_score": 50,
            "entry_score": 50,
            "proposal_status": ["관찰"] * 6,
            "learned_rank": [6, 5, 4, 3, 2, 1],
            "kw_dilution": [True, False, False, False, False, False],
            "kw_adverse": False,
        }
    )
    signals = live_signals(scored, picks=2)
    by = signals.groupby("strategy")["symbol"].apply(list).to_dict()
    assert by["official"] == [""]  # a recorded cash day
    assert by["discovery_top5"] == ["000000", "000001"]
    assert by["learned_top5"] == ["000005", "000004"]
    assert by["filing_veto_top5"] == ["000001", "000002"]


def test_update_writes_forward_report(tmp_path):
    (tmp_path / "config.toml").write_text(
        (ROOT / "config.toml").read_text(encoding="utf-8").replace('start_date = "2026-09-29"', 'start_date = "2026-01-01"'),
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    opens, closes = _prices(days=10)
    history = []
    for day in opens.index:
        for symbol in opens.columns:
            history.append({"date": day, "symbol": symbol, "open": opens.loc[day, symbol], "close": closes.loc[day, symbol]})
    history = pd.DataFrame(history)
    scored = pd.DataFrame(
        {
            "as_of": [opens.index[0].strftime("%Y-%m-%d")] * 2,
            "symbol": list(opens.columns),
            "discovery_eligible": True,
            "discovery_score": [90, 80],
            "timing_score": 50,
            "entry_score": 50,
        }
    )
    summary = update_paper_trading(scored, history, settings, datetime(2026, 10, 15))
    assert "discovery_top5" in set(summary["strategy"])
    assert (tmp_path / "reports" / "paper_trading.md").exists()
    assert (tmp_path / "data" / "paper_signals.csv").exists()
