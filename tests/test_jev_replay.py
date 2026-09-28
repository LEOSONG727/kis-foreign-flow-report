import json
from pathlib import Path

import numpy as np
import pandas as pd

from kis_flow_bot.config import load_settings
from kis_flow_bot.jev_replay import (
    INPUT_COLUMNS,
    daily_jev_allowed,
    evaluate_replay,
    merge_replay,
    replay_shortlists,
    replicability_r2,
    run_jev_replay,
    run_replay_calls,
)
from kis_flow_bot.jev_shadow import MODEL, PROMPT_HASH

ROOT = Path(__file__).resolve().parents[1]


def _panel(dates=50, names=20, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(dates):
        for n in range(names):
            row = {
                "as_of": pd.Timestamp("2026-03-02") + pd.Timedelta(days=d),
                "date_index": d,
                "symbol": f"{n:06d}",
                "market": "KOSPI",
                "discovery_score": 100 - n,
                "timing_score": 50.0,
                "entry_score": 50.0,
                "foreign_net5": 1e9,
                "foreign_net20": 4e9,
                "foreign_buy_days10": 7,
                "sellback_ratio20": rng.uniform(0, 1),
                "foreign_20_adtv": rng.uniform(0, 2),
                "price_return5": rng.normal(0, 0.03),
                "price_return20": rng.normal(0, 0.08),
                "market_return20": 0.01,
                "ma20_distance": rng.normal(0, 0.03),
                "rebound_from_low20": rng.uniform(0, 0.2),
                "atr20_pct": rng.uniform(0.01, 0.05),
            }
            row["excess5"] = rng.normal(0, 0.03)
            row["excess20"] = -row["rebound_from_low20"] + rng.normal(0, 0.02)
            rows.append(row)
    return pd.DataFrame(rows)


class _Response:
    def __init__(self, risk, status=200):
        self.risk = risk
        self.status_code = status

    def json(self):
        return {
            "model": MODEL,
            "answers": {
                key: {"type": "noul", "noul": self.risk}
                for key in ("fragile_flow", "price_trap", "chase_risk")
            },
            "usage": {"input_tokens": 100},
        }


class _EchoSession:
    """Answers with a risk that is a pure function of the rebound it was shown."""

    def __init__(self, status=200):
        self.calls = 0
        self.status = status

    def post(self, url, **kwargs):
        self.calls += 1
        rebound = kwargs["json"]["state"]["entry"]["rebound_from_20_session_low"]
        risk = min(1.0, max(0.0, float(rebound.rstrip("%")) / 20.0))
        return _Response(risk, self.status)

    def close(self):
        pass


def test_shortlist_matches_live_selection_size_and_order():
    shortlist = replay_shortlists(_panel(dates=3))
    assert shortlist.groupby("as_of").size().eq(12).all()
    first = shortlist[shortlist["as_of"] == shortlist["as_of"].min()]
    assert first["discovery_score"].is_monotonic_decreasing


def test_calls_are_resumable_and_budgeted():
    shortlist = replay_shortlists(_panel(dates=3))
    session = _EchoSession()
    ledger = run_replay_calls(shortlist, pd.DataFrame(), api_key="k", max_calls=10, session=session, pause=lambda _: None)
    assert session.calls == 10
    assert ledger["status"].eq("OK").sum() == 10
    ledger = run_replay_calls(shortlist, ledger, api_key="k", max_calls=100, session=session, pause=lambda _: None)
    assert session.calls == 36
    assert len(ledger) == 36
    assert ledger["prompt_hash"].eq(PROMPT_HASH).all()


def test_auth_failure_stops_the_replay():
    shortlist = replay_shortlists(_panel(dates=2))
    session = _EchoSession(status=401)
    ledger = run_replay_calls(shortlist, pd.DataFrame(), api_key="k", max_calls=100, session=session, pause=lambda _: None)
    assert session.calls == 1
    assert ledger["status"].eq("API_ERROR").all()


def test_a_jev_that_only_echoes_its_inputs_adds_no_value():
    settings = load_settings(ROOT)
    shortlist = replay_shortlists(_panel())
    ledger = run_replay_calls(shortlist, pd.DataFrame(), api_key="k", max_calls=10_000, session=_EchoSession(), pause=lambda _: None)
    merged = merge_replay(shortlist, ledger)
    assert merged["as_of"].nunique() == 50
    # Pure function of one input: fully replicable by code.
    assert replicability_r2(merged) > 0.9
    result = evaluate_replay(merged, settings)
    # Raw answer tracks returns (rebound drives both) ...
    assert result["signals"]["jev_max"]["corr20"][0] < -0.3
    # ... but nothing is left once the inputs are controlled.
    assert abs(result["signals"]["jev_max"]["incremental"][0]) < 0.1
    assert result["verdict"] == "NO_VALUE"


def test_daily_calls_stop_only_for_the_same_prompt_and_model():
    assert daily_jev_allowed(None)
    assert daily_jev_allowed({"verdict": "INSUFFICIENT_DATA", "prompt_hash": PROMPT_HASH, "model": MODEL})
    assert not daily_jev_allowed({"verdict": "NO_VALUE", "prompt_hash": PROMPT_HASH, "model": MODEL})
    assert daily_jev_allowed({"verdict": "NO_VALUE", "prompt_hash": "older", "model": MODEL})


def test_run_without_key_evaluates_stored_answers(tmp_path):
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    result = run_jev_replay(settings, api_key="", panel=_panel(dates=5))
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert (tmp_path / "reports" / "jev_replay.md").exists()
    saved = json.loads((tmp_path / "data" / "jev_verdict.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == "INSUFFICIENT_DATA"
    assert set(INPUT_COLUMNS) <= set(_panel(dates=1).columns)


def test_a_jev_with_information_beyond_its_inputs_adds_value():
    settings = load_settings(ROOT)
    shortlist = replay_shortlists(_panel(dates=60, seed=5))
    rng = np.random.default_rng(1)
    # Answers carry the part of the return the ten inputs cannot explain.
    hidden = shortlist["excess20"] + shortlist["rebound_from_low20"]
    risk = (0.5 - 10 * hidden + rng.normal(0, 0.05, len(shortlist))).clip(0, 1)
    ledger = pd.DataFrame(
        {
            "as_of": shortlist["as_of"].dt.strftime("%Y-%m-%d"),
            "symbol": shortlist["symbol"],
            "prompt_hash": PROMPT_HASH,
            "status": "OK",
            "jev_fragile_flow": risk,
            "jev_price_trap": risk,
            "jev_chase_risk": risk,
        }
    )
    result = evaluate_replay(merge_replay(shortlist, ledger), settings)
    assert result["signals"]["jev_max"]["incremental"][2] < 0
    assert result["verdict"] == "ADDS_VALUE"
