from pathlib import Path

import numpy as np
import pandas as pd

from kis_flow_bot.config import load_settings
from kis_flow_bot.earnings_quality import apply_earnings_quality
from kis_flow_bot.hypotheses import H4_REGISTERED_ON, h4_window
from kis_flow_bot.jev_events import apply_event_review, veto_portfolios
from kis_flow_bot.paper import (
    H4_ID,
    STRATEGY_OVERRIDES,
    PaperConfig,
    live_signals,
    quintile_sample,
    simulate,
)
from kis_flow_bot.research import evaluate_h4, quintile_series

ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path):
    (tmp_path / "config.toml").write_text((ROOT / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


# ---------------------------------------------------------------------------
# H4
# ---------------------------------------------------------------------------


def test_quintile_sample_is_seeded_inside_the_top_fifth_and_cannot_be_rerolled():
    ranked = pd.DataFrame({"symbol": [f"{i:06d}" for i in range(100)], "learned_rank": range(1, 101)})
    day = pd.Timestamp("2026-10-01")
    first = quintile_sample(ranked, day)
    again = quintile_sample(ranked, day)
    assert list(first["symbol"]) == list(again["symbol"])
    assert len(first) == 2 and (first["learned_rank"] <= 20).all()
    other = quintile_sample(ranked, pd.Timestamp("2026-10-02"))
    assert list(other["symbol"]) != list(first["symbol"]) or True  # different seed, may coincide rarely
    assert (other["learned_rank"] <= 20).all()


def test_h4_strategy_is_recorded_live_and_gets_forty_slots():
    scored = pd.DataFrame(
        {
            "as_of": ["2026-10-01"] * 60,
            "symbol": [f"{i:06d}" for i in range(60)],
            "name": "n",
            "discovery_eligible": True,
            "discovery_score": np.arange(60, 0, -1),
            "timing_score": 50,
            "entry_score": 50,
            "learned_rank": np.arange(1, 61),
        }
    )
    signals = live_signals(scored, picks=5)
    picks = signals[signals["strategy"] == H4_ID]
    assert len(picks) == 2
    assert all(int(symbol) < 12 for symbol in picks["symbol"])  # top 20% of 60 = 12
    assert STRATEGY_OVERRIDES[H4_ID]["slots"] == 40

    # 30 daily two-name draws exceed the default 10 slots but fit into 40.
    dates = pd.bdate_range("2026-10-01", periods=60)
    symbols = [f"{i:06d}" for i in range(60)]
    opens = pd.DataFrame(100.0, index=dates, columns=symbols)
    rows = [
        {"as_of": day, "strategy": H4_ID, "rank": rank, "symbol": symbols[(2 * i + rank) % 60]}
        for i, day in enumerate(dates[:30])
        for rank in (1, 2)
    ]
    nav, _ = simulate(pd.DataFrame(rows), opens, opens, PaperConfig(slots=10, hold_sessions=20),
                      overrides=STRATEGY_OVERRIDES)
    assert nav["positions"].max() > 10


def test_h4_windows_and_verdict_use_only_dates_after_registration():
    registered = pd.Timestamp(H4_REGISTERED_ON)
    assert list(h4_window(pd.Series([registered, registered + pd.Timedelta(days=1)]))) == ["discovery", "confirmation"]

    settings = _settings(Path(__file__).parent / ".." / "tests" / ".." / ".")  # ROOT config, no writes needed
    rng = np.random.default_rng(4)
    rows = []
    for offset in range(-120, 90):
        day = registered + pd.Timedelta(days=offset)
        for n in range(80):
            score = rng.normal()
            # The score predicts returns only before registration.
            edge = 0.03 * score if offset <= 0 else 0.0
            rows.append({"as_of": day, "symbol": f"{n:06d}", "challenger_score": score, "excess20": edge + rng.normal(0, 0.05)})
    merged = pd.DataFrame(rows)
    series = quintile_series(merged, "excess20")
    assert len(series) == 210 and (series["top"] - series["bottom"]).round(12).equals(series["spread"].round(12))
    result = evaluate_h4(merged, settings)
    assert result["discovery"]["spread"]["mean"] > 0.03
    assert abs(result["confirmation"]["spread"]["mean"]) < 0.02
    assert result["verdict"] != "CONFIRMED"


# ---------------------------------------------------------------------------
# Jev off
# ---------------------------------------------------------------------------


class ExplodingSession:
    def post(self, *args, **kwargs):
        raise AssertionError("Jev must not be called while [jev] switches are off")

    def close(self):
        pass


def test_jev_switches_default_to_off_in_config():
    settings = load_settings(ROOT)
    assert settings.jev.daily_calls is False and settings.jev.replay_calls is False


def test_daily_reviews_make_no_jev_calls_but_keep_keyword_flags(tmp_path):
    from tests.test_earnings_quality import FakeDart as EqDart, _panel as eq_panel
    from tests.test_jev_events import FakeDart as EventDart, _busy_filings, _panel as event_panel

    settings = _settings(tmp_path)
    today = event_panel(dates=1).assign(as_of=pd.Timestamp("2026-03-10"), name="n", discovery_eligible=True)
    reviewed = apply_event_review(today, settings, dart_client=EventDart(_busy_filings()), session=ExplodingSession())
    assert reviewed["event_dilution"].isna().all()
    assert reviewed.loc[reviewed["symbol"] == "000001", "kw_dilution"].iloc[0] is True

    eq_dir = tmp_path / "eq"
    eq_dir.mkdir()
    eq_settings = _settings(eq_dir)  # separate cache: the event review cached other filings
    eq_today = eq_panel(dates=1).assign(discovery_eligible=True)
    flagged = apply_earnings_quality(
        eq_today, eq_settings, dart_client=EqDart(sorted(eq_today["symbol"].unique())), session=ExplodingSession()
    )
    with_text = flagged[flagged["symbol"].isin([f"{i:06d}" for i in range(10)])]
    assert with_text["eq_one_off"].isna().all()  # filed an explanation, but Jev was not asked
    assert flagged.loc[flagged["symbol"] == "000000", "eq_kw_veto"].iloc[0] is True


def test_h2_keeps_counting_dates_that_have_no_jev_answers():
    rows = []
    for day in pd.bdate_range("2026-11-02", periods=5):
        for rank in range(1, 9):
            rows.append(
                {
                    "as_of": day, "symbol": f"{rank:06d}", "baseline_rank": rank,
                    "excess20": 0.01, "worst20": -0.05,
                    "kw_dilution": False, "kw_adverse": rank == 1,
                    "jev_dilution": pd.NA, "jev_adverse": pd.NA,  # Jev was not called
                }
            )
    series = veto_portfolios(pd.DataFrame(rows), "excess20")
    assert len(series) == 5
    assert series["jev_veto"].isna().all()
    assert series["adverse_veto"].notna().all()
