from datetime import date, timedelta

import numpy as np
import pandas as pd

from kis_flow_bot import research
from kis_flow_bot.pipeline import _backfill_symbol_older
from kis_flow_bot.research import (
    LEARNED_FEATURES,
    SPEC_VERSION,
    SeriesSummary,
    apply_learned_ranker,
    challenger_decision,
    champion_decision,
    daily_rank_ic,
    partial_rank_ic,
    summarize_series,
    top_k_excess,
    walk_forward_predictions,
)


def _panel(dates: int = 6, names: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(dates):
        as_of = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
        for n in range(names):
            size = rng.normal()
            rows.append(
                {
                    "as_of": as_of,
                    "date_index": d,
                    "symbol": f"{n:06d}",
                    "market": "KOSPI",
                    "pit_size": size,
                    "atr20_pct": rng.normal(),
                    "price_return60": rng.normal(),
                    "signal": float(n),
                    "size_proxy": size,
                    "excess20": size + 0.01 * rng.normal(),
                    "noise_target": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


def test_daily_rank_ic_is_one_for_perfect_ordering():
    panel = _panel()
    panel["excess20"] = panel["signal"]
    ic = daily_rank_ic(panel, "signal", "excess20")
    assert len(ic) == 6
    assert np.allclose(ic.to_numpy(), 1.0)


def test_partial_ic_strips_a_pure_size_tilt():
    panel = _panel(dates=8, names=200)
    rng = np.random.default_rng(3)
    # A "flow" score that is really size plus unrelated noise.
    panel["tilted_score"] = panel["pit_size"] + 0.3 * rng.normal(size=len(panel))
    panel["excess20"] = panel["pit_size"] + 0.3 * rng.normal(size=len(panel))
    raw = daily_rank_ic(panel, "tilted_score", "excess20")
    partial = partial_rank_ic(panel, "tilted_score", "excess20", controls=("pit_size",))
    assert raw.mean() > 0.6
    assert abs(partial.mean()) < 0.1


def test_top_k_excess_takes_highest_scores_per_date():
    panel = _panel(dates=2, names=30)
    panel["excess20"] = panel["signal"] / 100.0
    top = top_k_excess(panel, "signal", "excess20", top_k=3)
    assert np.allclose(top.to_numpy(), (29 + 28 + 27) / 3 / 100.0)


def test_walk_forward_only_trains_on_observable_labels(monkeypatch):
    rows = []
    for d in range(12):
        for n in range(25):
            row = {name: float((n * 7 + d) % 25) for name in LEARNED_FEATURES}
            row.update(
                {
                    "as_of": pd.Timestamp("2026-02-02") + pd.Timedelta(days=d),
                    "date_index": 100 + d,
                    "symbol": f"{n:06d}",
                    "excess5": float(n),
                }
            )
            rows.append(row)
    panel = pd.DataFrame(rows)
    seen: list[tuple[int, int]] = []

    def fake_fit(frames, target, alpha, features=LEARNED_FEATURES):
        seen.append((len(frames), max(int(frame["date_index"].iloc[0]) for frame in frames)))
        return np.zeros(len(features))

    monkeypatch.setattr(research, "fit_ridge", fake_fit)
    predictions = walk_forward_predictions(
        panel, horizon=5, retrain_every=1, min_train_dates=3, alpha=20.0
    )
    predicted_dates = sorted(predictions["date_index"].unique())
    # First prediction needs 3 known dates: 100..102 are known on 107.
    assert predicted_dates[0] == 107
    for (count, newest), target_index in zip(seen, predicted_dates):
        assert newest <= target_index - 5
        assert count == target_index - 5 - 100 + 1


def test_gate_requires_enough_oos_dates_and_positive_lower_bounds():
    good = SeriesSummary(150, 0.04, 0.01, 0.07, 0.7, 2.5)
    weak = SeriesSummary(150, 0.02, -0.01, 0.05, 0.6, 1.0)
    short = SeriesSummary(40, 0.05, 0.02, 0.08, 0.7, 2.0)
    top_good = SeriesSummary(150, 0.01, -0.01, 0.03, 0.55, 1.0)
    assert challenger_decision(short, good, top_good, 120)[0] == "INSUFFICIENT_DATA"
    assert challenger_decision(weak, good, top_good, 120)[0] == "NO_EDGE"
    assert challenger_decision(good, weak, top_good, 120)[0] == "NOT_BETTER"
    assert challenger_decision(good, good, top_good, 120)[0] == "PROMOTE"
    assert champion_decision(short, 120) == "INSUFFICIENT_DATA"
    assert champion_decision(good, 120) == "VALIDATED"
    assert champion_decision(SeriesSummary(150, -0.05, -0.08, -0.01, 0.2, -2.0), 120) == "NEGATIVE"


def test_summarize_series_effective_t_uses_block_length():
    values = pd.Series([0.02, 0.03, 0.01, 0.02] * 10)
    one = summarize_series(values, block=1, samples=200)
    twenty = summarize_series(values, block=20, samples=200)
    assert one.dates == twenty.dates == 40
    assert twenty.effective_t < one.effective_t
    assert one.low > 0


def test_apply_learned_ranker_scores_only_eligible_rows():
    rows = []
    for n in range(30):
        row = {name: 0.0 for name in LEARNED_FEATURES}
        row.update({"symbol": f"{n:06d}", "name": f"S{n}", "discovery_eligible": n < 25})
        row["atr20_pct"] = float(n)
        rows.append(row)
    scored = pd.DataFrame(rows)
    ranker = {
        "spec_version": SPEC_VERSION,
        "features": list(LEARNED_FEATURES),
        "weights": {"atr20_pct": -1.0},
        "challenger": {"status": "INSUFFICIENT_DATA"},
    }
    result = apply_learned_ranker(scored, ranker)
    assert result.loc[result["discovery_eligible"], "learned_rank"].notna().all()
    assert result.loc[~result["discovery_eligible"], "learned_rank"].isna().all()
    best = result.sort_values("learned_rank").iloc[0]
    assert best["symbol"] == "000000"
    assert (result["learned_status"] == "INSUFFICIENT_DATA").all()


def test_apply_learned_ranker_without_model_leaves_scores_blank():
    scored = pd.DataFrame({"symbol": ["000001"], "discovery_eligible": [True]})
    result = apply_learned_ranker(scored, None)
    assert result["learned_score"].isna().all()


class _FakeClient:
    def __init__(self, first_available: date):
        self.first_available = first_available
        self.calls: list[date] = []

    def investor_daily(self, symbol: str, as_of: date):
        self.calls.append(as_of)
        rows = []
        day = as_of
        while len(rows) < 30 and day >= self.first_available:
            if day.weekday() < 5:
                rows.append(
                    {
                        "stck_bsop_date": day.strftime("%Y%m%d"),
                        "stck_oprc": "100",
                        "stck_hgpr": "101",
                        "stck_lwpr": "99",
                        "stck_clpr": "100",
                        "acml_vol": "1000",
                        "acml_tr_pbmn": "100000",
                        "frgn_ntby_qty": "10",
                        "frgn_ntby_tr_pbmn": "1000",
                        "orgn_ntby_qty": "0",
                        "orgn_ntby_tr_pbmn": "0",
                    }
                )
            day -= timedelta(days=1)
        return rows


def test_deep_backfill_walks_backwards_until_target_or_listing():
    client = _FakeClient(first_available=date(2025, 11, 3))
    earliest = date(2026, 1, 7)
    frame = _backfill_symbol_older(
        client, "000001", "S", "KOSPI", earliest, target=date(2025, 6, 1)
    )
    assert not frame.empty
    dates = pd.to_datetime(frame["date"])
    assert dates.max().date() < earliest
    assert dates.min().date() == date(2025, 11, 3)
    # Each call anchors on the day before the oldest row received.
    assert all(later < earlier for earlier, later in zip(client.calls, client.calls[1:]))
    assert len(client.calls) <= 6


def test_all_missing_feature_has_empty_ic_and_summary():
    panel = _panel()
    panel["empty_feature"] = np.nan
    ic = daily_rank_ic(panel, "empty_feature", "excess20")
    assert ic.empty
    summary = summarize_series(ic, block=20, samples=50)
    assert summary.dates == 0
    assert partial_rank_ic(panel, "empty_feature", "excess20", controls=("pit_size",)).empty
