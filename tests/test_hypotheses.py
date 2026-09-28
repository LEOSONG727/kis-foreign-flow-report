from pathlib import Path

import numpy as np
import pandas as pd

from kis_flow_bot.config import load_settings
from kis_flow_bot.hypotheses import HYPOTHESES, hypothesis_score, verdict, window_of
from kis_flow_bot.research import evaluate_hypotheses

ROOT = Path(__file__).resolve().parents[1]


def test_windows_split_on_the_registered_discovery_dates():
    dates = pd.Series(pd.to_datetime(["2025-11-03", "2026-04-06", "2026-08-26", "2026-09-30"]))
    assert list(window_of(dates)) == ["confirmation", "discovery", "discovery", "confirmation"]


def test_scores_are_within_date_ranks_and_missing_is_neutral():
    frame = pd.DataFrame(
        {
            "as_of": ["d1"] * 3 + ["d2"] * 3,
            "op_yield": [0.1, 0.2, np.nan, 5.0, 6.0, 7.0],
        }
    )
    only_yield = next(h for h in HYPOTHESES if h.id == "H1b-op-yield-only")
    score = hypothesis_score(frame, only_yield)
    assert score.iloc[2] == 0.0  # missing → neutral
    # Same ordering within each day regardless of the raw scale.
    assert score.iloc[3] < score.iloc[4] < score.iloc[5]


def test_verdict_requires_confirmation_sample_and_both_lower_bounds():
    good = {"dates": 80, "mean": 0.05, "low": 0.01, "high": 0.09}
    weak = {"dates": 80, "mean": 0.02, "low": -0.01, "high": 0.05}
    short = {"dates": 20, "mean": 0.05, "low": 0.01, "high": 0.09}
    assert verdict(short, good) == "INSUFFICIENT_DATA"
    assert verdict(good, good) == "CONFIRMED"
    assert verdict(good, weak) == "NOT_CONFIRMED"
    assert verdict({"dates": 80, "low": -0.09, "high": -0.01}, weak) == "REVERSED"


def test_discovery_dates_never_count_as_confirmation():
    settings = load_settings(ROOT)
    rng = np.random.default_rng(0)
    rows = []
    # Signal only inside the discovery window; pure noise elsewhere.
    for day in pd.bdate_range("2025-10-01", "2026-08-26"):
        inside = day >= pd.Timestamp("2026-04-06")
        for n in range(40):
            yield_value = rng.normal()
            rows.append(
                {
                    "as_of": day,
                    "symbol": f"{n:06d}",
                    "market": "KOSPI",
                    "op_yield": yield_value,
                    "pit_size": rng.normal(),
                    "atr20_pct": rng.normal(),
                    "price_return60": rng.normal(),
                    "excess20": (yield_value if inside else 0.0) + rng.normal(),
                }
            )
    result = evaluate_hypotheses(pd.DataFrame(rows), settings)
    only_yield = next(item for item in result if item["id"] == "H1b-op-yield-only")
    assert only_yield["discovery"]["ic"]["mean"] > 0.3
    assert abs(only_yield["confirmation"]["ic"]["mean"]) < 0.1
    assert only_yield["verdict"] != "CONFIRMED"
