from __future__ import annotations

import pandas as pd

from kis_flow_bot.brief import annotate_watch_brief
from kis_flow_bot.config import ScreeningConfig
from kis_flow_bot.eligibility import apply_candidate_eligibility


def test_candidate_eligibility_blocks_stale_or_illiquid_rows():
    scored = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "as_of": "2026-08-21",
                "adtv20": 2_000_000_000,
                "data_points": 60,
            },
            {
                "symbol": "000002",
                "as_of": "2026-08-20",
                "adtv20": 2_000_000_000,
                "data_points": 60,
            },
            {
                "symbol": "000003",
                "as_of": "2026-08-21",
                "adtv20": 500_000_000,
                "data_points": 60,
            },
        ]
    )

    result = apply_candidate_eligibility(
        scored,
        ScreeningConfig(),
        report_date="2026-08-21",
    ).set_index("symbol")

    assert bool(result.loc["000001", "discovery_eligible"])
    assert not bool(result.loc["000002", "discovery_eligible"])
    assert "당일 수급" in result.loc["000002", "eligibility_reason"]
    assert not bool(result.loc["000003", "discovery_eligible"])
    assert "거래대금" in result.loc["000003", "eligibility_reason"]


def test_stale_watch_row_can_be_displayed_but_not_proposed():
    row = {
        "symbol": "000001",
        "name": "관찰주",
        "as_of": "2026-08-20",
        "adtv20": 2_000_000_000,
        "data_points": 60,
        "discovery_score": 90.0,
        "timing_score": 80.0,
        "distribution_score": 5.0,
        "campaign_regime": "지속매집",
        "stage": "매집·눌림",
        "action": "초기 분할매수",
        "decision_bucket": "신규매수",
    }
    eligible = apply_candidate_eligibility(
        pd.DataFrame([row]),
        ScreeningConfig(),
        report_date="2026-08-21",
    )
    annotated = annotate_watch_brief(eligible)

    assert annotated.iloc[0]["watch_tier"] == "제외"
    assert annotated.iloc[0]["proposal_status"] == "제외"
    assert "당일 수급" in annotated.iloc[0]["block_reason"]
