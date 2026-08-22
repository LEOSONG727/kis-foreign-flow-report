from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from kis_flow_bot.config import (
    APIConfig,
    BriefConfig,
    ModelConfig,
    ScreeningConfig,
    Settings,
    TrackingConfig,
)
from kis_flow_bot.live_tracking import (
    EVENT_COLUMNS,
    SNAPSHOT_COLUMNS,
    append_official_snapshots,
    build_daily_snapshots,
    refresh_outcomes,
    update_episode_ledger,
    update_live_tracking,
)


def _settings(tmp_path) -> Settings:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "reports").mkdir(exist_ok=True)
    return Settings(
        root=tmp_path,
        screening=ScreeningConfig(),
        api=APIConfig(),
        model=ModelConfig(),
        brief=BriefConfig(),
        tracking=TrackingConfig(),
        watchlist=(),
    )


def _scored_row(**overrides) -> dict:
    row = {
        "as_of": "2026-08-21",
        "symbol": "000001",
        "name": "후보주",
        "market": "KOSPI",
        "close": 100.0,
        "discovery_grade": "A",
        "proposal_status": "강한 후보·가격 대기",
        "proposal_priority": 2,
        "discovery_score": 84.0,
        "timing_score": 55.0,
        "distribution_score": 8.0,
        "stage": "매집·눌림",
        "campaign_regime": "지속매집",
        "watch_tier": "고확신눌림",
        "entry_readiness": "눌림대기",
        "entry_gate_passed": False,
        "foreign_1_adtv": 0.04,
        "foreign_3_adtv": 0.16,
        "foreign_5_adtv": 0.25,
        "foreign_20_adtv": 0.80,
        "foreign_buy_days10": 8,
        "sellback_ratio20": 0.15,
        "price_return3": 0.01,
        "price_return20": 0.02,
        "rebound_from_low20": 0.04,
        "data_fresh": True,
        "liquidity_eligible": True,
        "history_eligible": True,
        "discovery_eligible": True,
    }
    row.update(overrides)
    return row


def _snapshot(date: str, proposal: str, grade: str = "A") -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            _scored_row(
                as_of=date,
                proposal_status=proposal,
                discovery_grade=grade,
            )
        ]
    )
    captured = datetime(2026, 8, 21, 20, 35, tzinfo=ZoneInfo("Asia/Seoul"))
    return build_daily_snapshots(frame, captured, "test-v1", "abc123")


def test_snapshot_keeps_only_eligible_abc_and_is_immutable_on_same_date():
    captured = datetime(2026, 8, 21, 20, 35, tzinfo=ZoneInfo("Asia/Seoul"))
    scored = pd.DataFrame(
        [
            _scored_row(),
            _scored_row(symbol="000002", discovery_grade="C", discovery_score=68.0),
            _scored_row(symbol="000003", discovery_grade="제외", discovery_score=55.0),
            _scored_row(symbol="000004", discovery_eligible=False),
        ]
    )
    incoming = build_daily_snapshots(scored, captured, "test-v1", "abc123")
    assert list(incoming["symbol"]) == ["000001", "000002"]

    first = append_official_snapshots(pd.DataFrame(columns=SNAPSHOT_COLUMNS), incoming)
    changed = incoming.copy()
    changed["discovery_score"] = 99.0
    rerun = append_official_snapshots(first, changed)

    assert len(rerun) == 2
    assert set(rerun["discovery_score"]) == {84.0, 68.0}


def test_episode_groups_repeats_and_records_first_buy_review_promotion():
    trading_dates = list(pd.bdate_range("2026-08-21", periods=9))
    episodes = pd.DataFrame()
    events = pd.DataFrame(columns=EVENT_COLUMNS)

    snapshots = _snapshot("2026-08-21", "강한 후보·가격 대기")
    episodes, events = update_episode_ledger(
        snapshots, episodes, events, "2026-08-21", trading_dates, 5
    )
    assert len(episodes) == 1
    assert list(events["event_type"]) == ["candidate_entry"]

    promoted = _snapshot("2026-08-24", "오늘 신규매수 검토", grade="B")
    snapshots = pd.concat([snapshots, promoted], ignore_index=True)
    episodes, events = update_episode_ledger(
        snapshots, episodes, events, "2026-08-24", trading_dates, 5
    )
    assert len(episodes) == 1
    assert episodes.iloc[0]["first_buy_review_date"] == "2026-08-24"
    assert set(events["event_type"]) == {"candidate_entry", "buy_review_promotion"}

    episodes, events = update_episode_ledger(
        snapshots, episodes, events, "2026-08-24", trading_dates, 5
    )
    assert len(events) == 2

    episodes, events = update_episode_ledger(
        snapshots, episodes, events, "2026-08-31", trading_dates, 5
    )
    assert episodes.iloc[0]["closed_at"] == "2026-08-31"

    reentry = _snapshot("2026-09-01", "강한 후보·가격 대기")
    snapshots = pd.concat([snapshots, reentry], ignore_index=True)
    episodes, events = update_episode_ledger(
        snapshots, episodes, events, "2026-09-01", trading_dates, 5
    )
    assert len(episodes) == 2


def _history_for_outcome(ambiguous: bool = False) -> pd.DataFrame:
    rows = [
        {
            "date": "2026-08-21",
            "symbol": "000001",
            "name": "후보주",
            "market": "KOSPI",
            "open": 100,
            "high": 1000,  # signal-day movement must never affect the outcome
            "low": 90,
            "close": 100,
        }
    ]
    for index, date in enumerate(pd.bdate_range("2026-08-24", periods=20), 1):
        close = 100 + index * 0.25
        rows.append(
            {
                "date": date,
                "symbol": "000001",
                "name": "후보주",
                "market": "KOSPI",
                "open": 100 if index == 1 else close - 0.1,
                "high": 109 if ambiguous and index == 1 else close + 0.3,
                "low": 93 if ambiguous and index == 1 else close - 0.3,
                "close": close,
            }
        )
        rows.append(
            {
                "date": date,
                "symbol": "000002",
                "name": "시장비교",
                "market": "KOSPI",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100 + index * 0.05,
            }
        )
    return pd.DataFrame(rows)


def _event() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "000001-20260821:candidate_entry:20260821",
                "episode_id": "000001-20260821",
                "event_date": "2026-08-21",
                "event_type": "candidate_entry",
                "symbol": "000001",
                "name": "후보주",
                "market": "KOSPI",
                "reference_close": 100.0,
                "discovery_grade": "A",
                "proposal_status": "강한 후보·가격 대기",
                "model_version": "test-v1",
                "config_hash": "abc123",
            }
        ]
    )


def test_outcome_uses_next_session_open_and_future_rows_only(tmp_path):
    outcomes = refresh_outcomes(_event(), pd.DataFrame(), _history_for_outcome(), _settings(tmp_path))
    row = outcomes.iloc[0]

    assert row["entry_date"] == "2026-08-24"
    assert row["entry_open"] == 100
    assert row["observed_sessions"] == 20
    assert row["return_5"] == pytest.approx(0.0125)
    assert row["net_return_5"] == pytest.approx(0.0095)
    assert row["signal_close_return_20"] == pytest.approx(0.05)
    assert row["event_outcome"] == "horizon"
    assert row["status"] == "resolved"


def test_same_day_target_and_stop_is_marked_ambiguous(tmp_path):
    outcomes = refresh_outcomes(
        _event(), pd.DataFrame(), _history_for_outcome(ambiguous=True), _settings(tmp_path)
    )
    assert outcomes.iloc[0]["event_outcome"] == "ambiguous"
    assert outcomes.iloc[0]["days_to_event"] == 1


def test_live_tracking_writes_files_and_does_not_rewrite_official_run(tmp_path):
    settings = _settings(tmp_path)
    generated = datetime(2026, 8, 21, 20, 35, tzinfo=ZoneInfo("Asia/Seoul"))
    scored = pd.DataFrame([_scored_row()])
    history = _history_for_outcome().iloc[:1].copy()

    first = update_live_tracking(scored, history, settings, generated)
    assert first is not None
    assert first.episodes == 1
    assert (tmp_path / "data" / "signal_runs.csv").exists()
    assert (tmp_path / "data" / "signal_snapshots.csv").exists()
    assert (tmp_path / "data" / "signal_episodes.csv").exists()
    assert (tmp_path / "data" / "signal_events.csv").exists()
    assert (tmp_path / "data" / "signal_outcomes.csv").exists()
    assert (tmp_path / "reports" / "live_evaluation.md").exists()

    rerun = pd.DataFrame(
        [
            _scored_row(discovery_score=99.0),
            _scored_row(symbol="000009", name="뒤늦은후보"),
        ]
    )
    update_live_tracking(rerun, history, settings, generated)
    saved = pd.read_csv(tmp_path / "data" / "signal_snapshots.csv", dtype={"symbol": "string"})
    runs = pd.read_csv(tmp_path / "data" / "signal_runs.csv")

    assert len(runs) == 1
    assert len(saved) == 1
    assert saved.iloc[0]["discovery_score"] == 84.0
