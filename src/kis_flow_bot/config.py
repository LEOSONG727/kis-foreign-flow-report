from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScreeningConfig:
    min_market_cap_eok: int = 1_000
    min_median_turnover_krw: int = 1_000_000_000
    min_data_points: int = 15
    max_universe: int = 1_800
    top_n: int = 20
    bootstrap_candidates: int = 120
    history_days: int = 180
    history_retention_days: int = 550
    holdings_candidates: int = 100
    opinion_candidates: int = 40


@dataclass(frozen=True)
class APIConfig:
    calls_per_second: float = 5.0
    timeout_seconds: int = 20
    max_retries: int = 5


@dataclass(frozen=True)
class ModelConfig:
    initial_buy_score: float = 72.0
    strong_buy_score: float = 80.0
    watch_score: float = 64.0
    entry_buy_score: float = 62.0
    entry_strong_score: float = 72.0
    entry_watch_score: float = 52.0
    discovery_candidate_score: float = 65.0
    # V2 keeps timing as a ranking/explanation signal. The first walk-forward
    # test did not support a monotonic hard cutoff.
    timing_buy_score: float = 0.0
    distribution_exit_score: float = 65.0
    entry_max_distribution_score: float = 25.0
    large_flow_threshold_krw: int = 50_000_000_000
    gap_take_profit_pct: float = 5.0
    atr_gap_multiple: float = 1.2
    recent_chase_return1_pct: float = 4.0
    recent_chase_return3_pct: float = 6.0
    absolute_chase_return20_pct: float = 8.0
    recent_chase_return5_pct: float = 5.0
    recent_chase_return10_pct: float = 10.0
    rebound_chase_return10_pct: float = 12.0
    rebound_chase_return20_pct: float = 15.0
    # A ranking candidate can remain attractive while its *entry* is not ready.
    # These tighter gates are applied only to fresh/add-on buy actions.
    entry_min_return3_pct: float = -3.0
    entry_max_return20_pct: float = 3.0
    entry_max_rebound20_pct: float = 5.0
    # Cheap-accumulation screen: price still unextended, ranked by intensity
    # (ADTV) rather than absolute 20-day won amount.
    cheap_max_return20_pct: float = 3.0
    cheap_max_rebound20_pct: float = 5.0
    cheap_max_ma20_distance_pct: float = 1.0
    cheap_max_distance_high60_pct: float = -8.0
    dip_watch_ma20_distance_pct: float = 5.0
    # Persistent daily buying: at least N meaningful buy days in the last 10.
    steady_min_net20_krw: int = 10_000_000_000
    steady_min_up_days10: int = 8
    steady_max_concentration: float = 0.36
    steady_max_sellback: float = 0.80
    persistent_entry_requires_reacceleration: bool = False
    ma20_atr_chase_multiple: float = 2.5
    rebound_atr_chase_multiple: float = 4.0
    # 20-day MA slope over 5 sessions. A falling line is not a buy, even on a touch,
    # unless price is down while foreigners keep buying (bullish divergence).
    ma20_falling_slope_pct: float = -0.5
    ma20_rising_slope_pct: float = 0.2
    structural_break_return20_pct: float = -25.0
    corporate_action_daily_return_pct: float = 33.0
    initial_order_krw: int = 10_000_000
    add_order_krw: int = 5_000_000
    add_trigger_pct: float = 3.0
    max_tranches: int = 4
    strict_add_confirmation: bool = True
    add_min_return3_pct: float = -1.5
    add_min_return1_pct: float = -0.5
    add_min_return20_pct: float = -15.0
    add_cooldown_days: int = 3
    backtest_target_pct: float = 8.0
    backtest_stop_pct: float = 6.0
    backtest_horizon_days: int = 20
    backtest_roundtrip_cost_pct: float = 0.3
    strategy_trailing_stop_pct: float = 8.0


@dataclass(frozen=True)
class BriefConfig:
    """Daily human-review shortlist. Does not change executable buy gates."""
    size: int = 5
    steady_slots: int = 3
    steady_priced_slots: int = 3
    cheap_slots: int = 1
    dip_slots: int = 1
    leftover_slots: int = 1
    steady_max_ma20_distance_pct: float = 3.0
    high_conviction_score: float = 80.0
    # Scores below this can remain visible, but are not promoted to today's
    # new-buy review even when the legacy price/action gate passes.
    buy_review_score: float = 75.0
    min_discovery_score: float = 65.0
    # Slightly looser than entry gates so "almost ready" names still surface.
    watch_min_return3_pct: float = -4.0
    watch_max_return20_pct: float = 6.0
    watch_max_rebound20_pct: float = 8.0


@dataclass(frozen=True)
class HorizonConfig:
    """Sticky mid/long boards. Does not change the daily price gate."""

    mid_slots: int = 3
    long_slots: int = 3
    # A-grade names can enter the mid core. Floor is a hard disqualify, min_score is sticky cut.
    mid_floor_discovery: float = 70.0
    mid_min_score: float = 62.0
    long_min_discovery: float = 65.0
    long_min_score: float = 55.0
    long_min_net60_krw: int = 40_000_000_000
    long_min_net120_krw: int = 50_000_000_000
    stay_buffer: float = 8.0
    replace_margin: float = 5.0
    appearance_lookback: int = 10


@dataclass(frozen=True)
class TrackingConfig:
    """Append-only live recommendation tracking.

    These settings affect the evaluation ledger, not candidate scoring.
    """

    enabled: bool = True
    episode_gap_sessions: int = 5
    model_version: str = "discovery-live-v2"
    shadow_enabled: bool = True
    shadow_min_score: float = 55.0
    shadow_controls_per_bucket: int = 2
    # Live policy comparison. Missing slots are treated as cash so a narrow
    # gate cannot look better merely by making fewer recommendations.
    policy_top_k: int = 5
    policy_block_size: int = 5
    policy_bootstrap_samples: int = 2000
    policy_min_signal_dates: int = 20


@dataclass(frozen=True)
class Settings:
    root: Path
    screening: ScreeningConfig
    api: APIConfig
    model: ModelConfig
    brief: BriefConfig
    tracking: TrackingConfig
    horizon: HorizonConfig
    watchlist: tuple[str, ...]

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"


def _build(cls, values: dict):
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_settings(root: Path | None = None) -> Settings:
    root = (root or Path.cwd()).resolve()
    path = root / "config.toml"
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    api_values = dict(raw.get("api", {}))
    if os.getenv("KIS_CALLS_PER_SECOND"):
        api_values["calls_per_second"] = float(os.environ["KIS_CALLS_PER_SECOND"])

    settings = Settings(
        root=root,
        screening=_build(ScreeningConfig, raw.get("screening", {})),
        api=_build(APIConfig, api_values),
        model=_build(ModelConfig, raw.get("model", {})),
        brief=_build(BriefConfig, raw.get("brief", {})),
        tracking=_build(TrackingConfig, raw.get("tracking", {})),
        horizon=_build(HorizonConfig, raw.get("horizon", {})),
        watchlist=tuple(raw.get("watchlist", {}).get("symbols", [])),
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings
