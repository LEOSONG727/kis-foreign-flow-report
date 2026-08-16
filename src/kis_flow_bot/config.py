from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScreeningConfig:
    min_market_cap_eok: int = 1_000
    min_median_turnover_krw: int = 1_000_000_000
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
    gap_take_profit_pct: float = 5.0
    atr_gap_multiple: float = 1.2


@dataclass(frozen=True)
class Settings:
    root: Path
    screening: ScreeningConfig
    api: APIConfig
    model: ModelConfig
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
        watchlist=tuple(raw.get("watchlist", {}).get("symbols", [])),
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings
