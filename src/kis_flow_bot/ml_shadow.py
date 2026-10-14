from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAW_FEATURES: tuple[str, ...] = (
    "foreign_1_adtv",
    "foreign_3_adtv",
    "foreign_5_adtv",
    "foreign_20_adtv",
    "foreign_60_adtv",
    "foreign_prior15_adtv",
    "foreign_acceleration3",
    "foreign_acceleration5",
    "foreign_flow_concentration20",
    "foreign_buy_days10",
    "foreign_sell_days10",
    "foreign_buy_days20",
    "foreign_buy_streak",
    "foreign_inventory_up_days10",
    "foreign_inventory_down_days10",
    "foreign_flow_share20",
    "foreign_ewm20",
    "foreign_ewm120",
    "tolerant_buy_ratio20",
    "sellback_ratio20",
    "absorption_share20",
    "institution_5_adtv",
    "institution_20_adtv",
    "price_return1",
    "price_return3",
    "price_return_prev3",
    "price_acceleration3",
    "price_return5",
    "price_return10",
    "price_return20",
    "price_return60",
    "price_return120",
    "ma5_slope",
    "ma20_slope",
    "ema20_slope",
    "ma_spread_pct",
    "ma20_distance",
    "ma20_atr_distance",
    "rebound_from_low10",
    "rebound_from_low20",
    "rebound_atr_distance20",
    "distance_high20",
    "distance_high60",
    "distance_low60",
    "atr20_pct",
    "up_days5",
    "up_days10",
    "price_up_streak",
    "gap_pct",
    "upper_wick",
    "max_abs_daily_return20",
    "volume_ratio5_20",
    "rebound_volume_ratio20",
    "up_down_volume_ratio10",
    "price_residual3",
    "price_residual5",
    "price_residual10",
    "price_residual20",
    "flow_residual20",
    "market_return20",
    "timing_score",
    "distribution_score",
    "campaign_score",
    "entry_score",
    "accumulation_score",
    "underreaction_score",
    "structure_score",
    "institution_overlay",
    "price_heat_penalty",
    "dip_absorption",
    "steady_accumulation",
    "entry_gate_passed",
    "large_flow_signal",
)
RANK_SOURCES: tuple[str, ...] = (
    "foreign_5_adtv",
    "foreign_20_adtv",
    "foreign_60_adtv",
    "sellback_ratio20",
    "price_return5",
    "price_return20",
    "rebound_from_low20",
    "ma20_distance",
    "atr20_pct",
    "volume_ratio5_20",
    "price_residual20",
    "flow_residual20",
    "timing_score",
    "distribution_score",
)
RANK_FEATURES: tuple[str, ...] = tuple(f"{name}_day_rank" for name in RANK_SOURCES)
MODEL_FEATURES: tuple[str, ...] = RAW_FEATURES + RANK_FEATURES
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)

SHADOW_MODEL_VERSION = "v3.1-c-extra-trees-r1"
SHADOW_TOP_K = 12
SHADOW_LEDGER_COLUMNS = [
    "signal_date",
    "model_version",
    "symbol",
    "name",
    "market",
    "ml_shadow_rank",
    "ml_shadow_score",
    "ml_shadow_prediction",
    "discovery_rank",
    "discovery_score",
    "ml_top1",
    "baseline_top1",
    "signal_close",
    "entry_date",
    "entry_open",
    "exit_date",
    "exit_close",
    "forward_return_20",
    "max_drawdown_20",
    "resolved",
]


def _blank_shadow_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["ml_shadow_available"] = False
    result["ml_shadow_model"] = SHADOW_MODEL_VERSION
    result["ml_shadow_prediction"] = np.nan
    result["ml_shadow_score"] = np.nan
    result["ml_shadow_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["ml_shadow_top"] = False
    result["ml_shadow_baseline_rank"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )
    return result


def prepare_shadow_candidates(
    scored: pd.DataFrame,
    *,
    top_k: int = SHADOW_TOP_K,
) -> pd.DataFrame:
    """Recreate the top-k cross-section used during historical training."""

    if scored.empty:
        return scored.copy()
    eligible = (
        scored["discovery_eligible"].fillna(False).astype(bool)
        if "discovery_eligible" in scored
        else pd.Series(True, index=scored.index)
    )
    sort_columns = [
        column
        for column in ("discovery_score", "timing_score", "entry_score")
        if column in scored
    ]
    if not sort_columns:
        return scored.head(0).copy()
    candidates = (
        scored.loc[eligible]
        .sort_values(sort_columns, ascending=[False] * len(sort_columns))
        .head(top_k)
        .copy()
    )
    for column in RAW_FEATURES:
        if column not in candidates:
            candidates[column] = np.nan
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    for source, target in zip(RANK_SOURCES, RANK_FEATURES):
        candidates[target] = candidates[source].rank(method="average", pct=True)
    return candidates


def _prepare_training(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"as_of", "symbol", "excess_forward_return", *RAW_FEATURES}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing shadow training columns: {missing}")
    training = raw.copy()
    training["as_of"] = pd.to_datetime(training["as_of"], errors="coerce")
    training["symbol"] = training["symbol"].astype("string").str.zfill(6)
    for column in (*RAW_FEATURES, "excess_forward_return"):
        training[column] = pd.to_numeric(training[column], errors="coerce")
    training = training.dropna(subset=["as_of", "symbol", "excess_forward_return"])
    training = training.drop_duplicates(["as_of", "symbol"], keep="last")
    for source, target in zip(RANK_SOURCES, RANK_FEATURES):
        training[target] = training.groupby("as_of")[source].rank(
            method="average", pct=True
        )
    training["target_rank"] = training.groupby("as_of")[
        "excess_forward_return"
    ].rank(method="average", pct=True)
    return training.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def _build_shadow_model() -> Any:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "ML dependencies are not installed. Run: pip install -e '.[ml]'"
        ) from exc
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=8,
                    max_features=0.65,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_shadow_model(training_path: Path) -> Any:
    from .storage import read_csv

    raw = read_csv(training_path)
    training = _prepare_training(raw)
    if len(training) < 500 or training["as_of"].nunique() < 50:
        raise ValueError(
            "Shadow training data is too small: "
            f"rows={len(training)} dates={training['as_of'].nunique()}"
        )
    model = _build_shadow_model()
    model.fit(training[list(MODEL_FEATURES)], training["target_rank"])
    return model


def apply_rich_shadow(
    scored: pd.DataFrame,
    training_path: Path,
    *,
    top_k: int = SHADOW_TOP_K,
    model: Any | None = None,
) -> pd.DataFrame:
    """Attach non-trading ML ranks. Failure leaves the main score untouched."""

    result = _blank_shadow_columns(scored)
    candidates = prepare_shadow_candidates(result, top_k=top_k)
    if len(candidates) < 2:
        LOGGER.warning("ML shadow skipped because fewer than two eligible candidates exist")
        return result
    if model is None:
        if not training_path.exists():
            LOGGER.warning("ML shadow training data is missing: %s", training_path)
            return result
        try:
            model = fit_shadow_model(training_path)
        except Exception as exc:
            LOGGER.warning("ML shadow model could not be trained: %s", exc)
            return result

    try:
        prediction = np.asarray(
            model.predict(candidates[list(MODEL_FEATURES)]), dtype=float
        )
    except Exception as exc:
        LOGGER.warning("ML shadow prediction failed: %s", exc)
        return result

    prediction_series = pd.Series(prediction, index=candidates.index)
    shadow_score = prediction_series.rank(method="average", pct=True) * 100.0
    shadow_rank = prediction_series.rank(method="first", ascending=False).astype(int)
    baseline_rank = (
        candidates["discovery_score"].rank(method="first", ascending=False).astype(int)
    )

    result.loc[candidates.index, "ml_shadow_available"] = True
    result.loc[candidates.index, "ml_shadow_prediction"] = prediction_series
    result.loc[candidates.index, "ml_shadow_score"] = shadow_score
    result.loc[candidates.index, "ml_shadow_rank"] = shadow_rank
    result.loc[candidates.index, "ml_shadow_top"] = shadow_rank.eq(1)
    result.loc[candidates.index, "ml_shadow_baseline_rank"] = baseline_rank
    result["ml_shadow_rank"] = result["ml_shadow_rank"].astype("Int64")
    result["ml_shadow_baseline_rank"] = result["ml_shadow_baseline_rank"].astype("Int64")
    LOGGER.info(
        "ML shadow ranked %d candidates; top=%s",
        len(candidates),
        ", ".join(
            result.loc[result["ml_shadow_top"], "name"].astype(str).tolist()
        ),
    )
    return result


def _new_snapshot_rows(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "ml_shadow_available" not in scored:
        return pd.DataFrame(columns=SHADOW_LEDGER_COLUMNS)
    available = scored[scored["ml_shadow_available"].fillna(False)].copy()
    if available.empty:
        return pd.DataFrame(columns=SHADOW_LEDGER_COLUMNS)
    signal_date = pd.to_datetime(available["as_of"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    rows = pd.DataFrame(
        {
            "signal_date": signal_date,
            "model_version": SHADOW_MODEL_VERSION,
            "symbol": available["symbol"].astype("string").str.zfill(6),
            "name": available.get("name", available["symbol"]).astype(str),
            "market": available.get(
                "market", pd.Series("", index=available.index)
            ).astype(str),
            "ml_shadow_rank": available["ml_shadow_rank"].astype("Int64"),
            "ml_shadow_score": available["ml_shadow_score"],
            "ml_shadow_prediction": available["ml_shadow_prediction"],
            "discovery_rank": available["ml_shadow_baseline_rank"].astype("Int64"),
            "discovery_score": available["discovery_score"],
            "ml_top1": available["ml_shadow_rank"].eq(1),
            "baseline_top1": available["ml_shadow_baseline_rank"].eq(1),
            "signal_close": pd.to_numeric(
                available.get("close", pd.Series(np.nan, index=available.index)),
                errors="coerce",
            ),
            "entry_date": "",
            "entry_open": np.nan,
            "exit_date": "",
            "exit_close": np.nan,
            "forward_return_20": np.nan,
            "max_drawdown_20": np.nan,
            "resolved": False,
        }
    )
    return rows[SHADOW_LEDGER_COLUMNS]


def _resolve_outcomes(
    ledger: pd.DataFrame,
    history: pd.DataFrame,
    roundtrip_cost_pct: float,
) -> pd.DataFrame:
    if ledger.empty or history.empty:
        return ledger
    result = ledger.copy()
    prices = history.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["symbol"] = prices["symbol"].astype("string").str.zfill(6)
    prices = prices.dropna(subset=["date"]).sort_values(["symbol", "date"])
    cost = float(roundtrip_cost_pct) / 100.0

    unresolved = ~result["resolved"].fillna(False).astype(bool)
    for index, row in result.loc[unresolved].iterrows():
        signal_date = pd.to_datetime(row["signal_date"], errors="coerce")
        if pd.isna(signal_date):
            continue
        future = prices[
            (prices["symbol"] == str(row["symbol"]).zfill(6))
            & (prices["date"] > signal_date)
        ].head(20)
        if len(future) < 20:
            continue
        entry = pd.to_numeric(future["open"].iloc[0], errors="coerce")
        exit_price = pd.to_numeric(future["close"].iloc[-1], errors="coerce")
        lows = pd.to_numeric(future["low"], errors="coerce").dropna()
        if pd.isna(entry) or pd.isna(exit_price) or float(entry) <= 0:
            continue
        result.at[index, "entry_date"] = pd.Timestamp(future["date"].iloc[0]).strftime(
            "%Y-%m-%d"
        )
        result.at[index, "entry_open"] = float(entry)
        result.at[index, "exit_date"] = pd.Timestamp(future["date"].iloc[-1]).strftime(
            "%Y-%m-%d"
        )
        result.at[index, "exit_close"] = float(exit_price)
        result.at[index, "forward_return_20"] = float(exit_price) / float(entry) - 1.0 - cost
        result.at[index, "max_drawdown_20"] = (
            float((lows / float(entry) - 1.0 - cost).min())
            if not lows.empty
            else np.nan
        )
        result.at[index, "resolved"] = True
    return result


def update_shadow_tracking(
    scored: pd.DataFrame,
    history: pd.DataFrame,
    output_path: Path,
    *,
    roundtrip_cost_pct: float = 0.3,
) -> pd.DataFrame:
    """Append the first snapshot for each day/model and resolve 20-session outcomes."""

    incoming = _new_snapshot_rows(scored)
    existing = read_csv(output_path)
    if existing.empty:
        combined = incoming
    elif incoming.empty:
        combined = existing
    else:
        for column in SHADOW_LEDGER_COLUMNS:
            if column not in existing:
                existing[column] = pd.NA
        keys = ["signal_date", "model_version", "symbol"]
        known = pd.MultiIndex.from_frame(existing[keys].astype(str))
        candidate_keys = pd.MultiIndex.from_frame(incoming[keys].astype(str))
        combined = pd.concat(
            [existing, incoming.loc[~candidate_keys.isin(known)]],
            ignore_index=True,
        )
    if combined.empty:
        return combined
    for column in SHADOW_LEDGER_COLUMNS:
        if column not in combined:
            combined[column] = pd.NA
    combined["symbol"] = combined["symbol"].astype("string").str.zfill(6)
    combined["resolved"] = combined["resolved"].fillna(False).astype(bool)
    combined = _resolve_outcomes(combined, history, roundtrip_cost_pct)
    combined = combined.sort_values(
        ["signal_date", "model_version", "ml_shadow_rank", "symbol"]
    ).reset_index(drop=True)
    write_csv(combined[SHADOW_LEDGER_COLUMNS], output_path)
    return combined[SHADOW_LEDGER_COLUMNS]
