"""Self-validation loop: does any score actually rank tomorrow's winners?

The rule model, the daily brief and the price gates stay untouched. This
module answers three questions from stored history only, every week:

1. Factor audit. For every eligible stock on every signal date, how well does
   each score (and each raw input) rank the next-open → D+h close return?
   Rank IC is measured on the whole cross-section, not only on the picks, and
   again after removing size / volatility / 60-day momentum so that a generic
   style tilt cannot masquerade as a foreign-flow edge.
2. Challenger. A small linear ranker on point-in-time features is refit
   walk-forward (labels are used only after they were observable) and scored
   on dates it has never seen. Its spec is fixed in code and config, so there
   is no hand-tuning loop to overfit.
3. Gate. Pre-registered thresholds decide whether the challenger has earned
   promotion and whether the rule champion has shown out-of-sample skill since
   ``rules_frozen_on``. The decision is appended to a registry; the gate never
   edits code or the rule model.
"""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Settings
from .eligibility import apply_candidate_eligibility
from .features import add_cross_sectional_residuals, build_base_features
from .fundamentals import (
    FUNDAMENTAL_FEATURES,
    attach_fundamentals,
    load_financials,
    refresh_financials_from_env,
)
from .jev_shadow import REQUIRED as JEV_INPUTS
from .live_tracking import moving_block_mean_interval
from .scoring import score_features
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)

SPEC_VERSION = "ridge-rank-pit-v2"
PANEL_HORIZONS: tuple[int, ...] = (5, 10, 20)
# Enough for the 120-day windows and the EWM tails used by the features.
FEATURE_LOOKBACK_SESSIONS = 250
MIN_SCORED_HISTORY = 60

CHAMPION = "discovery_score"
RULE_SCORES: tuple[str, ...] = (
    "discovery_score",
    "score",
    "entry_score",
    "timing_score",
    "signal_score",
    "campaign_score",
    "accumulation_score",
    "underreaction_score",
    "distribution_score",
)
# Point-in-time inputs only. Current financials, analyst opinions, holdings
# snapshots and today's market cap are excluded: in a replay they leak the
# future. pit_size uses the close on the signal date.
LEARNED_FEATURES: tuple[str, ...] = (
    "foreign_1_adtv",
    "foreign_3_adtv",
    "foreign_5_adtv",
    "foreign_20_adtv",
    "foreign_60_adtv",
    "foreign_acceleration5",
    "foreign_flow_concentration20",
    "foreign_buy_days10",
    "tolerant_buy_ratio20",
    "sellback_ratio20",
    "flow_residual20",
    "institution_5_adtv",
    "institution_20_adtv",
    "price_return1",
    "price_return5",
    "price_return20",
    "price_return60",
    "price_residual5",
    "price_residual20",
    "rebound_from_low20",
    "distance_high60",
    "ma20_distance",
    "ma20_slope",
    "atr20_pct",
    "max_abs_daily_return20",
    "upper_wick",
    "volume_ratio5_20",
    "up_down_volume_ratio10",
    "log_adtv20",
    "pit_size",
    # v2: point-in-time DART financials (usable only after each report's receipt date).
    *FUNDAMENTAL_FEATURES,
)
# Style controls for the partial IC: a flow edge must survive these.
CONTROLS: tuple[str, ...] = ("pit_size", "atr20_pct", "price_return60")
DIAGNOSTIC_FEATURES: tuple[str, ...] = (
    "foreign_5_adtv",
    "foreign_20_adtv",
    "foreign_60_adtv",
    "flow_residual20",
    "tolerant_buy_ratio20",
    "sellback_ratio20",
    "price_return20",
    "price_residual20",
    "rebound_from_low20",
    "atr20_pct",
    "pit_size",
    "op_yoy_mcap",
    "op_yield",
    "book_to_market",
    "revenue_yoy",
    "fundamental_score",
)
IDENTITY_COLUMNS: tuple[str, ...] = ("as_of", "date_index", "symbol", "name", "market")

STATUS_LABELS = {
    "INSUFFICIENT_DATA": "표본 부족",
    "NO_EDGE": "예측력 미확인",
    "NOT_BETTER": "현행 대비 우위 없음",
    "PROMOTE": "승격 기준 통과",
    "VALIDATED": "실전 예측력 확인",
    "NEGATIVE": "역방향 예측",
}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

_PANEL_CTX: dict[str, Any] = {}


def _init_panel_worker(payload: dict[str, Any]) -> None:
    _PANEL_CTX.clear()
    _PANEL_CTX.update(payload)


def _panel_one_date(date_index: int) -> pd.DataFrame:
    ctx = _PANEL_CTX
    dates: list[pd.Timestamp] = ctx["dates"]
    history: pd.DataFrame = ctx["history"]
    settings: Settings = ctx["settings"]
    as_of = pd.Timestamp(dates[date_index])
    start = pd.Timestamp(dates[max(0, date_index - FEATURE_LOOKBACK_SESSIONS + 1)])
    view = history[(history["date"] >= start) & (history["date"] <= as_of)]
    snapshots = ctx["snapshots"]
    if not snapshots.empty and "date" in snapshots:
        snapshots = snapshots[snapshots["date"] <= as_of]
    opinions = ctx["opinions"]
    if not opinions.empty and "date" in opinions:
        opinions = opinions[opinions["date"] <= as_of]
    features = add_cross_sectional_residuals(
        build_base_features(view, ctx["universe"], snapshots, opinions)
    )
    if features.empty:
        return pd.DataFrame()
    scored = score_features(features, settings.model)
    scored = apply_candidate_eligibility(scored, settings.screening, report_date=as_of)
    scored = scored[scored["discovery_eligible"]].copy()
    if scored.empty:
        return pd.DataFrame()
    scored["as_of"] = as_of
    scored["date_index"] = date_index
    scored["symbol"] = scored["symbol"].astype(str).str.zfill(6)

    opens: pd.DataFrame = ctx["opens"]
    closes: pd.DataFrame = ctx["closes"]
    cost = settings.model.backtest_roundtrip_cost_pct / 100.0
    for horizon in PANEL_HORIZONS:
        column = f"fwd{horizon}"
        if date_index + horizon < len(dates):
            entry = opens.loc[dates[date_index + 1]]
            exit_ = closes.loc[dates[date_index + horizon]]
            forward = (exit_ / entry.where(entry > 0) - 1.0 - cost).reindex(scored["symbol"])
            scored[column] = forward.to_numpy()
        else:
            scored[column] = np.nan
        benchmark = scored.groupby("market")[column].transform("mean")
        scored[f"excess{horizon}"] = scored[column] - benchmark
    # Worst entry-relative low over the 20-session hold (tail-risk label).
    if date_index + 20 < len(dates):
        entry = opens.loc[dates[date_index + 1]]
        window = ctx["lows"].loc[dates[date_index + 1] : dates[date_index + 20]].min()
        scored["worst20"] = (window / entry.where(entry > 0) - 1.0).reindex(scored["symbol"]).to_numpy()
    else:
        scored["worst20"] = np.nan

    keep = list(
        dict.fromkeys(
            [
                *IDENTITY_COLUMNS,
                *RULE_SCORES,
                *LEARNED_FEATURES,
                *DIAGNOSTIC_FEATURES,
                *JEV_INPUTS,
                "foreign_20_adtv",
                "market_return20",
                "timing_score",
                "close",
                *(f"fwd{h}" for h in PANEL_HORIZONS),
                *(f"excess{h}" for h in PANEL_HORIZONS),
                "worst20",
            ]
        )
    )
    for column in keep:
        if column not in scored:
            scored[column] = np.nan
    return scored[keep].reset_index(drop=True)


def _load_inputs(settings: Settings) -> dict[str, Any]:
    history = read_csv(settings.data_dir / "history.csv")
    universe = read_csv(settings.data_dir / "universe.csv")
    if history.empty or universe.empty:
        raise RuntimeError("history.csv / universe.csv are required for research")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["symbol"] = history["symbol"].astype("string").str.zfill(6)
    history = history.dropna(subset=["date"])
    for column in ("open", "close", "low"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    snapshots = read_csv(settings.data_dir / "holdings.csv")
    opinions = read_csv(settings.data_dir / "opinions.csv")
    for frame in (snapshots, opinions):
        if not frame.empty and "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    universe["symbol"] = universe["symbol"].astype("string").str.zfill(6)
    return {
        "history": history,
        "universe": universe,
        "snapshots": snapshots,
        "opinions": opinions,
        "dates": [pd.Timestamp(value) for value in sorted(history["date"].unique())],
        "opens": history.pivot_table(index="date", columns="symbol", values="open", aggfunc="last"),
        "closes": history.pivot_table(index="date", columns="symbol", values="close", aggfunc="last"),
        "lows": history.pivot_table(index="date", columns="symbol", values="low", aggfunc="last"),
    }


def build_research_panel(settings: Settings, workers: int | None = None) -> pd.DataFrame:
    """Score every eligible stock on every date that has enough history."""

    payload = _load_inputs(settings)
    payload["settings"] = settings
    dates = payload["dates"]
    indices = list(range(MIN_SCORED_HISTORY - 1, len(dates) - 1))
    if not indices:
        return pd.DataFrame()
    workers = workers or settings.research.workers or os.cpu_count() or 1
    LOGGER.info("Research panel: %d signal dates on %d workers", len(indices), workers)
    if workers <= 1:
        _init_panel_worker(payload)
        parts = [_panel_one_date(index) for index in indices]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_panel_worker,
            initargs=(payload,),
        ) as pool:
            parts = list(pool.map(_panel_one_date, indices, chunksize=2))
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    try:
        financials = refresh_financials_from_env(
            settings, panel["symbol"].unique(), pd.Timestamp(dates[0]).date()
        )
    except Exception as exc:  # stale cache is better than no panel
        LOGGER.warning("DART financials refresh failed; using cache: %s", exc)
        financials = load_financials(settings.data_dir)
    return attach_fundamentals(panel, financials)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def daily_rank_ic(panel: pd.DataFrame, column: str, target: str) -> pd.Series:
    """Spearman correlation between a score and the forward return, per date."""

    if panel.empty or column not in panel or target not in panel:
        return pd.Series(dtype=float)
    data = panel[["as_of", column, target]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna()
    grouped = data.groupby("as_of")
    x = grouped[column].rank(method="average")
    y = grouped[target].rank(method="average")
    frame = pd.DataFrame({"as_of": data["as_of"], "x": x, "y": y})
    counts = frame.groupby("as_of")["x"].transform("count")
    frame = frame[counts >= 20]
    if frame.empty:
        # An all-missing feature (e.g. financials not collected) has no IC.
        return pd.Series(dtype=float)
    result = frame.groupby("as_of")[["x", "y"]].apply(
        lambda group: group["x"].corr(group["y"]) if group["x"].nunique() > 1 else np.nan
    )
    if not isinstance(result, pd.Series):
        return pd.Series(dtype=float)
    return result.dropna().astype(float)


def _rank_matrix(group: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> np.ndarray:
    ranks = group[list(columns)].apply(pd.to_numeric, errors="coerce").rank(pct=True) - 0.5
    return ranks.fillna(0.0).to_numpy(dtype=float)


def partial_rank_ic(
    panel: pd.DataFrame,
    column: str,
    target: str,
    controls: tuple[str, ...] = CONTROLS,
) -> pd.Series:
    """Rank IC after projecting both sides off the style controls, per date."""

    if panel.empty or column not in panel or target not in panel:
        return pd.Series(dtype=float)
    present = [control for control in controls if control in panel and control != column]
    values: dict[pd.Timestamp, float] = {}
    for as_of, group in panel.groupby("as_of"):
        group = group[pd.to_numeric(group[target], errors="coerce").notna()]
        group = group[pd.to_numeric(group[column], errors="coerce").notna()]
        if len(group) < 30:
            continue
        design = np.column_stack([np.ones(len(group)), _rank_matrix(group, present)])
        x = group[column].rank(pct=True).to_numpy(dtype=float)
        y = pd.to_numeric(group[target], errors="coerce").rank(pct=True).to_numpy(dtype=float)
        beta_x, *_ = np.linalg.lstsq(design, x, rcond=None)
        beta_y, *_ = np.linalg.lstsq(design, y, rcond=None)
        rx = x - design @ beta_x
        ry = y - design @ beta_y
        if rx.std() <= 1e-12 or ry.std() <= 1e-12:
            continue
        values[pd.Timestamp(as_of)] = float(np.corrcoef(rx, ry)[0, 1])
    return pd.Series(values, dtype=float).sort_index()


@dataclass(frozen=True)
class SeriesSummary:
    dates: int
    mean: float
    low: float
    high: float
    positive_share: float
    effective_t: float

    def as_dict(self) -> dict[str, float]:
        return {
            "dates": self.dates,
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "positive_share": self.positive_share,
            "effective_t": self.effective_t,
        }


def summarize_series(
    values: pd.Series,
    block: int,
    confidence: float = 0.90,
    samples: int = 2000,
) -> SeriesSummary:
    """Mean with a moving-block bootstrap interval.

    Daily IC of a 20-session return overlaps for 20 days, so the effective
    t-statistic divides the date count by the horizon.
    """

    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        nan = float("nan")
        return SeriesSummary(0, nan, nan, nan, nan, nan)
    mean = float(clean.mean())
    low, high = moving_block_mean_interval(
        clean.reset_index(drop=True),
        block_size=block,
        samples=samples,
        confidence=confidence,
    )
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    effective = max(len(clean) / max(block, 1), 1.0)
    t_value = mean / std * math.sqrt(effective) if std and std > 0 else float("nan")
    return SeriesSummary(
        dates=int(len(clean)),
        mean=mean,
        low=float(low) if low is not None else float("nan"),
        high=float(high) if high is not None else float("nan"),
        positive_share=float((clean > 0).mean()),
        effective_t=float(t_value),
    )


def top_k_excess(panel: pd.DataFrame, column: str, target: str, top_k: int) -> pd.Series:
    """Mean excess of the top-k names per date (excess is vs same-market EW mean)."""

    if panel.empty or column not in panel or target not in panel:
        return pd.Series(dtype=float)
    data = panel[["as_of", column, target]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna()
    ranked = data.sort_values(["as_of", column], ascending=[True, False])
    top = ranked.groupby("as_of").head(top_k)
    return top.groupby("as_of")[target].mean().astype(float)


# ---------------------------------------------------------------------------
# Walk-forward challenger
# ---------------------------------------------------------------------------


def _design(group: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    return _rank_matrix(group, features)


def _target(group: pd.DataFrame, target: str) -> np.ndarray:
    return (pd.to_numeric(group[target], errors="coerce").rank(pct=True) - 0.5).to_numpy(dtype=float)


def fit_ridge(
    frames: list[pd.DataFrame],
    target: str,
    alpha: float,
    features: tuple[str, ...] = LEARNED_FEATURES,
) -> np.ndarray:
    """Closed-form ridge on per-date ranks. Penalty is per sample: alpha / 1000."""

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for group in frames:
        labelled = group[pd.to_numeric(group[target], errors="coerce").notna()]
        if len(labelled) < 20:
            continue
        xs.append(_design(labelled, features))
        ys.append(_target(labelled, target))
    if not xs:
        return np.zeros(len(features))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    n = len(y)
    gram = x.T @ x / n + (alpha / 1000.0) * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y / n)


def walk_forward_predictions(
    panel: pd.DataFrame,
    horizon: int,
    retrain_every: int,
    min_train_dates: int,
    alpha: float,
    features: tuple[str, ...] = LEARNED_FEATURES,
) -> pd.DataFrame:
    """Predict each date with a model fit only on labels observable by then.

    A label for signal date d needs the close of d + horizon sessions, so on
    date t only dates with index <= t - horizon may be used for training.
    """

    target = f"excess{horizon}"
    if panel.empty or target not in panel:
        return pd.DataFrame(columns=["as_of", "symbol", "challenger_score"])
    groups = {int(index): group for index, group in panel.groupby("date_index", sort=True)}
    indices = sorted(groups)
    outputs: list[pd.DataFrame] = []
    weights: np.ndarray | None = None
    last_fit: int | None = None
    for index in indices:
        known = [i for i in indices if i <= index - horizon]
        if len(known) < min_train_dates:
            continue
        if weights is None or last_fit is None or index - last_fit >= retrain_every:
            weights = fit_ridge([groups[i] for i in known], target, alpha, features)
            last_fit = index
        group = groups[index]
        prediction = _design(group, features) @ weights
        outputs.append(
            pd.DataFrame(
                {
                    "as_of": group["as_of"].to_numpy(),
                    "date_index": index,
                    "symbol": group["symbol"].to_numpy(),
                    "challenger_score": prediction,
                }
            )
        )
    if not outputs:
        return pd.DataFrame(columns=["as_of", "date_index", "symbol", "challenger_score"])
    return pd.concat(outputs, ignore_index=True)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def challenger_decision(
    challenger_ic: SeriesSummary,
    ic_delta: SeriesSummary,
    challenger_top: SeriesSummary,
    min_oos_dates: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if challenger_ic.dates < min_oos_dates:
        reasons.append(f"OOS 신호일 {challenger_ic.dates}/{min_oos_dates}")
        return "INSUFFICIENT_DATA", reasons
    if not (challenger_ic.low > 0):
        reasons.append(f"IC 하한 {challenger_ic.low:+.3f} ≤ 0")
        return "NO_EDGE", reasons
    if not (ic_delta.low > 0):
        reasons.append(f"현행 대비 IC 차이 하한 {ic_delta.low:+.3f} ≤ 0")
        return "NOT_BETTER", reasons
    if not (challenger_top.mean > 0):
        reasons.append(f"Top-K 초과수익 평균 {challenger_top.mean:+.2%} ≤ 0")
        return "NO_EDGE", reasons
    reasons.append(
        f"IC {challenger_ic.mean:+.3f} (하한 {challenger_ic.low:+.3f}), "
        f"현행 대비 +{ic_delta.mean:.3f}, Top-K {challenger_top.mean:+.2%}"
    )
    return "PROMOTE", reasons


def champion_decision(post_freeze_ic: SeriesSummary, min_oos_dates: int) -> str:
    if post_freeze_ic.dates < min_oos_dates:
        return "INSUFFICIENT_DATA"
    if post_freeze_ic.high < 0:
        return "NEGATIVE"
    if post_freeze_ic.low > 0:
        return "VALIDATED"
    return "NO_EDGE"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _fmt(value: float, pct: bool = False, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:+.2f}%" if pct else f"{value:+.{digits}f}"


def analyze_panel(panel: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    research = settings.research
    horizon = research.horizon
    target = f"excess{horizon}"
    kwargs = {"confidence": research.confidence, "samples": research.bootstrap_samples}
    labelled = panel[pd.to_numeric(panel.get(target), errors="coerce").notna()] if not panel.empty else panel

    factor_rows: list[dict[str, Any]] = []
    for column in dict.fromkeys([*RULE_SCORES, *DIAGNOSTIC_FEATURES]):
        if column not in labelled:
            continue
        row: dict[str, Any] = {"feature": column}
        for h in (5, horizon):
            raw = summarize_series(daily_rank_ic(labelled, column, f"excess{h}"), h, **kwargs)
            row[f"ic{h}"] = raw.mean
            row[f"ic{h}_low"] = raw.low
            row[f"ic{h}_t"] = raw.effective_t
        partial = summarize_series(partial_rank_ic(labelled, column, target), horizon, **kwargs)
        row["partial_ic"] = partial.mean
        row["partial_ic_low"] = partial.low
        row["partial_ic_t"] = partial.effective_t
        row["dates"] = partial.dates
        factor_rows.append(row)
    factors = pd.DataFrame(factor_rows)

    # Rule champion before/after the freeze date.
    freeze = pd.to_datetime(research.rules_frozen_on, errors="coerce")
    champion_ic = daily_rank_ic(labelled, CHAMPION, target)
    champion_partial = partial_rank_ic(labelled, CHAMPION, target)
    if pd.notna(freeze):
        post_ic = champion_ic[champion_ic.index > freeze]
    else:
        post_ic = champion_ic.iloc[0:0]
    champion = {
        "in_sample_ic": summarize_series(champion_ic, horizon, **kwargs).as_dict(),
        "in_sample_partial_ic": summarize_series(champion_partial, horizon, **kwargs).as_dict(),
        "post_freeze_ic": summarize_series(post_ic, horizon, **kwargs).as_dict(),
        "top_k_excess": summarize_series(
            top_k_excess(labelled, CHAMPION, target, research.top_k), horizon, **kwargs
        ).as_dict(),
    }
    champion["status"] = champion_decision(
        summarize_series(post_ic, horizon, **kwargs), research.min_oos_dates
    )

    # Walk-forward challenger on the same eligible universe.
    predictions = walk_forward_predictions(
        panel,
        horizon,
        research.retrain_every,
        research.min_train_dates,
        research.ridge_alpha,
    )
    merged = (
        labelled.merge(predictions[["as_of", "symbol", "challenger_score"]], on=["as_of", "symbol"], how="inner")
        if not predictions.empty
        else labelled.iloc[0:0].assign(challenger_score=np.nan)
    )
    challenger_ic_series = daily_rank_ic(merged, "challenger_score", target)
    champion_same_dates = daily_rank_ic(merged, CHAMPION, target)
    delta = (challenger_ic_series - champion_same_dates).dropna()
    challenger_top_series = top_k_excess(merged, "challenger_score", target, research.top_k)
    champion_top_series = top_k_excess(merged, CHAMPION, target, research.top_k)
    challenger_ic = summarize_series(challenger_ic_series, horizon, **kwargs)
    ic_delta = summarize_series(delta, horizon, **kwargs)
    challenger_top = summarize_series(challenger_top_series, horizon, **kwargs)
    status, reasons = challenger_decision(
        challenger_ic, ic_delta, challenger_top, research.min_oos_dates
    )
    oos = pd.DataFrame(
        {
            "challenger_ic": challenger_ic_series,
            "champion_ic": champion_same_dates,
            "challenger_top_k_excess": challenger_top_series,
            "champion_top_k_excess": champion_top_series,
        }
    ).rename_axis("as_of").reset_index()

    # Final model: every label observable as of the latest panel date.
    final_weights = fit_ridge(
        [group for _, group in labelled.groupby("date_index")],
        target,
        research.ridge_alpha,
    )
    challenger = {
        "status": status,
        "reasons": reasons,
        "oos_ic": challenger_ic.as_dict(),
        "ic_delta_vs_champion": ic_delta.as_dict(),
        "oos_top_k_excess": challenger_top.as_dict(),
        "champion_top_k_excess_same_dates": summarize_series(
            champion_top_series, horizon, **kwargs
        ).as_dict(),
    }
    return {
        "predictions": predictions,
        "factors": factors,
        "oos": oos,
        "champion": champion,
        "challenger": challenger,
        "weights": dict(zip(LEARNED_FEATURES, (float(value) for value in final_weights))),
        "labelled_dates": int(labelled["as_of"].nunique()) if not labelled.empty else 0,
        "panel_dates": int(panel["as_of"].nunique()) if not panel.empty else 0,
        "first_date": str(pd.Timestamp(panel["as_of"].min()).date()) if not panel.empty else "",
        "last_labelled": str(pd.Timestamp(labelled["as_of"].max()).date()) if not labelled.empty else "",
        "mean_names": float(labelled.groupby("as_of").size().mean()) if not labelled.empty else 0.0,
    }


def render_research_markdown(result: dict[str, Any], settings: Settings, generated_at: datetime) -> str:
    research = settings.research
    h = research.horizon
    champion = result["champion"]
    challenger = result["challenger"]
    lines = [
        "# 자가검증 리포트",
        "",
        f"- 갱신: {generated_at.isoformat(timespec='seconds')} · 사양 `{SPEC_VERSION}`",
        f"- 표본: 신호일 {result['panel_dates']}일 (라벨 완료 {result['labelled_dates']}일, {result['first_date']} ~ {result['last_labelled']}) · 일평균 적격 종목 {result['mean_names']:.0f}개",
        f"- 수익률: 다음 거래일 시가 → D+{h} 종가, 왕복 {settings.model.backtest_roundtrip_cost_pct:.1f}% · 초과수익은 같은 시장 적격 종목 동일가중 평균 대비",
        f"- 구간: {int(research.confidence * 100)}% 이동블록 부트스트랩(블록 {h}일). 유효 t는 신호일 수를 {h}로 나눈 독립 구간 기준",
        "",
        "## 결론",
        "",
        f"- 현행 발견점수(챔피언): **{STATUS_LABELS.get(champion['status'], champion['status'])}** — 규칙 동결일({research.rules_frozen_on or '미설정'}) 이후 신호일 {champion['post_freeze_ic']['dates']}일",
        f"- 학습 도전자: **{STATUS_LABELS.get(challenger['status'], challenger['status'])}** — {' · '.join(challenger['reasons'])}",
        "- 이 판정은 코드·규칙·게이트를 자동 변경하지 않습니다. 승격 기준 통과 시에만 리포트에 검증 순위로 표시합니다.",
        "",
        "## 1. 현행 점수의 전종목 순위 예측력",
        "",
        "IC는 그날 적격 종목 전체에서 점수 순위와 이후 초과수익 순위의 상관입니다. 0.02~0.05면 실무적으로 의미 있는 수준이고, 부분 IC는 시가총액·변동성·60일 모멘텀을 제거한 뒤에도 남는 예측력입니다.",
        "",
        f"| 점수·변수 | IC D+5 | IC D+{h} | 하한 | 유효 t | 부분 IC | 부분 하한 | 신호일 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    factors = result["factors"]
    for _, row in factors.iterrows():
        lines.append(
            f"| {row['feature']} | {_fmt(row.get('ic5'))} | {_fmt(row.get(f'ic{h}'))} | "
            f"{_fmt(row.get(f'ic{h}_low'))} | {_fmt(row.get(f'ic{h}_t'), digits=2)} | "
            f"{_fmt(row.get('partial_ic'))} | {_fmt(row.get('partial_ic_low'))} | {int(row.get('dates', 0))} |"
        )
    in_sample = champion["in_sample_ic"]
    top = champion["top_k_excess"]
    lines.extend(
        [
            "",
            f"- 발견점수 Top {research.top_k} 초과수익 평균 {_fmt(top['mean'], pct=True)} (구간 {_fmt(top['low'], pct=True)} ~ {_fmt(top['high'], pct=True)})",
            f"- 발견점수 전체기간 IC {_fmt(in_sample['mean'])}: 규칙을 이 기간 데이터로 조정했으므로 표본 내 수치입니다.",
            "- `fundamental_score`는 현재 재무값을 과거에 적용하므로 미래정보가 섞여 있습니다. 참고만 합니다.",
            "",
            "## 2. 학습 도전자 (워크포워드)",
            "",
            f"적격 전종목의 시점 정보만으로 순위를 학습합니다(릿지·일별 순위 변환, 변수 {len(LEARNED_FEATURES)}개). "
            f"각 신호일은 그날까지 결과가 확정된 데이터(D+{h} 경과분)만으로 학습한 모델이 예측하며, {research.retrain_every}일마다 재학습합니다.",
            "",
            "| 지표 | 신호일 | 평균 | 하한 | 상한 | 양수 비율 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key, pct in (
        ("도전자 OOS IC", "oos_ic", False),
        ("도전자 − 현행 IC", "ic_delta_vs_champion", False),
        (f"도전자 Top {research.top_k} 초과수익", "oos_top_k_excess", True),
        (f"현행 Top {research.top_k} 초과수익(같은 날)", "champion_top_k_excess_same_dates", True),
    ):
        item = challenger[key]
        lines.append(
            f"| {label} | {item['dates']} | {_fmt(item['mean'], pct=pct)} | {_fmt(item['low'], pct=pct)} | "
            f"{_fmt(item['high'], pct=pct)} | {_fmt(item['positive_share'], pct=True).replace('+', '')} |"
        )
    weights = sorted(result["weights"].items(), key=lambda item: abs(item[1]), reverse=True)
    lines.extend(
        [
            "",
            "최종 모델 가중치 상위 (순위 −0.5~+0.5 기준, 양수면 높을수록 유리):",
            "",
            " · ".join(f"`{name}` {value:+.3f}" for name, value in weights[:10]),
            "",
            "## 승격 규칙 (사전등록)",
            "",
            f"1. OOS 신호일 {research.min_oos_dates}일 이상",
            f"2. 도전자 IC {int(research.confidence * 100)}% 하한 > 0",
            "3. 같은 날 현행 발견점수 대비 IC 차이 하한 > 0",
            f"4. 도전자 Top {research.top_k} 초과수익 평균 > 0",
            "",
            "현행 규칙은 동결일 이후 신호만으로 판정합니다. 규칙을 고치면 `rules_frozen_on`을 갱신해 표본을 다시 모읍니다. "
            "판정 결과를 보고 사양(변수·릿지 강도·기간)을 바꾸면 검증이 무효가 되므로, 바꿀 때는 `SPEC_VERSION`을 올리고 기록을 새로 시작합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_research_outputs(result: dict[str, Any], settings: Settings, generated_at: datetime) -> None:
    reports = settings.reports_dir
    data = settings.data_dir
    (reports / "research_latest.md").write_text(
        render_research_markdown(result, settings, generated_at), encoding="utf-8"
    )
    write_csv(result["factors"], reports / "research_factors.csv")
    write_csv(result["oos"], reports / "research_oos.csv")

    record = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "spec_version": SPEC_VERSION,
        "rules_frozen_on": settings.research.rules_frozen_on,
        "horizon": settings.research.horizon,
        "panel_dates": result["panel_dates"],
        "labelled_dates": result["labelled_dates"],
        "last_labelled": result["last_labelled"],
        "champion": result["champion"],
        "challenger": result["challenger"],
    }
    ranker = {
        **record,
        "features": list(LEARNED_FEATURES),
        "weights": result["weights"],
        "ridge_alpha": settings.research.ridge_alpha,
    }
    (data / "learned_ranker.json").write_text(
        json.dumps(_json_safe(ranker), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    registry_path = data / "model_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registry = {"history": []}
    history = [item for item in registry.get("history", []) if isinstance(item, dict)]
    history.append(
        {
            "generated_at": record["generated_at"],
            "spec_version": SPEC_VERSION,
            "labelled_dates": record["labelled_dates"],
            "champion_status": record["champion"]["status"],
            "champion_post_freeze_ic": record["champion"]["post_freeze_ic"]["mean"],
            "challenger_status": record["challenger"]["status"],
            "challenger_oos_ic": record["challenger"]["oos_ic"]["mean"],
            "challenger_oos_dates": record["challenger"]["oos_ic"]["dates"],
            "ic_delta": record["challenger"]["ic_delta_vs_champion"]["mean"],
        }
    )
    registry = {"current": record, "history": history[-104:]}
    registry_path.write_text(
        json.dumps(_json_safe(registry), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_research(
    settings: Settings,
    workers: int | None = None,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    generated_at = datetime.now(ZoneInfo("Asia/Seoul"))
    if panel is None:
        panel = build_research_panel(settings, workers=workers)
    if panel.empty:
        raise RuntimeError("Research panel is empty; at least 60 stored sessions are required")
    result = analyze_panel(panel, settings)
    write_research_outputs(result, settings, generated_at)
    try:
        from .paper import portfolio_backtest

        history = read_csv(settings.data_dir / "history.csv")
        result["portfolio"] = portfolio_backtest(panel, result["predictions"], history, settings, generated_at)
    except Exception as exc:  # the audit must still be written
        LOGGER.warning("Portfolio backtest failed: %s", exc)
    LOGGER.info(
        "Research: champion=%s challenger=%s (OOS dates=%d)",
        result["champion"]["status"],
        result["challenger"]["status"],
        result["challenger"]["oos_ic"]["dates"],
    )
    return result


# ---------------------------------------------------------------------------
# Daily use: score today's eligible names with the stored ranker
# ---------------------------------------------------------------------------


def load_learned_ranker(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("spec_version") != SPEC_VERSION or not payload.get("weights"):
        return None
    return payload


def apply_learned_ranker(scored: pd.DataFrame, ranker: dict[str, Any] | None) -> pd.DataFrame:
    """Attach learned_score (0-100 percentile among eligible names) without touching the rules."""

    result = scored.copy()
    result["learned_score"] = np.nan
    result["learned_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["learned_status"] = ""
    if ranker is None or result.empty:
        return result
    features = tuple(ranker.get("features", LEARNED_FEATURES))
    weights = np.array([float(ranker["weights"].get(name, 0.0)) for name in features])
    eligible = (
        result["discovery_eligible"].fillna(False).astype(bool)
        if "discovery_eligible" in result
        else pd.Series(True, index=result.index)
    )
    pool = result.loc[eligible].copy()
    if len(pool) < 20:
        return result
    for name in features:
        if name not in pool:
            pool[name] = np.nan
    prediction = pd.Series(_design(pool, features) @ weights, index=pool.index)
    result.loc[pool.index, "learned_score"] = (prediction.rank(pct=True) * 100.0).round(1)
    result.loc[pool.index, "learned_rank"] = prediction.rank(method="first", ascending=False).astype(int)
    result["learned_rank"] = result["learned_rank"].astype("Int64")
    result["learned_status"] = str(ranker.get("challenger", {}).get("status", ""))
    return result


def learned_section(scored: pd.DataFrame, ranker: dict[str, Any] | None, top_k: int = 5) -> tuple[str, str]:
    """Markdown and HTML blocks for the daily report."""

    import html as html_lib

    if ranker is None or "learned_rank" not in scored or scored["learned_rank"].isna().all():
        text = "자가검증: 학습 모델 없음 (주간 `research` 실행 후 표시)"
        return f"\n\n## 자가검증 모델\n- {text}\n", f"<section class='panel'><h2>자가검증 모델</h2><p>{html_lib.escape(text)}</p></section>"
    champion = ranker.get("champion", {})
    challenger = ranker.get("challenger", {})
    champion_label = STATUS_LABELS.get(champion.get("status", ""), champion.get("status", ""))
    challenger_label = STATUS_LABELS.get(challenger.get("status", ""), challenger.get("status", ""))
    oos = challenger.get("oos_ic", {})
    header = (
        f"현행 발견점수: {champion_label} · 학습 도전자: {challenger_label} "
        f"(OOS {oos.get('dates', 0)}일, IC {_fmt(oos.get('mean') if oos.get('mean') is not None else float('nan'))})"
    )
    top = scored.dropna(subset=["learned_rank"]).sort_values("learned_rank").head(top_k)
    rows = [
        f"{int(row['learned_rank'])}. {row.get('name', row['symbol'])} ({row['symbol']}) · 학습 백분위 {float(row['learned_score']):.1f} · 발견 {float(row.get('discovery_score', 0.0)):.0f}"
        for _, row in top.iterrows()
    ]
    promoted = challenger.get("status") == "PROMOTE"
    note = (
        "승격 기준을 통과한 순위입니다. 가격 게이트와 함께 참고하세요."
        if promoted
        else "승격 기준 미통과: 매수 근거가 아니라 검증용 기록입니다."
    )
    markdown = "\n\n## 자가검증 모델\n- " + header + "\n- " + note + "\n" + "\n".join(f"- {line}" for line in rows) + "\n"
    html_block = (
        "<section class='panel'><h2>자가검증 모델</h2><p>"
        + html_lib.escape(header)
        + "</p><p>"
        + html_lib.escape(note)
        + "</p><ol>"
        + "".join(f"<li>{html_lib.escape(line.split('. ', 1)[1])}</li>" for line in rows)
        + "</ol></section>"
    )
    return markdown, html_block
