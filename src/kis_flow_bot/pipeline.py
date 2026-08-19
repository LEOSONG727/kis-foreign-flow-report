from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import Settings
from .features import (
    add_cross_sectional_residuals,
    build_base_features,
    normalize_investor_rows,
    normalize_opinions,
    normalize_price_snapshot,
    validate_investor_amounts,
)
from .kis_client import KISAPIError, KISClient
from .notify import deliver_notifications
from .brief import annotate_watch_brief
from .portfolio import apply_portfolio_actions, load_portfolio
from .report import render_report
from .scoring import score_features
from .storage import (
    merge_history,
    merge_snapshots,
    read_csv,
    read_state,
    write_csv,
    write_state,
)
from .universe import load_universe

LOGGER = logging.getLogger(__name__)


def _resolve_investor_query_date(
    client: KISClient,
    requested_as_of: date,
    probe_symbol: str = "005930",
    max_lookback_days: int = 7,
) -> date:
    """Find the latest date accepted by the settled investor-flow endpoint.

    KIS returns OPSQ2001 when the requested calendar date has not been
    settled yet (before the daily cutoff, weekend, or holiday). The endpoint
    itself returns the latest trading rows for an accepted earlier date, so a
    short calendar-day fallback is sufficient and does not require a separate
    exchange-holiday calendar.
    """
    last_error: KISAPIError | None = None
    for days_back in range(max_lookback_days + 1):
        candidate = requested_as_of - timedelta(days=days_back)
        try:
            rows = client.investor_daily(probe_symbol, candidate)
            if rows:
                if days_back:
                    LOGGER.info(
                        "Investor-flow query date adjusted from %s to %s because the newer date is not settled",
                        requested_as_of,
                        candidate,
                    )
                return candidate
        except KISAPIError as exc:
            if exc.code != "OPSQ2001":
                raise
            last_error = exc
            LOGGER.info("Investor-flow date %s is not settled yet; trying the prior day", candidate)
    raise RuntimeError(
        f"Could not find a settled investor-flow date within {max_lookback_days} days of "
        f"{requested_as_of}: {last_error}"
    )


def _select_universe(universe: pd.DataFrame, limit: int | None, watchlist: tuple[str, ...]) -> pd.DataFrame:
    if not limit or limit >= len(universe):
        return universe
    head = universe.sort_values("market_cap_eok", ascending=False).head(limit)
    watched = universe[universe["symbol"].isin(watchlist)]
    return pd.concat([head, watched]).drop_duplicates("symbol").reset_index(drop=True)


def _collect_windows(
    client: KISClient,
    universe: pd.DataFrame,
    as_of: date,
    label: str,
) -> tuple[pd.DataFrame, int, list[str]]:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    total = len(universe)
    for index, (_, meta) in enumerate(universe.iterrows(), 1):
        symbol = str(meta["symbol"]).zfill(6)
        try:
            rows = client.investor_daily(symbol, as_of)
            frame = normalize_investor_rows(symbol, str(meta["name"]), str(meta["market"]), rows)
            if frame.empty:
                failures.append(symbol)
            else:
                frames.append(frame)
        except Exception as exc:  # 종목 하나의 오류가 전체 리포트를 중단시키지 않도록 격리
            failures.append(symbol)
            LOGGER.warning("%s failed for %s: %s", label, symbol, exc)
        if index % 50 == 0 or index == total:
            LOGGER.info("%s progress: %d/%d (failures=%d)", label, index, total, len(failures))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result, total - len(failures), failures


def _features_and_scores(
    history: pd.DataFrame,
    universe: pd.DataFrame,
    snapshots: pd.DataFrame,
    opinions: pd.DataFrame,
    settings: Settings,
    state: dict,
    calibration: dict,
) -> pd.DataFrame:
    features = build_base_features(history, universe, snapshots, opinions)
    features = add_cross_sectional_residuals(features)
    scored = score_features(features, settings.model, state, calibration)
    if scored.empty:
        return scored
    liquid = (
        (scored["adtv20"] >= settings.screening.min_median_turnover_krw)
        & (scored["data_points"] >= 15)
    )
    watched = scored["symbol"].isin(settings.watchlist)
    return scored[liquid | watched].sort_values(
        ["discovery_score", "timing_score", "entry_score"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _backfill_candidates(
    client: KISClient,
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    as_of: date,
    settings: Settings,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    counts = history.groupby("symbol")["date"].nunique().to_dict() if not history.empty else {}
    needed_symbols = [
        str(symbol).zfill(6)
        for symbol in candidates["symbol"].tolist()
        if counts.get(str(symbol).zfill(6), 0) < int(settings.screening.history_days * 0.72)
    ]
    if not needed_symbols:
        LOGGER.info("Long-history bootstrap is already complete for current candidates")
        return pd.DataFrame()
    metadata = candidates.drop_duplicates("symbol").set_index("symbol")
    frames: list[pd.DataFrame] = []
    anchors = range(35, settings.screening.history_days + 1, 28)
    LOGGER.info("Bootstrapping %d candidates to ~%d days", len(needed_symbols), settings.screening.history_days)
    for position, symbol in enumerate(needed_symbols, 1):
        meta = metadata.loc[symbol]
        for days_back in anchors:
            try:
                rows = client.investor_daily(symbol, as_of - timedelta(days=days_back))
                frame = normalize_investor_rows(symbol, str(meta["name"]), str(meta["market"]), rows)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                LOGGER.warning("Backfill failed for %s at -%dd: %s", symbol, days_back, exc)
        if position % 20 == 0 or position == len(needed_symbols):
            LOGGER.info("Backfill progress: %d/%d", position, len(needed_symbols))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_snapshots(
    client: KISClient,
    candidates: pd.DataFrame,
    report_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for index, symbol in enumerate(candidates["symbol"].drop_duplicates().tolist(), 1):
        try:
            raw = client.current_price(str(symbol).zfill(6))
            if raw:
                rows.append(normalize_price_snapshot(str(symbol), report_date, raw))
        except Exception as exc:
            LOGGER.warning("Current-price enrichment failed for %s: %s", symbol, exc)
        if index % 25 == 0:
            LOGGER.info("Snapshot progress: %d/%d", index, len(candidates))
    return pd.DataFrame(rows)


def _collect_opinions(
    client: KISClient,
    candidates: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    frames = []
    start = as_of - timedelta(days=180)
    for symbol in candidates["symbol"].drop_duplicates().tolist():
        try:
            rows = client.invest_opinion(str(symbol).zfill(6), start, as_of)
            frame = normalize_opinions(str(symbol), rows)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            LOGGER.warning("Opinion enrichment failed for %s: %s", symbol, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _merge_opinions(current: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (current, incoming) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["symbol"] = merged["symbol"].astype("string").str.zfill(6)
    return (
        merged.dropna(subset=["date", "symbol"])
        .drop_duplicates(["date", "symbol", "target_price", "opinion"], keep="last")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def _write_outputs(
    scored: pd.DataFrame,
    settings: Settings,
    generated_at: datetime,
) -> Path:
    rendered = render_report(
        scored,
        settings.screening.top_n,
        settings.watchlist,
        generated_at,
        brief=settings.brief,
        model=settings.model,
    )
    report_date = pd.to_datetime(scored["as_of"].max()).strftime("%Y-%m-%d")
    latest_html = settings.reports_dir / "latest.html"
    latest_html.write_text(rendered.html, encoding="utf-8")
    (settings.reports_dir / f"{report_date}.html").write_text(rendered.html, encoding="utf-8")
    (settings.reports_dir / "latest.md").write_text(rendered.markdown, encoding="utf-8")

    csv_columns = [
        "symbol", "name", "market", "as_of", "score", "campaign_score", "entry_score",
        "discovery_score", "timing_score", "distribution_score", "signal_score",
        "price_heat_penalty", "price_heat_flag", "price_heat_reason", "dip_absorption",
        "entry_readiness", "entry_gate_passed", "entry_gate_reason",
        "score_delta", "entry_score_delta", "data_points",
        "success_probability", "expected_value", "stage", "action",
        "campaign_regime", "large_flow_signal", "probability_source",
        "foreign_net1", "foreign_net3", "foreign_net5", "foreign_net20", "foreign_net60", "foreign_net120",
        "foreign_1_adtv", "foreign_3_adtv", "foreign_5_adtv", "foreign_acceleration3", "foreign_acceleration5",
        "foreign_flow_concentration20", "foreign_prior15_adtv",
        "foreign_20_adtv", "tolerant_buy_ratio20", "sellback_ratio20",
        "price_return1", "price_return3", "price_return5", "price_return10",
        "price_return20", "price_return60", "price_return120", "price_residual3", "price_residual5", "price_residual10", "price_residual20",
        "rebound_from_low10", "rebound_from_low20", "rebound_atr_distance20",
        "distance_high20", "ma20_distance", "ma20_atr_distance",
        "above_ma20", "touched_ma20", "touched_ma20_3", "near_ma20", "ma20_support",
        "up_days5", "up_days10", "price_up_streak", "up_down_volume_ratio10",
        "price_acceleration3", "ma5_slope", "above_ma5", "above_ma10", "market_return20",
        "absorption_signal", "volume_ratio5_20", "rebound_volume_ratio20", "atr20_pct",
        "accumulation_score", "underreaction_score", "structure_score",
        "fundamental_score", "institution_overlay", "roe", "per", "pbr",
        "target_price", "target_upside", "foreign_holding_change",
        "foreign_ownership_days_adtv", "foreign_holding_change_adtv",
        "opinion_coverage", "opinion_recency_days", "opinion_dispersion",
        "history_confidence",
        "model_action", "has_position", "position_quantity", "average_price",
        "last_buy_price", "tranche_count", "position_return", "position_drawdown",
        "add_score", "decision_bucket", "action_priority", "promotion_condition",
        "invalidation_condition", "suggested_order_krw",
        "watch_tier", "watch_priority", "flow_freshness", "flow_summary", "block_reason",
        "next_trigger", "high_conviction", "in_daily_brief",
        "cheap_accumulation", "steady_accumulation",
        "foreign_roll20_up_days10", "foreign_roll20_down_days10",
        "foreign_roll20_up_days20", "foreign_roll20_up_streak", "foreign_roll20_today_up",
        "screen_bucket",
    ]
    write_csv(scored[[column for column in csv_columns if column in scored]], settings.reports_dir / "latest.csv")
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(rendered.markdown)
    return latest_html


def _new_state(scored: pd.DataFrame, previous: dict, report_date: str) -> dict:
    old = previous.get("symbols", {})
    symbols = {}
    for _, row in scored.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        symbols[symbol] = {
            "name": str(row["name"]),
            "score": float(row["score"]),
            "entry_score": float(row.get("entry_score", 0.0)),
            "campaign_score": float(row.get("campaign_score", 0.0)),
            "discovery_score": float(row.get("discovery_score", 0.0)),
            "timing_score": float(row.get("timing_score", 0.0)),
            "distribution_score": float(row.get("distribution_score", 0.0)),
            "stage": str(row["stage"]),
            "campaign_regime": str(row.get("campaign_regime", "중립")),
            "action": str(row["action"]),
            "first_seen": old.get(symbol, {}).get("first_seen", report_date),
            "last_seen": report_date,
        }
    return {"report_date": report_date, "symbols": symbols}


def run_pipeline(
    settings: Settings,
    as_of: date | None = None,
    universe_limit: int | None = None,
    skip_backfill: bool = False,
    full_history_bootstrap: bool = False,
    notify: bool = True,
) -> pd.DataFrame:
    generated_at = datetime.now(ZoneInfo("Asia/Seoul"))
    as_of = as_of or generated_at.date()
    client = KISClient.from_env(settings.api)
    client.authenticate()

    universe = load_universe(settings.screening, settings.watchlist, client.session)
    universe = _select_universe(universe, universe_limit, settings.watchlist)
    write_csv(universe, settings.data_dir / "universe.csv")
    as_of = _resolve_investor_query_date(client, as_of)

    history_path = settings.data_dir / "history.csv"
    snapshots_path = settings.data_dir / "holdings.csv"
    opinions_path = settings.data_dir / "opinions.csv"
    state_path = settings.data_dir / "state.json"
    calibration_path = settings.data_dir / "calibration.json"
    portfolio_path = settings.data_dir / "portfolio.csv"
    history = read_csv(history_path)
    snapshots = read_csv(snapshots_path)
    opinions = read_csv(opinions_path)
    state = read_state(state_path)
    calibration = read_state(calibration_path)
    portfolio = load_portfolio(portfolio_path)

    incoming, success_count, failures = _collect_windows(client, universe, as_of, "Daily scan")
    success_rate = success_count / max(len(universe), 1)
    if incoming.empty or success_rate < 0.70:
        raise RuntimeError(
            f"Daily scan data quality too low: success={success_count}/{len(universe)}, failures={failures[:10]}"
        )
    validate_investor_amounts(incoming)
    history = merge_history(
        history, incoming, settings.screening.history_retention_days, as_of
    )

    preliminary = _features_and_scores(
        history, universe, snapshots, opinions, settings, state, calibration
    )
    if preliminary.empty:
        raise RuntimeError("No securities passed the preliminary data and liquidity checks")

    if not skip_backfill:
        if full_history_bootstrap:
            candidates = universe.copy()
            LOGGER.info("Full-market history bootstrap requested: %d securities", len(candidates))
        else:
            candidate_symbols = list(
                dict.fromkeys(
                    preliminary.head(settings.screening.bootstrap_candidates)["symbol"].tolist()
                    + list(settings.watchlist)
                )
            )
            candidates = universe[universe["symbol"].isin(candidate_symbols)]
        backfill = _backfill_candidates(client, candidates, history, as_of, settings)
        if not backfill.empty:
            history = merge_history(
                history, backfill, settings.screening.history_retention_days, as_of
            )
            preliminary = _features_and_scores(
                history, universe, snapshots, opinions, settings, state, calibration
            )

    enrichment_symbols = list(
        dict.fromkeys(
            preliminary.head(settings.screening.holdings_candidates)["symbol"].tolist()
            + list(settings.watchlist)
        )
    )
    enrichment = universe[universe["symbol"].isin(enrichment_symbols)]
    report_date = pd.to_datetime(history["date"]).max()
    new_snapshots = _collect_snapshots(client, enrichment, report_date)
    snapshots = merge_snapshots(snapshots, new_snapshots)

    opinion_symbols = list(
        dict.fromkeys(
            preliminary.head(settings.screening.opinion_candidates)["symbol"].tolist()
            + list(settings.watchlist)
        )
    )
    opinion_candidates = universe[universe["symbol"].isin(opinion_symbols)]
    new_opinions = _collect_opinions(client, opinion_candidates, as_of)
    opinions = _merge_opinions(opinions, new_opinions)

    scored = _features_and_scores(
        history, universe, snapshots, opinions, settings, state, calibration
    )
    if scored.empty:
        raise RuntimeError("No securities passed final scoring")
    scored = apply_portfolio_actions(scored, portfolio, settings.model)
    scored = annotate_watch_brief(scored, settings.model, settings.brief)

    write_csv(history, history_path)
    write_csv(snapshots, snapshots_path)
    write_csv(opinions, opinions_path)
    latest_html = _write_outputs(scored, settings, generated_at)
    report_date_text = pd.to_datetime(scored["as_of"].max()).strftime("%Y-%m-%d")
    write_state(_new_state(scored, state, report_date_text), state_path)

    if notify:
        deliver_notifications(scored, latest_html, settings.brief, settings.model)
    LOGGER.info("Report complete: %s (%d ranked securities)", latest_html, len(scored))
    return scored


def rerank_stored_data(settings: Settings, notify: bool = False) -> pd.DataFrame:
    """Rebuild scores and reports from stored market data without calling KIS."""
    generated_at = datetime.now(ZoneInfo("Asia/Seoul"))
    history = read_csv(settings.data_dir / "history.csv")
    universe = read_csv(settings.data_dir / "universe.csv")
    snapshots = read_csv(settings.data_dir / "holdings.csv")
    opinions = read_csv(settings.data_dir / "opinions.csv")
    state_path = settings.data_dir / "state.json"
    state = read_state(state_path)
    calibration = read_state(settings.data_dir / "calibration.json")
    portfolio = load_portfolio(settings.data_dir / "portfolio.csv")
    if history.empty or universe.empty:
        raise RuntimeError("Stored history.csv or universe.csv is empty; run the daily pipeline first")

    scored = _features_and_scores(
        history, universe, snapshots, opinions, settings, state, calibration
    )
    if scored.empty:
        raise RuntimeError("No securities passed stored-data scoring")
    scored = apply_portfolio_actions(scored, portfolio, settings.model)
    scored = annotate_watch_brief(scored, settings.model, settings.brief)
    latest_html = _write_outputs(scored, settings, generated_at)
    report_date = pd.to_datetime(scored["as_of"].max()).strftime("%Y-%m-%d")
    write_state(_new_state(scored, state, report_date), state_path)
    if notify:
        deliver_notifications(scored, latest_html, settings.brief, settings.model)
    LOGGER.info("Stored-data rerank complete: %s (%d ranked securities)", latest_html, len(scored))
    return scored


def smoke_test(settings: Settings) -> None:
    client = KISClient.from_env(settings.api)
    client.authenticate()
    samsung = client.current_price("005930")
    requested_as_of = datetime.now(ZoneInfo("Asia/Seoul")).date()
    query_as_of = _resolve_investor_query_date(client, requested_as_of)
    celltrion = client.investor_daily("068270", query_as_of)
    price = samsung.get("stck_prpr", "?")
    LOGGER.info("Authentication OK; Samsung current price=%s", price)
    LOGGER.info("Celltrion investor history rows=%d", len(celltrion))
    if not samsung or not celltrion:
        raise KISAPIError("Smoke test returned an empty market-data response")
