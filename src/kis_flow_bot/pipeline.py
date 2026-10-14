from __future__ import annotations

import logging
import html
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .brief import annotate_watch_brief
from .config import Settings
from .eligibility import apply_candidate_eligibility
from .features import (
    add_cross_sectional_residuals,
    build_base_features,
    normalize_investor_rows,
    normalize_opinions,
    normalize_price_snapshot,
    validate_investor_amounts,
)
from .horizon import annotate_horizon, empty_horizon_state
from .jev_shadow import OUTPUTS as JEV_OUTPUTS, apply_jev_shadow
from .jev_experiment import update_jev_experiment, write_jev_comparison
from .jev_monitor import monitor as monitor_jev, save_monitor
from .earnings_quality import apply_earnings_quality, eq_section
from .jev_events import apply_event_review, event_section
from .jev_replay import daily_jev_allowed, load_jev_verdict
from .kis_client import KISAPIError, KISClient
from .live_tracking import update_live_tracking
from .ml_shadow import apply_rich_shadow, update_shadow_tracking
from .notify import deliver_notifications
from .paper import paper_section, update_paper_trading
from .portfolio import apply_portfolio_actions, load_portfolio
from .report import render_report
from .fundamentals import attach_fundamentals, load_financials
from .research import apply_learned_ranker, learned_section, load_learned_ranker
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
    features = build_base_features(history, universe, snapshots, opinions, point_in_time=True)
    features = attach_fundamentals(features, load_financials(settings.data_dir))
    features = add_cross_sectional_residuals(features)
    scored = score_features(features, settings.model, state, calibration)
    if scored.empty:
        return scored
    scored = apply_candidate_eligibility(scored, settings.screening)
    watched = scored["symbol"].isin(settings.watchlist)
    return scored[scored["discovery_eligible"] | watched].sort_values(
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


def _horizon_previous(path: Path) -> dict:
    raw = read_state(path)
    if "mid_core" in raw or "tactical" in raw or "long_core" in raw:
        return raw
    return empty_horizon_state()


def _write_outputs(
    scored: pd.DataFrame,
    settings: Settings,
    generated_at: datetime,
    horizon=None,
    paper_summary: pd.DataFrame | None = None,
) -> Path:
    rendered = render_report(
        scored,
        settings.screening.top_n,
        settings.watchlist,
        generated_at,
        brief=settings.brief,
        model=settings.model,
        horizon=horizon,
    )
    report_date = pd.to_datetime(scored["as_of"].max()).strftime("%Y-%m-%d")
    latest_html = settings.reports_dir / "latest.html"
    shadow_rows = scored[scored["jev_review"].notna()] if "jev_review" in scored else scored.head(0)
    shadow_lines = [
        f"{row['name']} ({row['symbol']}): {row['jev_review']} · 수급 취약 {row['jev_fragile_flow']:.2f} · 하락 함정 {row['jev_price_trap']:.2f} · 추격 위험 {row['jev_chase_risk']:.2f}"
        for _, row in shadow_rows.head(20).iterrows()
    ]
    status_counts = scored["jev_status"].value_counts().to_dict() if "jev_status" in scored else {}
    shadow_md = "\n\n## Jev Shadow · 참고용\n" + ("\n".join(f"- {line}" for line in shadow_lines) if shadow_lines else "- 평가 결과 없음")
    if status_counts and set(status_counts) == {"VALIDATION_OFF"}:
        shadow_lines = ["과거 재생 검증에서 추가 가치가 없어 일일 호출을 멈췄습니다 (reports/jev_replay.md)"]
        shadow_md = "\n\n## Jev Shadow · 참고용\n- " + shadow_lines[0]
    shadow_md += "\n진단 상태: " + str(status_counts)
    shadow_html = "<section class='panel'><h2>Jev Shadow · 참고용</h2><p>숫자가 높을수록 해당 위험 패턴에 대한 동의가 강합니다. 손실 확률은 아닙니다. 기존 매수 후보와 순위에는 반영되지 않습니다.</p><ul>" + "".join(
        f"<li>{html.escape(line)}</li>" for line in shadow_lines
    ) + "</ul></section>"
    shadow_html += "<p>진단 상태: " + html.escape(str(status_counts)) + "</p>"
    monitor_note = scored.attrs.get("jev_monitor_note", "")
    if monitor_note:
        shadow_html += "<section><h2>모델 감독</h2><p>" + html.escape(monitor_note).replace("\n", "<br>") + "</p></section>"
        shadow_md += "\n\n" + monitor_note
    data_dir = getattr(settings, "data_dir", None)
    learned_md, learned_html = learned_section(
        scored, load_learned_ranker(data_dir / "learned_ranker.json") if data_dir else None
    )
    event_md, event_html = event_section(
        scored, load_jev_verdict(data_dir / "jev_event_verdict.json", as_of=scored["as_of"].max()) if data_dir else None
    )
    paper_md, paper_html = paper_section(paper_summary)
    eq_md, eq_html = eq_section(
        scored, load_jev_verdict(data_dir / "earnings_quality_verdict.json", as_of=scored["as_of"].max()) if data_dir else None
    )
    rendered_html = rendered.html.replace(
        "</main>", paper_html + eq_html + event_html + learned_html + shadow_html + "</main>"
    )
    rendered_markdown = rendered.markdown + paper_md + eq_md + event_md + learned_md + shadow_md + "\n"
    latest_html.write_text(rendered_html, encoding="utf-8")
    (settings.reports_dir / f"{report_date}.html").write_text(rendered_html, encoding="utf-8")
    (settings.reports_dir / "latest.md").write_text(rendered_markdown, encoding="utf-8")

    csv_columns = list(JEV_OUTPUTS) + [
        "symbol", "name", "market", "as_of", "score", "campaign_score", "entry_score",
        "discovery_score", "timing_score", "distribution_score", "signal_score",
        "ml_shadow_available", "ml_shadow_model", "ml_shadow_status", "ml_shadow_prediction",
        "ml_shadow_score", "ml_shadow_rank", "ml_shadow_top", "ml_shadow_baseline_rank",
        "learned_score", "learned_rank", "learned_status", "pit_size",
        "op_yoy_mcap", "op_yield", "book_to_market", "revenue_yoy", "report_age",
        "financial_rcept_no", "financial_available",
        "event_filings", "event_dilution", "event_adverse", "event_catalyst",
        "kw_dilution", "kw_adverse", "kw_catalyst",
        "eq_one_off", "eq_recurring", "eq_kw_veto",
        "jev_fragile_flow", "jev_price_trap", "jev_chase_risk", "jev_review",
        "jev_review_confidence", "jev_model", "jev_error",
        "price_heat_penalty", "price_heat_flag", "price_heat_reason", "dip_absorption",
        "entry_readiness", "entry_gate_passed", "entry_gate_reason",
        "score_delta", "entry_score_delta", "data_points", "adtv20",
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
        "ema20", "ma20_slope", "ema20_slope", "ma_spread_pct",
        "ma_aligned_up", "ma_aligned_down", "ma_compressed", "ma_fan_up",
        "ma20_rising", "ma20_falling", "flow_price_divergence",
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
        "next_trigger", "high_conviction", "discovery_grade", "proposal_status",
        "proposal_priority", "in_daily_brief",
        "data_fresh", "liquidity_eligible", "history_eligible", "discovery_eligible",
        "eligibility_reason",
        "cheap_accumulation", "steady_accumulation", "steady_price_ok",
        "foreign_buy_days10", "foreign_sell_days10", "foreign_buy_days20",
        "foreign_buy_streak", "foreign_bought_today",
        "foreign_inventory_up_days10", "foreign_inventory_down_days10",
        # Legacy aliases kept during the CSV migration.
        "foreign_roll20_up_days10", "foreign_roll20_down_days10",
        "foreign_roll20_up_days20", "foreign_roll20_up_streak", "foreign_roll20_today_up",
        "screen_bucket",
        "mid_horizon_score", "long_horizon_score", "in_mid_core", "in_long_core",
    ]
    write_csv(scored[[column for column in dict.fromkeys(csv_columns) if column in scored]], settings.reports_dir / "latest.csv")
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(rendered_markdown)
    return latest_html


def _write_jev_shadow(scored: pd.DataFrame, reports_dir: Path) -> None:
    """Store a separate shadow ledger without affecting candidate selection."""
    columns = ["as_of", "symbol", "name", "discovery_score", "proposal_status", *JEV_OUTPUTS]
    if "jev_status" not in scored:
        return
    observed = scored[~scored["jev_status"].isin(["NOT_SELECTED", "DISABLED", "VALIDATION_OFF"])]
    write_csv(observed[[col for col in columns if col in observed]], reports_dir / "jev_shadow_latest.csv")


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
    scored = apply_rich_shadow(
        scored,
        settings.reports_dir / "ml_rich_features.csv",
        trading_dates=history["date"],
        embargo_sessions=settings.research.embargo_sessions,
    )
    scored = attach_fundamentals(scored, load_financials(settings.data_dir))
    scored = apply_learned_ranker(
        scored, load_learned_ranker(settings.data_dir / "learned_ranker.json")
    )
    scored = apply_portfolio_actions(scored, portfolio, settings.model)
    scored = annotate_watch_brief(scored, settings.model, settings.brief)
    horizon_path = settings.data_dir / "horizon_core.json"
    scored, horizon_board = annotate_horizon(
        scored,
        _horizon_previous(horizon_path),
        read_csv(settings.data_dir / "signal_snapshots.csv"),
        settings.brief,
        settings.model,
        settings.horizon,
    )

    jev_verdict = load_jev_verdict(settings.data_dir / "jev_verdict.json")
    if settings.jev.daily_calls and daily_jev_allowed(jev_verdict):
        scored = apply_jev_shadow(scored)
    else:
        # Replays showed Jev adds nothing over code rules; [jev] daily_calls is off.
        scored = apply_jev_shadow(scored, api_key="")
        scored["jev_status"] = "VALIDATION_OFF"
    scored = apply_event_review(scored, settings)
    scored = apply_earnings_quality(scored, settings)

    write_csv(history, history_path)
    write_csv(snapshots, snapshots_path)
    write_csv(opinions, opinions_path)
    jev_ledger = pd.DataFrame()
    tracking_error = False
    _write_jev_shadow(scored, settings.reports_dir)
    try:
        jev_ledger = update_jev_experiment(scored, history, settings.data_dir / "jev_experiment.csv",
            roundtrip_cost_pct=settings.model.backtest_roundtrip_cost_pct)
        write_jev_comparison(jev_ledger, settings.reports_dir / "jev_experiment_latest.md")
    except Exception as exc:
        tracking_error = True
        LOGGER.warning("Jev experiment tracking failed without blocking the report: %s", type(exc).__name__)
    try:
        health = monitor_jev(
            scored,
            jev_ledger,
            verdict=jev_verdict,
            registry=read_state(settings.data_dir / "model_registry.json"),
        )
        if tracking_error:
            health.update(decision="DIAGNOSE", source="code_tracking_guard")
            health["note"] = "모델 감독: 성과 기록 오류 점검 필요\n이번 실행의 성과 추적에 실패했습니다."
        scored.attrs["jev_monitor_note"] = health["note"]
        save_monitor(health, settings.reports_dir / "jev_monitor_latest.json")
    except Exception as exc:
        scored.attrs["jev_monitor_note"] = "모델 감독: 감독 처리 오류 점검 필요 (" + type(exc).__name__ + ")"
    # Keep DataFrames out of DataFrame.attrs: pandas compares attrs with ==
    # during concat, and a frame there makes every later concat raise.
    paper_summary = None
    try:
        paper_summary = update_paper_trading(scored, history, settings, generated_at)
    except Exception as exc:  # paper trading must never block the report
        LOGGER.warning("Paper trading update failed without blocking the report: %s", exc)
    latest_html = _write_outputs(scored, settings, generated_at, horizon_board, paper_summary)
    try:
        shadow_ledger = update_shadow_tracking(
            scored,
            history,
            settings.data_dir / "ml_shadow_snapshots.csv",
            roundtrip_cost_pct=settings.model.backtest_roundtrip_cost_pct,
        )
        LOGGER.info(
            "ML shadow tracking updated: rows=%d resolved=%d",
            len(shadow_ledger),
            int(shadow_ledger.get("resolved", pd.Series(dtype=bool)).fillna(False).sum()),
        )
    except Exception as exc:
        LOGGER.warning("ML shadow tracking update failed without blocking the report: %s", exc)
    tracking = update_live_tracking(scored, history, settings, generated_at)
    if tracking:
        LOGGER.info(
            "Live evaluation updated: episodes=%d events=%d resolved20=%d",
            tracking.episodes,
            tracking.events,
            tracking.resolved_20,
        )
    report_date_text = pd.to_datetime(scored["as_of"].max()).strftime("%Y-%m-%d")
    write_state(_new_state(scored, state, report_date_text), state_path)
    write_state(horizon_board.to_state(), horizon_path)

    if notify:
        deliver_notifications(scored, latest_html, settings.brief, settings.model, horizon_board)
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
    scored = apply_rich_shadow(
        scored,
        settings.reports_dir / "ml_rich_features.csv",
        trading_dates=history["date"],
        embargo_sessions=settings.research.embargo_sessions,
    )
    scored = attach_fundamentals(scored, load_financials(settings.data_dir))
    scored = apply_learned_ranker(
        scored, load_learned_ranker(settings.data_dir / "learned_ranker.json")
    )
    scored = apply_portfolio_actions(scored, portfolio, settings.model)
    scored = annotate_watch_brief(scored, settings.model, settings.brief)
    horizon_path = settings.data_dir / "horizon_core.json"
    scored, horizon_board = annotate_horizon(
        scored,
        _horizon_previous(horizon_path),
        read_csv(settings.data_dir / "signal_snapshots.csv"),
        settings.brief,
        settings.model,
        settings.horizon,
    )
    latest_html = _write_outputs(scored, settings, generated_at, horizon_board)
    try:
        update_shadow_tracking(
            scored,
            history,
            settings.data_dir / "ml_shadow_snapshots.csv",
            roundtrip_cost_pct=settings.model.backtest_roundtrip_cost_pct,
        )
    except Exception as exc:
        LOGGER.warning("ML shadow tracking update failed without blocking rerank: %s", exc)
    report_date = pd.to_datetime(scored["as_of"].max()).strftime("%Y-%m-%d")
    write_state(_new_state(scored, state, report_date), state_path)
    write_state(horizon_board.to_state(), horizon_path)
    if notify:
        deliver_notifications(scored, latest_html, settings.brief, settings.model, horizon_board)
    LOGGER.info("Stored-data rerank complete: %s (%d ranked securities)", latest_html, len(scored))
    return scored


def _backfill_symbol_older(
    client: KISClient,
    symbol: str,
    name: str,
    market: str,
    earliest: date,
    target: date,
    max_calls: int = 40,
) -> pd.DataFrame:
    """Walk one symbol backwards from its earliest stored day until ``target``.

    Each KIS call returns the rows ending at the requested date, so the next
    anchor is simply the day before the oldest row received. The walk stops
    when a call adds nothing older (listing date or API depth reached).
    """

    frames: list[pd.DataFrame] = []
    anchor = earliest - timedelta(days=1)
    for _ in range(max_calls):
        if anchor < target:
            break
        rows = client.investor_daily(symbol, anchor)
        frame = normalize_investor_rows(symbol, name, market, rows)
        if frame.empty:
            break
        oldest = pd.to_datetime(frame["date"]).min().date()
        frames.append(frame)
        if oldest >= anchor + timedelta(days=1) or oldest > earliest:
            break
        anchor = oldest - timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def deep_backfill(
    settings: Settings,
    days: int | None = None,
    max_minutes: float = 150.0,
    universe_limit: int | None = None,
) -> pd.DataFrame:
    """Extend stored history backwards for the whole universe (resumable).

    More history is the only way to make the weekly self-validation meaningful:
    20-session outcomes overlap, so ~6 months hold only ~6 independent periods.
    The run saves progress every 50 symbols and stops at ``max_minutes``; the
    next run continues from each symbol's current earliest date.
    """

    started = datetime.now(ZoneInfo("Asia/Seoul"))
    days = min(int(days or settings.screening.history_retention_days), settings.screening.history_retention_days)
    history_path = settings.data_dir / "history.csv"
    history = read_csv(history_path)
    if history.empty:
        raise RuntimeError("Run the daily pipeline once before deep-backfill")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["symbol"] = history["symbol"].astype("string").str.zfill(6)
    universe = read_csv(settings.data_dir / "universe.csv")
    universe["symbol"] = universe["symbol"].astype("string").str.zfill(6)
    if universe_limit:
        universe = universe.head(universe_limit)
    latest = history["date"].max().date()
    target = latest - timedelta(days=days)
    earliest_by_symbol = history.groupby("symbol")["date"].min().dt.date.to_dict()
    pending = [
        row
        for _, row in universe.iterrows()
        if earliest_by_symbol.get(str(row["symbol"]), latest) > target + timedelta(days=7)
    ]
    LOGGER.info(
        "Deep backfill to %s: %d/%d symbols need older history",
        target,
        len(pending),
        len(universe),
    )
    if not pending:
        return history
    client = KISClient.from_env(settings.api)
    client.authenticate()
    frames: list[pd.DataFrame] = []

    def flush() -> None:
        nonlocal history, frames
        if not frames:
            return
        history = merge_history(
            history,
            pd.concat(frames, ignore_index=True),
            settings.screening.history_retention_days,
            latest,
        )
        write_csv(history, history_path)
        frames = []

    for position, meta in enumerate(pending, 1):
        symbol = str(meta["symbol"]).zfill(6)
        try:
            frame = _backfill_symbol_older(
                client,
                symbol,
                str(meta["name"]),
                str(meta["market"]),
                earliest_by_symbol.get(symbol, latest + timedelta(days=1)),
                target,
            )
            if not frame.empty:
                frames.append(frame)
            if len(pending) <= 20:
                LOGGER.info(
                    "Deep backfill %s: oldest %s (was %s)",
                    symbol,
                    pd.to_datetime(frame["date"]).min().date() if not frame.empty else "no older rows",
                    earliest_by_symbol.get(symbol),
                )
        except Exception as exc:
            LOGGER.warning("Deep backfill failed for %s: %s", symbol, exc)
        if position % 50 == 0:
            flush()
            LOGGER.info("Deep backfill progress: %d/%d", position, len(pending))
        elapsed = (datetime.now(ZoneInfo("Asia/Seoul")) - started).total_seconds() / 60.0
        if elapsed >= max_minutes:
            LOGGER.info("Deep backfill time budget reached at %d/%d; resume next run", position, len(pending))
            break
    flush()
    LOGGER.info(
        "Deep backfill stored %s ~ %s (%d sessions)",
        history["date"].min().date(),
        history["date"].max().date(),
        history["date"].nunique(),
    )
    return history


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


