from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import market_forward_benchmarks
from .config import Settings, load_settings
from .eligibility import apply_candidate_eligibility
from .features import add_cross_sectional_residuals, build_base_features
from .ml_alpha import purged_train_test_split, select_daily_top, summarize_selection
from .scoring import score_features
from .storage import read_csv, write_csv

LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_K = 12
DEFAULT_HORIZON = 20

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


def _to_float(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) else float("nan")


def build_rich_feature_history(
    settings: Settings,
    *,
    step: int = 1,
    top_k: int = DEFAULT_TOP_K,
    output_path: Path | None = None,
) -> pd.DataFrame:
    history = read_csv(settings.data_dir / "history.csv")
    universe = read_csv(settings.data_dir / "universe.csv")
    snapshots = read_csv(settings.data_dir / "holdings.csv")
    opinions = read_csv(settings.data_dir / "opinions.csv")
    if history.empty or universe.empty:
        raise ValueError("history.csv and universe.csv are required.")

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["symbol"] = history["symbol"].astype("string").str.zfill(6)
    history = history.dropna(subset=["date"]).sort_values(["date", "symbol"]).reset_index(drop=True)
    if not snapshots.empty and "date" in snapshots:
        snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
    if not opinions.empty and "date" in opinions:
        opinions["date"] = pd.to_datetime(opinions["date"], errors="coerce")

    horizon = int(settings.model.backtest_horizon_days or DEFAULT_HORIZON)
    dates = pd.DatetimeIndex(sorted(history["date"].dropna().unique()))
    if len(dates) < 120 + horizon:
        raise ValueError("At least 140 trading sessions are required for rich feature replay.")

    rows: list[dict[str, Any]] = []
    start_index = 59
    for date_index in range(start_index, len(dates) - horizon, max(1, step)):
        as_of = pd.Timestamp(dates[date_index])
        if date_index % 20 == 0:
            LOGGER.info("Rich feature replay %s (%s/%s)", as_of.date(), date_index, len(dates))
        history_view = history[history["date"] <= as_of]
        snapshot_view = snapshots
        if not snapshots.empty and "date" in snapshots:
            snapshot_view = snapshots[snapshots["date"] <= as_of]
        opinion_view = opinions
        if not opinions.empty and "date" in opinions:
            opinion_view = opinions[opinions["date"] <= as_of]

        features = add_cross_sectional_residuals(
            build_base_features(history_view, universe, snapshot_view, opinion_view)
        )
        scored = score_features(features, settings.model)
        scored = apply_candidate_eligibility(scored, settings.screening, report_date=as_of)
        candidates = (
            scored[scored["discovery_eligible"]]
            .sort_values(
                ["discovery_score", "timing_score", "entry_score"],
                ascending=[False, False, False],
            )
            .head(top_k)
        )
        if candidates.empty:
            continue

        end_date = pd.Timestamp(dates[date_index + horizon])
        market_benchmarks = market_forward_benchmarks(
            history,
            as_of,
            end_date,
            settings.model.backtest_roundtrip_cost_pct / 100.0,
        )
        for _, candidate in candidates.iterrows():
            symbol = str(candidate["symbol"]).zfill(6)
            future = (
                history[(history["symbol"] == symbol) & (history["date"] > as_of)]
                .sort_values("date")
                .head(horizon)
            )
            if len(future) < horizon:
                continue
            entry_raw = pd.to_numeric(future["open"].iloc[0], errors="coerce")
            close_raw = pd.to_numeric(future["close"].iloc[-1], errors="coerce")
            if pd.isna(entry_raw) or pd.isna(close_raw) or float(entry_raw) <= 0:
                continue
            entry = float(entry_raw)
            cost = settings.model.backtest_roundtrip_cost_pct / 100.0
            forward_return = float(close_raw) / entry - 1.0 - cost
            lows = pd.to_numeric(future["low"], errors="coerce").dropna()
            max_drawdown = float((lows / entry - 1.0 - cost).min()) if not lows.empty else np.nan
            market = str(candidate.get("market", ""))
            market_forward = float(
                market_benchmarks.get(
                    market,
                    np.median(list(market_benchmarks.values())) if market_benchmarks else 0.0,
                )
            )
            record: dict[str, Any] = {
                "as_of": as_of,
                "entry_date": future["date"].iloc[0],
                "symbol": symbol,
                "name": candidate.get("name", symbol),
                "market": market,
                "candidate_rank": len(rows),
                "discovery_score": _to_float(candidate.get("discovery_score")),
                "forward_return": forward_return,
                "market_forward_return": market_forward,
                "excess_forward_return": forward_return - market_forward,
                "max_drawdown": max_drawdown,
            }
            for column in RAW_FEATURES:
                record[column] = _to_float(candidate.get(column))
            rows.append(record)

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Rich feature replay produced no training rows.")
    result["candidate_rank"] = (
        result.groupby("as_of")["discovery_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    path = output_path or settings.reports_dir / "ml_rich_features.csv"
    write_csv(result, path)
    return result


def prepare_rich_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "as_of",
        "symbol",
        "discovery_score",
        "forward_return",
        "excess_forward_return",
        "max_drawdown",
        *RAW_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing rich feature columns: {missing}")
    result = frame.copy()
    result["as_of"] = pd.to_datetime(result["as_of"], errors="coerce")
    result["symbol"] = result["symbol"].astype("string").str.zfill(6)
    numeric = list(RAW_FEATURES) + [
        "discovery_score",
        "forward_return",
        "excess_forward_return",
        "max_drawdown",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["as_of", "excess_forward_return", "discovery_score"])
    result = result.drop_duplicates(["as_of", "symbol"], keep="last")
    for source, target in zip(RANK_SOURCES, RANK_FEATURES):
        result[target] = result.groupby("as_of")[source].rank(method="average", pct=True)
    result["target_rank"] = result.groupby("as_of")["excess_forward_return"].rank(
        method="average", pct=True
    )
    return result.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def build_candidate_models() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("ML dependencies are not installed. Run: pip install -e '.[ml]'") from exc

    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=20.0)),
            ]
        ),
        "extra_trees": Pipeline(
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
        ),
        "hist_gradient": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.04,
                        max_iter=180,
                        max_leaf_nodes=9,
                        min_samples_leaf=15,
                        l2_regularization=4.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(MODEL_FEATURES)].copy()


def add_blended_score(frame: pd.DataFrame, prediction: np.ndarray, ml_weight: float) -> pd.DataFrame:
    scored = frame.copy()
    scored["ml_prediction"] = prediction
    scored["ml_day_rank"] = scored.groupby("as_of")["ml_prediction"].rank(
        method="average", pct=True
    )
    scored["discovery_day_rank"] = scored.groupby("as_of")["discovery_score"].rank(
        method="average", pct=True
    )
    scored["rich_rank_score"] = (
        ml_weight * scored["ml_day_rank"]
        + (1.0 - ml_weight) * scored["discovery_day_rank"]
    )
    return scored


def _daily_top_stats(frame: pd.DataFrame, score_column: str) -> tuple[pd.DataFrame, dict[str, float]]:
    selected = select_daily_top(frame, score_column)
    return selected, summarize_selection(selected)


def _paired_bootstrap(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    samples: int = 2000,
) -> dict[str, float]:
    paired = (
        selected[["as_of", "excess_forward_return"]]
        .rename(columns={"excess_forward_return": "model"})
        .merge(
            baseline[["as_of", "excess_forward_return"]].rename(
                columns={"excess_forward_return": "baseline"}
            ),
            on="as_of",
            how="inner",
        )
    )
    differences = (paired["model"] - paired["baseline"]).to_numpy(dtype=float)
    if len(differences) == 0:
        return {"probability_positive": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "probability_positive": float((means > 0).mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _fmt_pct(value: float, *, signed: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:{'+' if signed else ''}.2f}%"


def run_rich_history_model(
    settings: Settings,
    *,
    rebuild_features: bool = True,
    step: int = 1,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    feature_path = settings.reports_dir / "ml_rich_features.csv"
    if rebuild_features or not feature_path.exists():
        raw = build_rich_feature_history(
            settings,
            step=step,
            top_k=top_k,
            output_path=feature_path,
        )
    else:
        raw = pd.read_csv(feature_path, dtype={"symbol": "string"})
    frame = prepare_rich_frame(raw)
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            read_csv(settings.data_dir / "history.csv")["date"],
            errors="coerce",
        ).dropna().unique()
    ).sort_values()

    train, test, outer = purged_train_test_split(
        frame,
        calendar,
        test_fraction=0.30,
        horizon_sessions=DEFAULT_HORIZON,
    )
    inner_train, validation, inner = purged_train_test_split(
        train,
        calendar,
        test_fraction=0.25,
        horizon_sessions=DEFAULT_HORIZON,
    )
    if len(inner_train) < 120 or len(validation) < 60 or len(test) < 100:
        raise ValueError(
            "Rich replay sample is too small for nested time validation: "
            f"inner={len(inner_train)} validation={len(validation)} test={len(test)}"
        )

    candidates: list[dict[str, Any]] = []
    weights = (0.0, 0.25, 0.50, 0.75, 1.0)
    for model_name, model in build_candidate_models().items():
        model.fit(feature_matrix(inner_train), inner_train["target_rank"])
        prediction = model.predict(feature_matrix(validation))
        for weight in weights:
            scored = add_blended_score(validation, prediction, weight)
            _, stats = _daily_top_stats(scored, "rich_rank_score")
            candidates.append(
                {
                    "model": model_name,
                    "weight": weight,
                    "mean_excess": stats["mean_excess"],
                    "mean_return": stats["mean_return"],
                    "positive_excess_rate": stats["positive_excess_rate"],
                    "mean_drawdown": stats["mean_drawdown"],
                }
            )
    tuning = pd.DataFrame(candidates).sort_values(
        ["mean_excess", "positive_excess_rate", "mean_drawdown", "weight"],
        ascending=[False, False, False, True],
    )
    chosen = tuning.iloc[0]
    chosen_name = str(chosen["model"])
    chosen_weight = float(chosen["weight"])

    model = build_candidate_models()[chosen_name]
    model.fit(feature_matrix(train), train["target_rank"])
    test_prediction = model.predict(feature_matrix(test))
    scored_test = add_blended_score(test, test_prediction, chosen_weight)
    selected, selected_stats = _daily_top_stats(scored_test, "rich_rank_score")
    baseline, baseline_stats = _daily_top_stats(scored_test, "discovery_score")

    excess_lift = selected_stats["mean_excess"] - baseline_stats["mean_excess"]
    drawdown_delta = selected_stats["mean_drawdown"] - baseline_stats["mean_drawdown"]
    hit_rate_delta = (
        selected_stats["positive_excess_rate"] - baseline_stats["positive_excess_rate"]
    )
    validation_baseline = add_blended_score(
        validation,
        np.zeros(len(validation)),
        0.0,
    )
    _, validation_baseline_stats = _daily_top_stats(
        validation_baseline,
        "rich_rank_score",
    )
    validation_lift = float(chosen["mean_excess"]) - validation_baseline_stats["mean_excess"]
    bootstrap = _paired_bootstrap(selected, baseline)
    same_pick_rate = float(
        selected.set_index("as_of")["symbol"].eq(
            baseline.set_index("as_of")["symbol"]
        ).mean()
    )
    passed = bool(
        chosen_weight > 0.0
        and validation_lift > 0.0
        and selected_stats["mean_excess"] > 0.0
        and excess_lift > 0.0
        and drawdown_delta >= -0.02
        and hit_rate_delta >= -0.05
        and bootstrap["probability_positive"] >= 0.70
        and selected_stats["dates"] >= 25
    )

    scored_path = settings.reports_dir / "ml_rich_ranking.csv"
    report_path = settings.reports_dir / "ml_rich_ranking.md"
    tuning_path = settings.reports_dir / "ml_rich_tuning.csv"
    write_csv(scored_test, scored_path)
    write_csv(tuning, tuning_path)

    lines = [
        "# V3.1-C 풍부한 과거 특징 모델",
        "",
        "과거 각 거래일에 실제 봇의 특징 계산을 재실행하고, 미래 데이터가 없는 시점 변수만으로 후보 순위를 학습합니다.",
        "모델 종류와 기존 점수 혼합비율은 내부 시간 검증에서만 선택하며, 마지막 30%는 최종 평가 전까지 봉인합니다.",
        "",
        "## 데이터",
        "",
        f"- 전체 후보: {len(frame):,}건 / {frame['as_of'].nunique():,}일",
        f"- 변수: {len(MODEL_FEATURES)}개",
        f"- 내부 학습: {len(inner_train):,}건 (마지막 {inner.train_end.date()})",
        f"- 내부 검증: {len(validation):,}건 (시작 {inner.test_start.date()})",
        f"- 최종 학습: {len(train):,}건 (마지막 {outer.train_end.date()})",
        f"- 봉인 홀드아웃: {len(test):,}건 (시작 {outer.test_start.date()})",
        f"- 각 분할 사이 purge: {DEFAULT_HORIZON} 거래세션",
        "",
        "## 내부 검증에서 고른 정책",
        "",
        f"- 모델: **{chosen_name}**",
        f"- ML 비중: **{chosen_weight:.0%}** / 기존 discovery_score 비중: **{1.0 - chosen_weight:.0%}**",
        f"- 내부 검증 초과수익: {_fmt_pct(float(chosen['mean_excess']))}",
        f"- 내부 검증 기존 점수 대비 개선: {_fmt_pct(validation_lift)}p",
        "",
        "## 봉인 홀드아웃",
        "",
        "| 방식 | 날짜 | 평균 20일 | 시장초과 | 시장 우위 | 평균 최대낙폭 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, stats in (
        ("V3.1-C 하루 1종목", selected_stats),
        ("기존 discovery_score 하루 1종목", baseline_stats),
    ):
        lines.append(
            f"| {label} | {int(stats['dates'])} | {_fmt_pct(stats['mean_return'])} | "
            f"{_fmt_pct(stats['mean_excess'])} | "
            f"{_fmt_pct(stats['positive_excess_rate'], signed=False)} | "
            f"{_fmt_pct(stats['mean_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "## 판정",
            "",
            f"- 기존 점수 대비 초과수익 개선: {_fmt_pct(excess_lift)}p",
            f"- 평균 낙폭 변화: {_fmt_pct(drawdown_delta)}p (양수면 개선)",
            f"- 시장 우위 비율 변화: {_fmt_pct(hit_rate_delta)}p",
            f"- 기존 점수와 같은 종목 선택: {_fmt_pct(same_pick_rate, signed=False)}",
            f"- 일자 부트스트랩 개선 확률: {_fmt_pct(bootstrap['probability_positive'], signed=False)}",
            f"- 개선폭 95% 구간: {_fmt_pct(bootstrap['ci_low'])}p ~ {_fmt_pct(bootstrap['ci_high'])}p",
            f"- 결과: **{'PASS' if passed else 'FAIL'}**",
            "",
            "PASS여도 즉시 주문에는 연결하지 않습니다. 다음 단계는 일일 리포트의 shadow 열에서 기존 점수와 병행 기록하는 것입니다.",
            "",
            "주의: 현재 유니버스를 과거로 재생하므로 상장폐지 종목이 빠지는 생존편향은 남아 있습니다. "
            "모델 변수에서는 현재 재무·현재 의견·종목코드를 제외했습니다.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "passed": passed,
        "model": chosen_name,
        "ml_weight": chosen_weight,
        "rows": len(frame),
        "dates": int(frame["as_of"].nunique()),
        "train_rows": len(train),
        "test_rows": len(test),
        "validation_lift": validation_lift,
        "selected": selected_stats,
        "baseline": baseline_stats,
        "excess_lift": excess_lift,
        "drawdown_delta": drawdown_delta,
        "hit_rate_delta": hit_rate_delta,
        "bootstrap": bootstrap,
        "same_pick_rate": same_pick_rate,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rich historical replay and ranking model")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--reuse-features", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = load_settings(args.root)
    result = run_rich_history_model(
        settings,
        rebuild_features=not args.reuse_features,
        step=args.step,
        top_k=args.top_k,
    )
    print(
        f"Rich ranking: {'PASS' if result['passed'] else 'FAIL'} | "
        f"model={result['model']} weight={result['ml_weight']:.0%} | "
        f"lift={result['excess_lift'] * 100:+.2f}%p | "
        f"rows={result['rows']} dates={result['dates']}"
    )


if __name__ == "__main__":
    main()
