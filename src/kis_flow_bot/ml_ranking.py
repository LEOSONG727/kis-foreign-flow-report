from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ml_alpha import (
    BASELINE_SCORE_COLUMN,
    DEFAULT_HORIZON_SESSIONS,
    DEFAULT_TEST_FRACTION,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    load_trading_calendar,
    prepare_training_frame,
    purged_train_test_split,
    select_daily_top,
    summarize_selection,
)

RAW_FEATURES: tuple[str, ...] = (
    "discovery_score",
    "foreign_20_adtv",
    "price_return20",
    "rebound_from_low20",
    "ma20_distance",
    "cheap_accumulation",
    "dip_absorption",
)
RANK_FEATURES: tuple[str, ...] = (
    "discovery_score_day_rank",
    "foreign_20_adtv_day_rank",
    "price_return20_day_rank",
    "rebound_from_low20_day_rank",
    "ma20_distance_day_rank",
)
MODEL_FEATURES: tuple[str, ...] = RAW_FEATURES + RANK_FEATURES + ("market_kosdaq",)
TARGET = "excess_forward_return"
MIN_PAIR_GAP = 0.005


def add_point_in_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["market_kosdaq"] = result["market"].astype("string").eq("KOSDAQ").astype(float)
    for source, target in zip(RAW_FEATURES[:5], RANK_FEATURES):
        values = pd.to_numeric(result[source], errors="coerce")
        result[target] = values.groupby(result["as_of"]).rank(method="average", pct=True)
    return result


def make_pairwise_training_set(
    frame: pd.DataFrame,
    *,
    min_pair_gap: float = MIN_PAIR_GAP,
) -> tuple[pd.DataFrame, pd.Series]:
    featured = add_point_in_time_features(frame)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for _, group in featured.groupby("as_of", sort=True):
        valid = group.dropna(subset=[TARGET])
        for left_index, right_index in combinations(range(len(valid)), 2):
            left = valid.iloc[left_index]
            right = valid.iloc[right_index]
            target_gap = float(left[TARGET]) - float(right[TARGET])
            if not np.isfinite(target_gap) or abs(target_gap) < min_pair_gap:
                continue
            left_values = pd.to_numeric(left[list(MODEL_FEATURES)], errors="coerce").to_numpy(dtype=float)
            right_values = pd.to_numeric(right[list(MODEL_FEATURES)], errors="coerce").to_numpy(dtype=float)
            difference = left_values - right_values
            label = int(target_gap > 0)
            rows.extend((difference, -difference))
            labels.extend((label, 1 - label))
    if not rows:
        raise ValueError("No usable within-day candidate pairs were found.")
    pairs = pd.DataFrame(rows, columns=MODEL_FEATURES)
    return pairs, pd.Series(labels, name="left_wins", dtype=int)


def build_pairwise_ranker() -> Any:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("ML dependencies are not installed. Run: pip install -e '.[ml]'") from exc
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "ranker",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def score_candidates(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    scored = add_point_in_time_features(frame)
    scored["pairwise_rank_score"] = model.decision_function(scored[list(MODEL_FEATURES)])
    return scored


def _fmt_pct(value: float, *, signed: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:{'+' if signed else ''}.2f}%"


def _coefficient_table(model: Any) -> pd.DataFrame:
    try:
        coefficients = model.named_steps["ranker"].coef_[0]
    except Exception:
        return pd.DataFrame(columns=["feature", "coefficient"])
    return (
        pd.DataFrame({"feature": MODEL_FEATURES, "coefficient": coefficients})
        .assign(abs_coefficient=lambda item: item["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )


def run_pairwise_ranking(
    screen_backtest_path: Path,
    history_path: Path,
    output_csv: Path,
    output_report: Path,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
) -> dict[str, Any]:
    raw = pd.read_csv(screen_backtest_path, dtype={"symbol": "string"})
    frame = prepare_training_frame(raw)
    calendar = load_trading_calendar(history_path)
    train, test, split = purged_train_test_split(
        frame,
        calendar,
        test_fraction=test_fraction,
        horizon_sessions=horizon_sessions,
    )
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(f"Training sample is too small: {len(train)} < {MIN_TRAIN_ROWS}")
    if len(test) < MIN_TEST_ROWS:
        raise ValueError(f"Test sample is too small: {len(test)} < {MIN_TEST_ROWS}")

    pair_x, pair_y = make_pairwise_training_set(train)
    model = build_pairwise_ranker()
    model.fit(pair_x, pair_y)

    scored = score_candidates(model, test)
    scored = scored.sort_values(["as_of", "pairwise_rank_score"], ascending=[True, False])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False, encoding="utf-8-sig")

    rank_top = select_daily_top(scored, "pairwise_rank_score")
    baseline_top = select_daily_top(scored, BASELINE_SCORE_COLUMN)
    oracle_top = select_daily_top(scored, TARGET)
    rank_stats = summarize_selection(rank_top)
    baseline_stats = summarize_selection(baseline_top)
    oracle_stats = summarize_selection(oracle_top)

    test_pair_x, test_pair_y = make_pairwise_training_set(test)
    pair_accuracy = float(model.score(test_pair_x, test_pair_y))
    same_pick_rate = float(
        rank_top.set_index("as_of")["symbol"].eq(
            baseline_top.set_index("as_of")["symbol"]
        ).mean()
    )
    excess_lift = rank_stats["mean_excess"] - baseline_stats["mean_excess"]
    drawdown_delta = rank_stats["mean_drawdown"] - baseline_stats["mean_drawdown"]
    hit_rate_delta = rank_stats["positive_excess_rate"] - baseline_stats["positive_excess_rate"]
    passed = bool(
        excess_lift > 0.0
        and drawdown_delta >= -0.02
        and hit_rate_delta >= -0.05
        and pair_accuracy > 0.50
    )

    lines = [
        "# V3.1-B Pairwise 랭킹 검증",
        "",
        "같은 신호일의 후보 두 종목을 비교해 어느 종목의 20거래일 시장초과수익이 더 높을지 학습합니다.",
        "실전 추천·알림·주문 로직에는 반영하지 않는 연구 전용 결과입니다.",
        "",
        "## 검증 구조",
        "",
        f"- 학습 마지막 신호일: {split.train_end.date()}",
        f"- 테스트 시작일: {split.test_start.date()}",
        f"- purge: {horizon_sessions} 거래세션",
        f"- 학습 후보 {len(train)}건 / 학습 pair {len(pair_x)}건 / 테스트 후보 {len(test)}건",
        f"- 0.5%p 미만의 초과수익 차이는 잡음으로 보고 pair 학습에서 제외",
        "",
        "## 홀드아웃 결과",
        "",
        "| 방식 | 날짜 | 평균 20일 | 시장초과 | 시장 우위 | 평균 최대낙폭 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, stats in (
        ("Pairwise 랭커 하루 1종목", rank_stats),
        ("기존 discovery_score 하루 1종목", baseline_stats),
        ("사후 최적 상한(참고)", oracle_stats),
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
            f"- 홀드아웃 pair 정확도: {_fmt_pct(pair_accuracy, signed=False)}",
            f"- 기존 점수와 같은 종목 선택 비율: {_fmt_pct(same_pick_rate, signed=False)}",
            f"- 결과: **{'PASS' if passed else 'FAIL'}**",
            "",
            "PASS 조건: 기존 점수보다 초과수익이 높고, 낙폭이 2%p 이상 악화되지 않으며, "
            "시장 우위 비율이 5%p 이상 악화되지 않고, pair 정확도가 50%를 넘을 것.",
            "",
            "## 방향성이 큰 변수",
            "",
        ]
    )
    for _, row in _coefficient_table(model).head(10).iterrows():
        direction = "높을수록 선호" if float(row["coefficient"]) > 0 else "낮을수록 선호"
        lines.append(f"- {row['feature']}: {float(row['coefficient']):+.3f} ({direction})")

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "passed": passed,
        "train_rows": len(train),
        "train_pairs": len(pair_x),
        "test_rows": len(test),
        "pair_accuracy": pair_accuracy,
        "same_pick_rate": same_pick_rate,
        "ranker": rank_stats,
        "baseline": baseline_stats,
        "excess_lift": excess_lift,
        "drawdown_delta": drawdown_delta,
        "hit_rate_delta": hit_rate_delta,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pairwise ranking experiment for the foreign-flow bot")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_pairwise_ranking(
        args.root / "reports" / "screen_backtest.csv",
        args.root / "data" / "history.csv",
        args.root / "reports" / "ml_pairwise_ranking.csv",
        args.root / "reports" / "ml_pairwise_ranking.md",
        test_fraction=args.test_fraction,
    )
    print(
        f"Pairwise ranking: {'PASS' if result['passed'] else 'FAIL'} | "
        f"lift={result['excess_lift'] * 100:+.2f}%p | "
        f"pair_acc={result['pair_accuracy'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
