from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES: tuple[str, ...] = (
    "discovery_score",
    "foreign_20_adtv",
    "price_return20",
    "rebound_from_low20",
    "ma20_distance",
    "cheap_accumulation",
    "dip_absorption",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("market",)
TARGET_COLUMN = "excess_forward_return"
BASELINE_SCORE_COLUMN = "discovery_score"
DEFAULT_HORIZON_SESSIONS = 20
DEFAULT_TEST_FRACTION = 0.30
MIN_TRAIN_ROWS = 60
MIN_TEST_ROWS = 20


@dataclass(frozen=True)
class SplitInfo:
    test_start: pd.Timestamp
    train_end: pd.Timestamp
    horizon_sessions: int


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _bool_to_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric).fillna(0.0).astype(float)


def prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "as_of",
        "symbol",
        "market",
        "forward_return",
        "excess_forward_return",
        "max_drawdown",
        *NUMERIC_FEATURES,
    ]
    _require_columns(frame, required)
    result = frame.copy()
    result["as_of"] = pd.to_datetime(result["as_of"], errors="coerce")
    result["symbol"] = result["symbol"].astype("string").str.zfill(6)
    result["market"] = result["market"].astype("string").fillna("UNKNOWN")

    for column in ("cheap_accumulation", "dip_absorption"):
        result[column] = _bool_to_float(result[column])

    for column in (
        "forward_return",
        "excess_forward_return",
        "max_drawdown",
        *[item for item in NUMERIC_FEATURES if item not in {"cheap_accumulation", "dip_absorption"}],
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["as_of", TARGET_COLUMN])
    result = result.drop_duplicates(["as_of", "symbol"], keep="last")
    return result.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def load_trading_calendar(history_path: Path) -> pd.DatetimeIndex:
    dates = pd.read_csv(history_path, usecols=["date"])
    calendar = pd.to_datetime(dates["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("Trading calendar is empty.")
    return pd.DatetimeIndex(calendar)


def purged_train_test_split(
    frame: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    if not 0.10 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.10 and 0.50")
    signal_dates = pd.DatetimeIndex(sorted(frame["as_of"].dropna().unique()))
    if len(signal_dates) < 10:
        raise ValueError("At least 10 distinct signal dates are required.")

    cut = int(np.floor(len(signal_dates) * (1.0 - test_fraction)))
    cut = min(max(cut, 1), len(signal_dates) - 1)
    test_start = pd.Timestamp(signal_dates[cut])

    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(trading_calendar).unique()))
    test_pos = int(calendar.searchsorted(test_start, side="left"))
    if test_pos >= len(calendar) or pd.Timestamp(calendar[test_pos]) != test_start:
        test_pos = int(calendar.searchsorted(test_start, side="right"))
    train_end_pos = test_pos - horizon_sessions - 1
    if train_end_pos < 0:
        raise ValueError("Not enough trading history for the requested purge horizon.")
    train_end = pd.Timestamp(calendar[train_end_pos])

    train = frame[frame["as_of"] <= train_end].copy()
    test = frame[frame["as_of"] >= test_start].copy()
    info = SplitInfo(test_start=test_start, train_end=train_end, horizon_sessions=horizon_sessions)
    return train, test, info


def build_model() -> Any:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise RuntimeError(
            "ML dependencies are not installed. Run: pip install -e '.[ml]'"
        ) from exc

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, list(NUMERIC_FEATURES)),
            ("cat", categorical_pipe, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    regressor = GradientBoostingRegressor(
        loss="huber",
        learning_rate=0.03,
        n_estimators=120,
        max_depth=2,
        min_samples_leaf=8,
        subsample=0.80,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)].copy()


def select_daily_top(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    _require_columns(frame, ["as_of", score_column])
    ranked = frame.sort_values(
        ["as_of", score_column, BASELINE_SCORE_COLUMN],
        ascending=[True, False, False],
        kind="mergesort",
    )
    return ranked.groupby("as_of", sort=True, as_index=False).head(1).reset_index(drop=True)


def summarize_selection(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "signals": 0.0,
            "dates": 0.0,
            "mean_return": float("nan"),
            "mean_excess": float("nan"),
            "positive_excess_rate": float("nan"),
            "mean_drawdown": float("nan"),
        }
    return {
        "signals": float(len(frame)),
        "dates": float(frame["as_of"].nunique()),
        "mean_return": float(pd.to_numeric(frame["forward_return"], errors="coerce").mean()),
        "mean_excess": float(pd.to_numeric(frame["excess_forward_return"], errors="coerce").mean()),
        "positive_excess_rate": float(
            (pd.to_numeric(frame["excess_forward_return"], errors="coerce") > 0).mean()
        ),
        "mean_drawdown": float(pd.to_numeric(frame["max_drawdown"], errors="coerce").mean()),
    }


def _fmt_pct(value: float, signed: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:{'+' if signed else ''}.2f}%"


def _feature_importance(model: Any) -> pd.DataFrame:
    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["regressor"]
        names = preprocessor.get_feature_names_out()
        values = regressor.feature_importances_
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_alpha_baseline(
    screen_backtest_path: Path,
    history_path: Path,
    output_csv: Path,
    output_report: Path,
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

    model = build_model()
    model.fit(feature_matrix(train), train[TARGET_COLUMN])

    scored = test.copy()
    scored["ml_alpha_pred"] = model.predict(feature_matrix(scored))
    scored = scored.sort_values(["as_of", "ml_alpha_pred"], ascending=[True, False])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False, encoding="utf-8-sig")

    ml_top = select_daily_top(scored, "ml_alpha_pred")
    baseline_top = select_daily_top(scored, BASELINE_SCORE_COLUMN)
    current_buy = (
        scored[scored["bucket"] == "오늘 신규매수 검토"].copy()
        if "bucket" in scored
        else scored.iloc[0:0].copy()
    )

    ml_stats = summarize_selection(ml_top)
    baseline_stats = summarize_selection(baseline_top)
    current_stats = summarize_selection(current_buy)

    excess_lift = ml_stats["mean_excess"] - baseline_stats["mean_excess"]
    drawdown_delta = ml_stats["mean_drawdown"] - baseline_stats["mean_drawdown"]
    hit_rate_delta = ml_stats["positive_excess_rate"] - baseline_stats["positive_excess_rate"]
    passed = bool(
        excess_lift > 0.0
        and drawdown_delta >= -0.02
        and hit_rate_delta >= -0.05
    )

    importance = _feature_importance(model)
    top_importance = importance.head(8)

    lines = [
        "# V3.1-A ML 알파 베이스라인",
        "",
        "기존 외인수급 숏리스트는 그대로 두고, 과거 신호 중 하루 1종목을 고르는 보조 ML 모델만 검증합니다.",
        "이 결과는 실전 추천이나 주문에 반영하지 않습니다.",
        "",
        "## 누수 방지",
        "",
        f"- 목표: 신호 다음 거래일 시가 진입 후 {horizon_sessions}거래일 시장초과수익",
        f"- 학습 마지막 신호일: {split.train_end.date()}",
        f"- 테스트 시작일: {split.test_start.date()}",
        f"- purge: 테스트 시작 전 {horizon_sessions} 거래세션의 학습 신호 제거",
        f"- 학습 {len(train)}건 / 테스트 {len(test)}건",
        "",
        "## 테스트 결과",
        "",
        "| 방식 | 신호 | 날짜 | 평균 20일 | 시장초과 | 시장 우위 | 평균 최대낙폭 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in (
        ("ML 하루 1종목", ml_stats),
        ("기존 discovery_score 하루 1종목", baseline_stats),
        ("기존 '오늘 신규매수 검토' 전체", current_stats),
    ):
        lines.append(
            f"| {label} | {int(stats['signals'])} | {int(stats['dates'])} | "
            f"{_fmt_pct(stats['mean_return'])} | {_fmt_pct(stats['mean_excess'])} | "
            f"{_fmt_pct(stats['positive_excess_rate'], signed=False)} | "
            f"{_fmt_pct(stats['mean_drawdown'])} |"
        )

    lines.extend(
        [
            "",
            "## 1차 게이트",
            "",
            f"- ML 초과수익 개선: {_fmt_pct(excess_lift)}p",
            f"- ML 낙폭 변화: {_fmt_pct(drawdown_delta)}p (양수면 개선)",
            f"- 시장 우위 비율 변화: {_fmt_pct(hit_rate_delta)}p",
            f"- 판정: **{'PASS' if passed else 'FAIL'}**",
            "",
            "PASS 조건은 기존 하루 1종목보다 시장초과수익이 높고, 평균 최대낙폭이 2%p 이상 악화되지 않으며, 시장 우위 비율이 5%p 이상 악화되지 않는 것입니다.",
            "PASS여도 바로 실전에 투입하지 않고 다음 단계에서 롤링 워크포워드와 라이브 shadow 검증을 거칩니다.",
            "",
            "## 모델이 많이 본 변수",
            "",
        ]
    )
    if top_importance.empty:
        lines.append("- 중요도 산출 불가")
    else:
        for _, row in top_importance.iterrows():
            lines.append(f"- {row['feature']}: {float(row['importance']):.3f}")

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "passed": passed,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_start": split.test_start,
        "train_end": split.train_end,
        "ml": ml_stats,
        "baseline": baseline_stats,
        "current_buy": current_stats,
        "excess_lift": excess_lift,
        "drawdown_delta": drawdown_delta,
        "hit_rate_delta": hit_rate_delta,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline ML alpha baseline for the foreign-flow bot")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root
    result = run_alpha_baseline(
        root / "reports" / "screen_backtest.csv",
        root / "data" / "history.csv",
        root / "reports" / "ml_alpha_baseline.csv",
        root / "reports" / "ml_alpha_baseline.md",
        test_fraction=args.test_fraction,
    )
    print(
        "ML alpha baseline: "
        f"{'PASS' if result['passed'] else 'FAIL'} | "
        f"lift={result['excess_lift'] * 100:+.2f}%p | "
        f"train={result['train_rows']} test={result['test_rows']}"
    )


if __name__ == "__main__":
    main()
