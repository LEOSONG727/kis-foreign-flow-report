from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from .backtest import run_backtest, run_gate_sweep, run_screen_backtest
from .config import load_settings
from .pipeline import deep_backfill, rerank_stored_data, run_pipeline, smoke_test
from .earnings_quality import run_earnings_quality
from .jev_events import run_event_replay
from .jev_replay import run_jev_replay
from .research import build_research_panel, run_research


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KIS foreign accumulation daily report")
    parser.add_argument(
        "command",
        choices=[
            "run",
            "rerank",
            "smoke",
            "backtest",
            "screen-backtest",
            "gate-sweep",
            "research",
            "deep-backfill",
            "jev-replay",
            "event-replay",
            "earnings-quality",
        ],
        nargs="?",
        default="run",
    )
    parser.add_argument("--as-of", help="조회 기준일 YYYY-MM-DD")
    parser.add_argument("--universe-limit", type=int, help="개발용 종목 수 제한")
    parser.add_argument("--skip-backfill", action="store_true", help="장기 이력 초기화를 생략")
    parser.add_argument("--full-history-bootstrap", action="store_true", help="전체 유니버스 장기 이력 초기화")
    parser.add_argument("--no-notify", action="store_true", help="텔레그램·이메일 알림 생략")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="screen-backtest 신호일 간격. 기본 1(매일 전수)",
    )
    parser.add_argument("--workers", type=int, default=None, help="research 병렬 프로세스 수")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="deep-backfill 목표 기간(달력일). 기본은 history_retention_days",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=150.0,
        help="deep-backfill 시간 예산(분). 넘으면 저장 후 종료하고 다음 실행에서 이어감",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=1500,
        help="jev-replay 이번 실행 최대 API 호출 수(재실행 시 이어서 호출)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    settings = load_settings(args.root)
    if args.command == "smoke":
        smoke_test(settings)
        return
    if args.command == "backtest":
        run_backtest(settings)
        return
    if args.command == "screen-backtest":
        run_screen_backtest(settings, step=args.step or 1)
        return
    if args.command == "gate-sweep":
        run_gate_sweep(settings)
        return
    if args.command == "research":
        # One panel feeds both the rule/challenger audit and the Jev replay.
        panel = build_research_panel(settings, workers=args.workers)
        run_research(settings, panel=panel)
        run_jev_replay(settings, max_calls=args.max_calls, panel=panel)
        run_event_replay(settings, max_calls=args.max_calls, panel=panel)
        run_earnings_quality(settings, panel=panel, max_calls=args.max_calls)
        return
    if args.command == "earnings-quality":
        run_earnings_quality(settings, max_calls=args.max_calls, workers=args.workers)
        return
    if args.command == "event-replay":
        run_event_replay(settings, max_calls=args.max_calls, workers=args.workers)
        return
    if args.command == "jev-replay":
        run_jev_replay(settings, max_calls=args.max_calls, workers=args.workers)
        return
    if args.command == "deep-backfill":
        deep_backfill(
            settings,
            days=args.days,
            max_minutes=args.max_minutes,
            universe_limit=args.universe_limit,
        )
        return
    if args.command == "rerank":
        rerank_stored_data(settings, notify=not args.no_notify)
        return
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    run_pipeline(
        settings,
        as_of=as_of,
        universe_limit=args.universe_limit,
        skip_backfill=args.skip_backfill,
        full_history_bootstrap=args.full_history_bootstrap,
        notify=not args.no_notify,
    )


if __name__ == "__main__":
    main()
