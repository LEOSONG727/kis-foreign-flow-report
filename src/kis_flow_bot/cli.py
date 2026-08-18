from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from .backtest import run_backtest, run_gate_sweep, run_screen_backtest
from .config import load_settings
from .pipeline import rerank_stored_data, run_pipeline, smoke_test


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KIS foreign accumulation daily report")
    parser.add_argument(
        "command",
        choices=["run", "rerank", "smoke", "backtest", "screen-backtest", "gate-sweep"],
        nargs="?",
        default="run",
    )
    parser.add_argument("--as-of", help="조회 기준일 YYYY-MM-DD")
    parser.add_argument("--universe-limit", type=int, help="개발용 종목 수 제한")
    parser.add_argument("--skip-backfill", action="store_true", help="장기 이력 초기화를 생략")
    parser.add_argument("--full-history-bootstrap", action="store_true", help="전체 유니버스 장기 이력 초기화")
    parser.add_argument("--no-notify", action="store_true", help="텔레그램·이메일 알림 생략")
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
        run_screen_backtest(settings)
        return
    if args.command == "gate-sweep":
        run_gate_sweep(settings)
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
