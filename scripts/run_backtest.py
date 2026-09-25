#!/usr/bin/env python3
"""Historical backtest çalıştırıcı — python scripts/run_backtest.py [--scenario base|conservative|stress|all]"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from pipeline.env import load_project_env  # noqa: E402
from pipeline.db import get_connection, init_schema, insert_backtest_run, insert_backtest_trades, insert_backtest_equity  # noqa: E402
from backtest.costs import BacktestCostConfig, SCENARIOS  # noqa: E402
from backtest.engine import run_backtest  # noqa: E402
from backtest.reports import write_all  # noqa: E402
from trading.config import load_trading_config  # noqa: E402

load_project_env()

BACKTEST_CONFIG_PATH = ROOT / "config" / "backtest.yaml"


def _config_hash(trading_cfg, cost_cfg: BacktestCostConfig, start_date, end_date) -> str:
    payload = json.dumps(
        {"trading": trading_cfg.__dict__, "cost": cost_cfg.__dict__, "start": start_date, "end": end_date},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_one(scenario: str, start_date: str | None, end_date: str | None, save_to_db: bool) -> dict:
    with open(BACKTEST_CONFIG_PATH, encoding="utf-8") as f:
        bt_cfg = yaml.safe_load(f)
    scenario_cfg = bt_cfg["scenarios"].get(scenario)
    if scenario_cfg is None:
        raise SystemExit(f"Bilinmeyen senaryo: {scenario} (config/backtest.yaml)")

    trading_cfg = load_trading_config()
    cost_cfg = BacktestCostConfig(**scenario_cfg)

    result = run_backtest(
        start_date=start_date,
        end_date=end_date,
        trading_cfg=trading_cfg,
        cost_cfg=cost_cfg,
        bist_only=bt_cfg.get("bist_only", True),
    )
    report = write_all(result, trading_cfg.starting_balance, scenario=scenario)

    if save_to_db:
        config_hash = _config_hash(trading_cfg, cost_cfg, start_date, end_date)
        with get_connection() as conn:
            init_schema(conn)
            run_id = insert_backtest_run(
                conn, scenario=scenario, model_version=None, config_hash=config_hash,
                start_date=start_date, end_date=end_date,
                starting_balance=trading_cfg.starting_balance,
                slippage_bps=cost_cfg.slippage_bps, spread_bps=cost_cfg.spread_bps,
                metrics=report["metrics"],
            )
            insert_backtest_trades(conn, run_id, report["trades"])
            insert_backtest_equity(conn, run_id, result.equity_curve)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical backtest çalıştırıcı")
    parser.add_argument("--scenario", default="base", choices=[*SCENARIOS.keys(), "all"])
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--no-db", action="store_true", help="backtest_runs/trades/equity tablolarına yazma")
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        report = run_one(scenario, args.start_date, args.end_date, save_to_db=not args.no_db)
        print(f"\n--- {scenario} ---")
        print(json.dumps(report["metrics"], indent=2, ensure_ascii=False, default=str))
    print(f"\nRaporlar: {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
