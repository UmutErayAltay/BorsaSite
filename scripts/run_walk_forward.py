#!/usr/bin/env python3
"""Walk-forward (Faz 4) backtest çalıştırıcı — python scripts/run_walk_forward.py [--scenario stress]"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from backtest.costs import SCENARIOS  # noqa: E402
from backtest.walk_forward import load_walk_forward_config, run_walk_forward  # noqa: E402
from trading.config import load_trading_config  # noqa: E402

load_project_env()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="base", choices=[*SCENARIOS.keys()])
    args = parser.parse_args()

    result = run_walk_forward(
        cfg=load_walk_forward_config(),
        trading_cfg=load_trading_config(),
        cost_cfg=SCENARIOS[args.scenario],
    )

    if not result.windows:
        print(
            "Yeterli veri yok: en az (train_days + validation_days + oos_days) "
            "kadar farklı feature_date gerekiyor (bkz. config/backtest.yaml::walk_forward)."
        )
        return

    print(f"{len(result.windows)} pencere, {len(result.trades)} kapanmış işlem")
    for w in result.windows:
        print(
            f"  OOS {w['oos_start']}..{w['oos_end']}: threshold={w['threshold']:.3f} "
            f"train={w['train_rows']} valid={w['validation_rows']} oos={w['oos_rows']}"
        )

    if result.equity_curve:
        start_equity = result.equity_curve[0][1]
        end_equity = result.equity_curve[-1][1]
        change_pct = (end_equity / start_equity - 1) * 100 if start_equity else 0.0
        print(f"Equity: {start_equity:.2f} TL -> {end_equity:.2f} TL ({change_pct:+.2f}%)")

    trades_summary = [
        {"symbol": t.symbol, "net_pnl": t.net_pnl, "exit_reason": t.exit_reason}
        for t in result.trades
    ]
    print(json.dumps(trades_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
