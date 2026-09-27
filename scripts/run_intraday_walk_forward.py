#!/usr/bin/env python3
"""Intraday v2 walk-forward backtest çalıştırıcı.

    python scripts/run_intraday_walk_forward.py \
        --initial-train-days 250 --val-days 60 --test-days 60 \
        --out data/reports/intraday_walk_forward.json

Metrikler `backtest/metrics.py`'daki mevcut fonksiyonlardan (`summarize`)
alınır — günlük backtest ile aynı hesap, aynı anahtar adları. Sonuç JSON'a
yazılır; DB'ye yazılmaz (bu bir araştırma koşusudur, `backtest_runs`
tablosunun günlük backtest'lerin kaydıdır).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402

from backtest.costs import BacktestCostConfig  # noqa: E402
from backtest.intraday_walk_forward import run_intraday_walk_forward  # noqa: E402
from backtest.metrics import summarize  # noqa: E402
from trading.config import load_trading_config  # noqa: E402

load_project_env()

DEFAULT_OUT = ROOT / "data" / "reports" / "intraday_walk_forward.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-train-days", type=int, default=250)
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument(
        "--entry-quantile",
        type=float,
        default=None,
        help="config/intraday_model.yaml::strategy.entry_quantile yerine kullanılacak kantil",
    )
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON çıktı yolu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    trading_cfg = load_trading_config()
    cost_cfg = BacktestCostConfig(
        slippage_bps=args.slippage_bps, spread_bps=args.spread_bps
    )

    result, folds = run_intraday_walk_forward(
        trading_cfg=trading_cfg,
        cost_cfg=cost_cfg,
        initial_train_days=args.initial_train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        entry_quantile=args.entry_quantile,
    )

    metrics = summarize(result.equity_curve, result.trades, trading_cfg.starting_balance)
    payload = {
        "params": {
            "initial_train_days": args.initial_train_days,
            "val_days": args.val_days,
            "test_days": args.test_days,
            "entry_quantile": args.entry_quantile,
            "slippage_bps": args.slippage_bps,
            "spread_bps": args.spread_bps,
        },
        "start_date": result.start_date,
        "end_date": result.end_date,
        "metrics": metrics,
        "folds": folds,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    if not folds:
        print(
            f"Yeterli veri yok: en az {args.initial_train_days + args.val_days + args.test_days} "
            f"farklı gün gerekiyor."
        )
        return

    print(f"\n{len(folds)} fold, {metrics['num_trades']} kapanmış işlem")
    for f in folds:
        ic = "yok" if f["test_ic"] is None else f"{f['test_ic']:+.4f}"
        print(
            f"  fold {f['fold']:>2}  test {f['test_start']}..{f['test_end']}  "
            f"eşik={f['entry_threshold']:+.6f}  test_ic={ic}"
        )
    print(
        f"Equity: {metrics['starting_balance']:.2f} -> "
        f"{metrics['ending_balance']:.2f} TL "
        f"(toplam {metrics['total_return_pct'] or 0.0:+.2f}%, "
        f"max DD {metrics['max_drawdown_pct']['max_drawdown_pct']:.2f}%)"
    )
    print(f"JSON: {out_path}")


if __name__ == "__main__":
    main()
