#!/usr/bin/env python3
"""Walk-forward (Faz 4) backtest çalıştırıcı — python scripts/run_walk_forward.py [--scenario base|conservative|stress|all]"""

import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from backtest.costs import SCENARIOS  # noqa: E402
from backtest.metrics import summarize  # noqa: E402
from backtest.walk_forward import load_walk_forward_config, run_walk_forward  # noqa: E402
from trading.config import load_trading_config  # noqa: E402

load_project_env()


def run_one(scenario: str, dataset: pd.DataFrame | None = None) -> None:
    trading_cfg = load_trading_config()
    result = run_walk_forward(
        cfg=load_walk_forward_config(),
        trading_cfg=trading_cfg,
        cost_cfg=SCENARIOS[scenario],
        dataset=dataset,
    )

    print(f"\n--- {scenario} ---")
    if not result.windows:
        print(
            "Yeterli veri yok: en az (train_days + validation_days + oos_days) "
            "kadar farklı feature_date gerekiyor (bkz. config/backtest.yaml::walk_forward)."
        )
        return

    print(f"{len(result.windows)} pencere, {len(result.trades)} kapanmış işlem")
    for w in result.windows:
        auc_str = f"{w['oos_auc']:.4f}" if w["oos_auc"] is not None else "n/a"
        ic_str = f"{w['oos_ic']:+.4f}" if w["oos_ic"] is not None else "n/a"
        print(
            f"  OOS {w['oos_start']}..{w['oos_end']}: threshold={w['threshold']:.3f} "
            f"train={w['train_rows']} valid={w['validation_rows']} oos={w['oos_rows']} "
            f"AUC={auc_str} IC={ic_str}"
        )

    # Havuzlanmış (tüm pencereler birleşik) AUC — asıl karar verici sayı,
    # tek pencerenin şansına bağlı olmasın diye. Pencere başı IC'lerin
    # ortalaması ayrıca basılır (bkz. plan: "IC 3 pencereden 2'sinde pozitif").
    preds = pd.DataFrame(result.oos_predictions)
    pooled_auc = None
    if not preds.empty and preds["target_up"].nunique() > 1:
        pooled_auc = roc_auc_score(preds["target_up"].astype(int), preds["prob_up"])
    window_ics = [w["oos_ic"] for w in result.windows if w["oos_ic"] is not None]
    positive_ic_windows = sum(1 for ic in window_ics if ic > 0)
    print(
        f"Havuzlanmış OOS AUC: {pooled_auc:.4f}" if pooled_auc is not None
        else "Havuzlanmış OOS AUC: n/a (tek sınıf)"
    )
    if window_ics:
        print(
            f"Ortalama IC: {statistics.fmean(window_ics):+.4f} "
            f"({positive_ic_windows}/{len(window_ics)} pencere pozitif)"
        )

    if result.equity_curve:
        m = summarize(result.equity_curve, result.trades, trading_cfg.starting_balance)
        dd = m["max_drawdown_pct"]["max_drawdown_pct"] if m["max_drawdown_pct"] else 0.0
        sharpe = f"{m['sharpe_ratio']:.3f}" if m["sharpe_ratio"] is not None else "n/a"
        print(
            f"Equity: {m['starting_balance']:.2f} TL -> {m['ending_balance']:.2f} TL "
            f"({m['total_return_pct']:+.2f}%) | Sharpe={sharpe} maxDD={dd:.2f}% "
            f"winRate={m['win_rate_pct'] or 0:.1f}% trades={m['num_trades']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="base", choices=[*SCENARIOS.keys(), "all"])
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    from pipeline.dataset import build_dataset  # noqa: E402 (env yüklendikten sonra)
    dataset = build_dataset(require_target=False)

    for scenario in scenarios:
        run_one(scenario, dataset=dataset)


if __name__ == "__main__":
    main()
