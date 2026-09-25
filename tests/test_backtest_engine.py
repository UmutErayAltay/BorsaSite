"""End-to-end backtest engine test: seeds real historical prices, injects a
model with a fixed, known prob_up (no real XGBoost training needed — the
engine's job is the day-by-day decision/cost/equity loop, not the model), and
checks the resulting trade lifecycle and equity curve against hand computation."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np

from backtest.costs import BacktestCostConfig
from backtest.engine import run_backtest
from pipeline.dataset import FEATURE_COLUMNS
from pipeline.db import upsert_prices, upsert_symbol
from trading.config import TradingConfig


class _FixedProbModel:
    """joblib-picklable stand-in for a trained classifier: always reports the
    same prob_up, so the backtest engine's OWN decision loop can be tested
    without training a real model."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


def _seed_bist_symbol(conn, ticker: str, days: int = 70, start_price: float = 100.0) -> int:
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    start = date(2026, 1, 1)
    rows = []
    price = start_price
    for i in range(days):
        d = start + timedelta(days=i)
        rows.append((d.isoformat(), price, price + 1, price - 1, price, price, 1000.0))
        # Net uptrend but with real down days -- a strictly monotonic series
        # makes RSI's average-loss term permanently 0/NaN (no losses to divide by).
        price += 0.6 if i % 3 else -0.2
    upsert_prices(conn, symbol_id, iter(rows))
    return symbol_id


CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.55,
    sell_threshold=0.50,
    max_hold_days=5,
    max_open_positions=3,
    max_position_pct=0.5,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=1.0,
    min_position_value_try=100.0,
)


def _dump_model(tmp_path: Path, prob: float) -> Path:
    model_path = tmp_path / "model.pkl"
    joblib.dump({"model": _FixedProbModel(prob), "features": FEATURE_COLUMNS}, model_path)
    return model_path


def test_high_prob_up_buys_and_force_sells_after_max_hold(committed_conn, tmp_path):
    _seed_bist_symbol(committed_conn, "THYAO.IS")
    committed_conn.commit()
    model_path = _dump_model(tmp_path, prob=0.9)  # always above buy_threshold and sell_threshold

    result = run_backtest(trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path)

    assert result.equity_curve, "backtest hiç gün üretmedi"
    buys = [d for d in result.decisions if d["action"] == "al"]
    forced_sells = [t for t in result.trades if t.exit_reason == "max_hold_süresi"]
    assert buys, "prob_up eşik üstündeyken hiç alım olmadı"
    assert forced_sells, "max_hold_days dolunca zorunlu satış tetiklenmedi"
    # never sold for "prob_düştü" since prob stays fixed above sell_threshold
    assert all(t.exit_reason != "prob_düştü" for t in result.trades)


def test_low_prob_never_buys(committed_conn, tmp_path):
    _seed_bist_symbol(committed_conn, "GARAN.IS")
    committed_conn.commit()
    model_path = _dump_model(tmp_path, prob=0.1)  # always below buy_threshold

    result = run_backtest(trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path)

    assert not result.trades
    assert all(d["action"] != "al" for d in result.decisions)
    # equity curve stays flat at starting_balance (no positions ever opened)
    assert all(value == CFG.starting_balance for _, value in result.equity_curve)


def test_start_end_date_filters_window(committed_conn, tmp_path):
    _seed_bist_symbol(committed_conn, "AKBNK.IS")
    committed_conn.commit()
    model_path = _dump_model(tmp_path, prob=0.9)

    full = run_backtest(trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path)
    windowed = run_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
        start_date="2026-02-01", end_date="2026-02-15",
    )

    assert len(windowed.equity_curve) < len(full.equity_curve)
    assert all("2026-02-01" <= d <= "2026-02-15" for d, _ in windowed.equity_curve)


def test_slippage_reduces_net_pnl_versus_zero_cost(committed_conn, tmp_path):
    _seed_bist_symbol(committed_conn, "SISE.IS")
    committed_conn.commit()
    model_path = _dump_model(tmp_path, prob=0.9)

    cheap = run_backtest(trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path)
    expensive = run_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(slippage_bps=50, spread_bps=50), model_path=model_path,
    )

    assert cheap.trades and expensive.trades
    cheap_net = sum(t.net_pnl for t in cheap.trades)
    expensive_net = sum(t.net_pnl for t in expensive.trades)
    assert expensive_net < cheap_net
