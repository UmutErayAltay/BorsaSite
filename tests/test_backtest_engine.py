"""End-to-end backtest engine test: seeds real historical prices, injects a
model with a fixed, known prob_up (no real XGBoost training needed — the
engine's job is the day-by-day decision/cost/equity loop, not the model), and
checks the resulting trade lifecycle and equity curve against hand computation."""
from __future__ import annotations

from dataclasses import replace
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


class _FixedReturnModel:
    """`pipeline/predict_model.py`'deki `magnitude_model` yerine geçen sabit
    getiri (oran) tahmin eden stand-in: her satır için aynı beklenen getiriyi
    döndürür."""

    def __init__(self, expected_return: float):
        self.expected_return = expected_return

    def predict(self, X):
        return np.full(len(X), self.expected_return)


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


def _dump_model_with_magnitude(tmp_path: Path, prob: float, expected_return: float) -> Path:
    """Faz 6/7 sonrası bundle: `magnitude_model` anahtarı da içerir."""
    model_path = tmp_path / "model_magnitude.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": _FixedProbModel(prob),
            "features": FEATURE_COLUMNS,
            "magnitude_model": _FixedReturnModel(expected_return),
        },
        model_path,
    )
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


# --- Beklenen edge filtresi (canlı `trading/portfolio.py::buy` ile aynı davranış) ---


def test_magnitude_model_feeds_expected_return_into_buy_decisions(committed_conn, tmp_path):
    """Bundle `magnitude_model` içeriyorsa motor `expected_return`'u hesaplayıp
    `buy()`'a geçirir. Aynı `prob_up` ile yalnızca beklenen getiri değiştiğinde
    karar değişiyorsa, değerin modelden geldiği ve doğru hesaplandığı kanıtlanır
    (pozisyon 5000 TL, round-trip ücret 2 x 2.63 = 5.26 TL, %1 pay 50 TL →
    gereken ≈ 55.26 TL; %2 → 100 TL geçer, %0.5 → 25 TL geçmez)."""
    _seed_bist_symbol(committed_conn, "KCHIS.IS")
    committed_conn.commit()
    edge_cfg = replace(CFG, min_expected_edge_pct=0.01)

    good = run_backtest(
        trading_cfg=edge_cfg, cost_cfg=BacktestCostConfig(),
        model_path=_dump_model_with_magnitude(tmp_path / "good", prob=0.9, expected_return=0.02),
    )
    bad = run_backtest(
        trading_cfg=edge_cfg, cost_cfg=BacktestCostConfig(),
        model_path=_dump_model_with_magnitude(tmp_path / "bad", prob=0.9, expected_return=0.005),
    )

    good_buys = [d for d in good.decisions if d["action"] == "al"]
    assert good_buys, "%2 beklenen getiri, %1 pay + round-trip ücreti aşmalıydı"
    assert good.trades
    assert not any(d["action"] == "al" for d in bad.decisions)
    assert all(
        "round-trip ücret" in d["reason"]
        for d in bad.decisions
        if d["action"] == "red"
    )


def test_low_expected_return_blocks_candidate_when_edge_enabled(committed_conn, tmp_path):
    """`min_expected_edge_pct` açıkken `magnitude_model`'in ürettiği düşük
    beklenen getiri adayı backtest'te ALINMAMALI — karar 'red' + edge sebebi
    olarak loglanmalı (canlı motorla aynı)."""
    _seed_bist_symbol(committed_conn, "TUPRS.IS")
    committed_conn.commit()
    # %0.5 beklenen getiri, %1 güvenlik payını + round-trip ücretini karşılamaz
    model_path = _dump_model_with_magnitude(tmp_path, prob=0.9, expected_return=0.005)
    edge_cfg = replace(CFG, min_expected_edge_pct=0.01)

    result = run_backtest(trading_cfg=edge_cfg, cost_cfg=BacktestCostConfig(), model_path=model_path)

    rejected = [d for d in result.decisions if d["action"] == "red"]
    assert rejected, "düşük beklenen getiri reddedilmeliydi"
    assert all("beklenen kâr" in d["reason"] for d in rejected)
    assert not any(d["action"] == "al" for d in result.decisions)
    assert not result.trades
    assert all(value == edge_cfg.starting_balance for _, value in result.equity_curve)
