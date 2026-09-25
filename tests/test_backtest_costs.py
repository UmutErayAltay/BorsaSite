import pytest

from backtest.costs import BacktestCostConfig, execution_price


def test_execution_price_buy_is_above_mid():
    cfg = BacktestCostConfig(slippage_bps=10, spread_bps=20)
    price = execution_price(100.0, "buy", cfg)
    assert price > 100.0
    assert price == pytest.approx(100.0 * (1 + (10 + 10) / 10000))


def test_execution_price_sell_is_below_mid():
    cfg = BacktestCostConfig(slippage_bps=10, spread_bps=20)
    price = execution_price(100.0, "sell", cfg)
    assert price < 100.0
    assert price == pytest.approx(100.0 * (1 - (10 + 10) / 10000))


def test_execution_price_zero_cost_is_mid():
    cfg = BacktestCostConfig()
    assert execution_price(100.0, "buy", cfg) == 100.0
    assert execution_price(100.0, "sell", cfg) == 100.0


def test_execution_price_rejects_invalid_side():
    with pytest.raises(ValueError):
        execution_price(100.0, "hold", BacktestCostConfig())
