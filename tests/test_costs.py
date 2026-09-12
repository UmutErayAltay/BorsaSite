from trading.config import TradingConfig
from trading.costs import calculate_fee, check_position_size

CFG = TradingConfig(
    starting_balance=100000.0,
    buy_threshold=0.62,
    sell_threshold=0.50,
    max_hold_days=10,
    max_open_positions=8,
    max_position_pct=0.15,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=5.0,
    min_position_value_try=500.0,
)


def test_calculate_fee_normal_trade():
    result = calculate_fee(10000.0, CFG)
    # komisyon: 10000 * 0.05% = 5.0 TL; BSMV: 5.0 * %5 = 0.25 TL
    assert result.commission == 5.0
    assert result.bsmv == 0.25
    assert result.total_fee == 5.25


def test_calculate_fee_applies_minimum_commission():
    # 100 TL'lik işlemde komisyon 0.05 TL + BSMV ~0.0025 TL — asgari 5 TL'nin altında
    result = calculate_fee(100.0, CFG)
    assert result.total_fee == 5.0


def test_check_position_size_rejects_below_minimum():
    check = check_position_size(499.0, CFG)
    assert check.allowed is False
    assert "499" in check.reason


def test_check_position_size_allows_at_minimum():
    check = check_position_size(500.0, CFG)
    assert check.allowed is True
