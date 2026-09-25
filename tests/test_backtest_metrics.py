import math

import pytest

from backtest.metrics import (
    average_fee_per_trade,
    average_holding_days,
    daily_returns,
    expectancy,
    max_drawdown_pct,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_fees,
    total_return_pct,
    turnover_pct,
    win_rate_pct,
)

CURVE = [("2026-01-01", 100.0), ("2026-01-02", 110.0), ("2026-01-03", 99.0), ("2026-01-04", 121.0)]

TRADE = dict(
    entry_price=100.0, exit_price=110.0, quantity=10.0,
    gross_pnl=100.0, fees_paid=5.0, net_pnl=95.0,
    opened_at="2026-01-01", closed_at="2026-01-05",
)
LOSING_TRADE = dict(
    entry_price=50.0, exit_price=40.0, quantity=5.0,
    gross_pnl=-50.0, fees_paid=2.0, net_pnl=-52.0,
    opened_at="2026-01-02", closed_at="2026-01-03",
)


def test_daily_returns_hand_computed():
    returns = daily_returns(CURVE)
    assert returns == pytest.approx([10.0, -10.0, 22.222222], rel=1e-4)


def test_total_return_pct():
    assert total_return_pct(CURVE) == pytest.approx(21.0)
    assert total_return_pct([]) is None
    assert total_return_pct([("d", 100.0)]) is None


def test_max_drawdown_pct_hand_computed():
    # peak 110 (01-02), trough 99 (01-03) -> (110-99)/110 = 10%; a LATER, higher
    # peak (121 on 01-04) must not overwrite the peak_date that produced this dd.
    result = max_drawdown_pct(CURVE)
    assert result["max_drawdown_pct"] == pytest.approx(10.0)
    assert result["peak_date"] == "2026-01-02"
    assert result["trough_date"] == "2026-01-03"


def test_max_drawdown_empty():
    assert max_drawdown_pct([]) == {"max_drawdown_pct": 0.0, "peak_date": None, "trough_date": None}


def test_sharpe_ratio_zero_variance_is_none():
    # Constant 1%/day compounding: 100 -> 101 -> 102.01 -> 103.0301 -> ...
    # Floating-point arithmetic makes pstdev of these returns a tiny nonzero
    # noise value, not an exact 0.0 -- this must still be treated as "no
    # variance" rather than blowing up into an absurd ratio.
    flat = [("2026-01-01", 100.0 * 1.01**i) for i in range(10)]
    assert sharpe_ratio(flat) is None


def test_sharpe_ratio_positive_for_uptrend_with_variance():
    curve = [("d1", 100.0), ("d2", 105.0), ("d3", 103.0), ("d4", 110.0)]
    result = sharpe_ratio(curve)
    assert result is not None
    assert result > 0


def test_sortino_ratio_none_when_no_negative_returns():
    always_up = [("d1", 100.0), ("d2", 101.0), ("d3", 103.0), ("d4", 106.0)]
    assert sortino_ratio(always_up) is None


def test_sortino_ratio_none_with_only_one_negative_day():
    # Standard downside deviation is RMS of min(0, r) over ALL observations,
    # not the negative subset's own stdev -- a single negative day still
    # carries real risk and must not be treated as zero-variance.
    curve = [("d1", 100.0), ("d2", 105.0), ("d3", 95.0), ("d4", 110.0)]
    result = sortino_ratio(curve)
    assert result is not None
    assert math.isfinite(result)


def test_sortino_ratio_defined_when_negative_returns_exist():
    curve = [("d1", 100.0), ("d2", 105.0), ("d3", 95.0), ("d4", 108.0), ("d5", 90.0)]
    result = sortino_ratio(curve)
    assert result is not None
    assert math.isfinite(result)


def test_profit_factor_hand_computed():
    assert profit_factor([TRADE, LOSING_TRADE]) == pytest.approx(100.0 / 50.0)
    assert profit_factor([TRADE]) is None  # no losses -> division by zero guarded


def test_win_rate_pct():
    assert win_rate_pct([TRADE, LOSING_TRADE]) == pytest.approx(50.0)
    assert win_rate_pct([]) is None


def test_expectancy():
    assert expectancy([TRADE, LOSING_TRADE]) == pytest.approx((95.0 - 52.0) / 2)


def test_average_holding_days():
    assert average_holding_days([TRADE]) == pytest.approx(4.0)  # 01-01 -> 01-05
    assert average_holding_days([]) is None


def test_turnover_pct():
    curve = [("d1", 1000.0), ("d2", 1000.0)]
    result = turnover_pct([TRADE], curve)
    volume = abs(100.0 * 10.0) + abs(110.0 * 10.0)
    assert result == pytest.approx(volume / 1000.0 * 100.0)
    assert turnover_pct([], curve) is None


def test_total_fees_and_average():
    assert total_fees([TRADE, LOSING_TRADE]) == pytest.approx(7.0)
    assert average_fee_per_trade([TRADE, LOSING_TRADE]) == pytest.approx(3.5)
    assert total_fees([]) == 0.0
    assert average_fee_per_trade([]) is None


def test_summarize_has_all_keys_and_no_exceptions_on_empty():
    result = summarize([], [], 10000.0)
    assert result["num_trades"] == 0
    assert result["ending_balance"] is None
    assert result["starting_balance"] == 10000.0

    full = summarize(CURVE, [TRADE, LOSING_TRADE], 100.0)
    assert full["num_trades"] == 2
    assert full["ending_balance"] == 121.0
