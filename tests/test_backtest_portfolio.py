from backtest.costs import BacktestCostConfig
from backtest.portfolio import BacktestPortfolio
from trading.config import TradingConfig

CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.55,
    sell_threshold=0.50,
    max_hold_days=10,
    max_open_positions=2,
    max_position_pct=0.5,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=1.0,
    min_position_value_try=100.0,
)
NO_COST = BacktestCostConfig()


def test_buy_deducts_balance_and_records_position():
    pf = BacktestPortfolio(starting_balance=10000.0)
    ok, reason = pf.buy("THYAO.IS", 100.0, 0.7, "2026-01-01", CFG, NO_COST)

    assert ok is True
    # position_value = 5000, fee = max(5000*0.0005*1.05, 1.0) = 2.625 -> 2.62/2.63 rounding
    assert pf.balance < 10000.0 - 5000.0
    assert "THYAO.IS" in pf.open_positions
    assert pf.open_positions["THYAO.IS"].quantity == 5000.0 / 100.0


def test_buy_rejects_when_max_positions_reached():
    pf = BacktestPortfolio(starting_balance=10000.0)
    cfg = CFG.__class__(**{**CFG.__dict__, "max_open_positions": 1})
    pf.buy("A", 100.0, 0.7, "2026-01-01", cfg, NO_COST)
    ok, reason = pf.buy("B", 100.0, 0.7, "2026-01-01", cfg, NO_COST)
    assert ok is False
    assert "limiti" in reason


def test_buy_rejects_duplicate_symbol():
    pf = BacktestPortfolio(starting_balance=10000.0)
    pf.buy("A", 100.0, 0.7, "2026-01-01", CFG, NO_COST)
    ok, reason = pf.buy("A", 100.0, 0.7, "2026-01-02", CFG, NO_COST)
    assert ok is False


def test_buy_rejects_insufficient_balance():
    pf = BacktestPortfolio(starting_balance=100.0)
    ok, reason = pf.buy("A", 100.0, 0.7, "2026-01-01", CFG, NO_COST)
    assert ok is False
    assert "yetersiz" in reason.lower() or "kucuk" in reason.lower() or "küçük" in reason.lower()


def test_sell_computes_net_pnl_and_returns_balance():
    pf = BacktestPortfolio(starting_balance=10000.0)
    pf.buy("A", 100.0, 0.7, "2026-01-01", CFG, NO_COST)
    balance_after_buy = pf.balance

    trade = pf.sell("A", 110.0, "prob_düştü", "2026-01-05", CFG, NO_COST)

    assert trade is not None
    assert trade.gross_pnl > 0
    assert "A" not in pf.open_positions
    assert pf.balance > balance_after_buy


def test_sell_returns_none_when_no_open_position():
    pf = BacktestPortfolio(starting_balance=10000.0)
    assert pf.sell("A", 100.0, "x", "2026-01-01", CFG, NO_COST) is None


def test_slippage_makes_buy_more_expensive_and_sell_cheaper():
    cost_cfg = BacktestCostConfig(slippage_bps=100)  # 1%
    pf_free = BacktestPortfolio(starting_balance=10000.0)
    pf_slip = BacktestPortfolio(starting_balance=10000.0)

    pf_free.buy("A", 100.0, 0.7, "2026-01-01", CFG, NO_COST)
    pf_slip.buy("A", 100.0, 0.7, "2026-01-01", CFG, cost_cfg)

    assert pf_slip.open_positions["A"].entry_price > pf_free.open_positions["A"].entry_price
    # same nominal position_value -> higher execution price -> fewer shares
    assert pf_slip.open_positions["A"].quantity < pf_free.open_positions["A"].quantity


def test_buy_allowed_after_balance_grows_past_original_starting_balance():
    """Regresyon: eski kod `%90 portföy riski` kontrolünü SABİT `starting_balance`'a
    bölüyordu — bakiye kâr edip başlangıcın biraz üzerine çıktığında oran kalıcı
    olarak %90'ı aşıyor ve portföy bir daha ASLA yeni pozisyon açamıyordu (gerçek
    5 yıllık BIST verisiyle walk-forward koşusunda tespit edildi — bkz.
    backtest/portfolio.py::buy dokstring'i). Artık mevcut toplam varlığa
    (nakit + açık pozisyon) göre hesaplanıyor."""
    pf = BacktestPortfolio(starting_balance=10000.0)
    pf.balance = 18500.0  # strateji kâr etti, bakiye başlangıcın çok üzerinde

    ok, reason = pf.buy("A", 100.0, 0.7, "2026-01-01", CFG, NO_COST)

    assert ok is True, reason


def test_positions_value_and_snapshot():
    pf = BacktestPortfolio(starting_balance=10000.0)
    pf.buy("A", 100.0, 0.7, "2026-01-01", CFG, NO_COST)
    pf.record_snapshot("2026-01-01", {"A": 120.0})

    assert pf.equity_curve[-1][0] == "2026-01-01"
    expected_total = pf.balance + 120.0 * pf.open_positions["A"].quantity
    assert pf.equity_curve[-1][1] == round(expected_total, 2)
