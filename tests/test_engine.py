from datetime import date, timedelta

from pipeline.db import upsert_prediction, upsert_prices, upsert_symbol
from trading.config import TradingConfig
from trading.engine import run_once
from trading.portfolio import ensure_portfolio, get_open_position, get_state

CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.62,
    sell_threshold=0.50,
    max_hold_days=3,
    max_open_positions=5,
    max_position_pct=0.3,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=1.0,
    min_position_value_try=100.0,
)


def _setup_symbol_with_price(conn, ticker: str, close: float, market: str = "BIST") -> int:
    symbol_id = upsert_symbol(conn, ticker, market, "TRY")
    upsert_prices(conn, symbol_id, [("2026-09-10", close, close, close, close, close, 1000.0)])
    return symbol_id


def test_run_once_buys_symbol_above_threshold(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 1
    assert get_open_position(conn, symbol_id) is not None


def test_run_once_ignores_symbol_below_threshold(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.40, 0, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 0
    assert get_open_position(conn, symbol_id) is None


def test_run_once_ignores_non_bist_symbol(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "AAPL", 100.0, market="US")
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.90, 1, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 0


def test_run_once_sells_when_prob_drops_below_sell_threshold(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, CFG, decision_date=date(2026, 9, 10))
    assert get_open_position(conn, symbol_id) is not None

    upsert_prices(conn, symbol_id, [("2026-09-11", 105.0, 105.0, 105.0, 105.0, 105.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.30, 0, "1.0")
    stats = run_once(conn, CFG, decision_date=date(2026, 9, 11))

    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None


def test_run_once_force_sells_after_max_hold_days(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-01", "2026-09-02", 0.75, 1, "1.0")
    run_once(conn, CFG, decision_date=date(2026, 9, 1))
    assert get_open_position(conn, symbol_id) is not None

    # prob_up hâlâ yüksek olsa bile max_hold_days (3) dolunca kapatılmalı
    late_date = date(2026, 9, 1) + timedelta(days=CFG.max_hold_days)
    upsert_prices(conn, symbol_id, [(late_date.isoformat(), 110.0, 110.0, 110.0, 110.0, 110.0, 1000.0)])
    upsert_prediction(conn, symbol_id, late_date.isoformat(), None, 0.80, 1, "1.0")
    stats = run_once(conn, CFG, decision_date=late_date)

    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None


def test_run_once_rejects_when_max_positions_reached(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    for i in range(5):
        symbol_id = _setup_symbol_with_price(conn, f"SYM{i}.IS", 100.0)
        upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    extra_id = _setup_symbol_with_price(conn, "EXTRA.IS", 100.0)
    upsert_prediction(conn, extra_id, "2026-09-10", "2026-09-11", 0.99, 1, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 5
    assert stats["rejected"] == 1
