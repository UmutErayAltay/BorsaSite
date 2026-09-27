from dataclasses import replace
from datetime import date

from pipeline.db import upsert_prices, upsert_symbol
from trading.config import TradingConfig
from trading.portfolio import (
    buy,
    ensure_portfolio,
    get_open_position,
    get_state,
    positions_market_value,
    record_snapshot,
    sell,
)

CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.62,
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


def _symbol(conn) -> int:
    return upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")


def test_buy_deducts_balance_and_fee(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)

    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    assert ok is True
    state = get_state(conn)
    # pozisyon değeri: 10000 * %50 = 5000; komisyon: 5000*0.05%=2.5, BSMV: 0.125→0.12 (2 ondalık), toplam 2.62
    assert state.balance == 10000.0 - 5000.0 - 2.62
    assert len(state.open_positions) == 1
    assert state.open_positions[0].symbol_id == symbol_id


def test_buy_rejects_when_balance_insufficient(conn):
    tiny_cfg = replace(CFG, max_position_pct=1.0, min_position_value_try=1.0, min_commission_try=5.0)
    ensure_portfolio(conn, 10.0)
    symbol_id = _symbol(conn)

    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=tiny_cfg)

    assert ok is False
    assert "yetersiz" in reason.lower()


def test_buy_rejects_when_max_open_positions_reached(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    sym1 = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    sym2 = upsert_symbol(conn, "AKBNK.IS", "BIST", "TRY")
    sym3 = upsert_symbol(conn, "GARAN.IS", "BIST", "TRY")

    assert buy(conn, sym1, 10.0, 0.7, date(2026, 9, 10), CFG)[0] is True
    assert buy(conn, sym2, 10.0, 0.7, date(2026, 9, 10), CFG)[0] is True
    ok, reason = buy(conn, sym3, 10.0, 0.7, date(2026, 9, 10), CFG)

    assert ok is False
    assert "limit" in reason.lower()


def test_sell_computes_net_pnl_after_both_fees(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)
    buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    trade = sell(conn, symbol_id, price=110.0, exit_reason="prob_düştü", decision_date=date(2026, 9, 11), cfg=CFG)

    assert trade is not None
    assert trade.gross_pnl == 5000.0 * 0.10  # %10 fiyat artışı, 50 adet * 10 TL
    assert trade.net_pnl < trade.gross_pnl  # iki yönlü komisyon düşülmüş olmalı
    assert get_open_position(conn, symbol_id) is None


def test_buy_allowed_after_balance_grows_past_original_starting_balance(conn):
    """Regresyon: bkz. trading/portfolio.py::buy dokstring'i ve
    backtest/portfolio.py'deki aynı testin dokstring'i — canlı motorda da
    aynı bug vardı."""
    ensure_portfolio(conn, CFG.starting_balance)
    conn.execute("UPDATE portfolio SET balance = ? WHERE id = 1", (18500.0,))
    symbol_id = _symbol(conn)

    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    assert ok is True, reason


def test_sell_returns_none_when_no_open_position(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)

    assert sell(conn, symbol_id, 100.0, "prob_düştü", date(2026, 9, 10), CFG) is None


def test_positions_market_value_uses_latest_close(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)
    buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)
    upsert_prices(conn, symbol_id, iter([("2026-09-11", 120.0, 120.0, 120.0, 120.0, 120.0, 1000)]))

    state = get_state(conn)
    value = positions_market_value(conn, state.open_positions)

    assert value == state.open_positions[0].quantity * 120.0


def test_positions_market_value_falls_back_to_entry_price_without_prices(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)
    buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    state = get_state(conn)
    value = positions_market_value(conn, state.open_positions)

    assert value == state.open_positions[0].quantity * 100.0


def test_record_snapshot_stores_totals(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)
    buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    record_snapshot(conn, date(2026, 9, 10))

    row = conn.execute(
        "SELECT balance, positions_value, total_value FROM portfolio_snapshots WHERE snapshot_date = ?",
        ("2026-09-10",),
    ).fetchone()
    state = get_state(conn)
    assert float(row["balance"]) == state.balance
    assert float(row["total_value"]) == round(state.balance + float(row["positions_value"]), 2)


def test_record_snapshot_overwrites_same_day(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    record_snapshot(conn, date(2026, 9, 10))
    record_snapshot(conn, date(2026, 9, 10))

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM portfolio_snapshots WHERE snapshot_date = ?",
        ("2026-09-10",),
    ).fetchone()
    assert rows["n"] == 1


# --- Faz 6/7: beklenen edge filtresi ----------------------------------------
# CFG'de pozisyon = 10000 * %50 = 5000 TL, komisyon = 2.5 + BSMV ≈ 2.62 TL,
# yani round-trip ücret ≈ 5.24 TL. min_expected_edge_pct=%1 → 50 TL güvenlik
# payı, gereken toplam ≈ 55.24 TL.


def test_buy_without_expected_return_behaves_exactly_as_before(conn):
    """Eski çağıran deseni: `expected_return` HİÇ verilmiyor. Edge kontrolü
    varsayılan 0.0 yani kapalı olduğu için alım yine de başarılı olmalı ve
    bakiye tam olarak eskisi gibi düşmeli (geriye uyumluluk)."""
    assert CFG.min_expected_edge_pct == 0.0
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)

    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    assert ok is True, reason
    state = get_state(conn)
    assert state.balance == 10000.0 - 5000.0 - 2.62
    assert len(state.open_positions) == 1


def test_buy_rejected_when_expected_return_below_cost_plus_margin(conn):
    """min_expected_edge_pct açıkken düşük beklenen getirili alım REDDEDİLİR
    ve sebep edge'den bahseder."""
    edge_cfg = replace(CFG, min_expected_edge_pct=0.01)
    ensure_portfolio(conn, edge_cfg.starting_balance)
    symbol_id = _symbol(conn)

    # %0.5 → 25 TL beklenen kâr, round-trip ~5.24 + %1 pay 50 TL = ~55.24 TL
    ok, reason = buy(
        conn, symbol_id, price=100.0, prob_up=0.7,
        decision_date=date(2026, 9, 10), cfg=edge_cfg, expected_return=0.005,
    )

    assert ok is False
    assert "beklenen kâr" in reason
    assert "round-trip ücret" in reason
    # reddedilen alım hiçbir iz bırakmamalı
    assert get_state(conn).balance == edge_cfg.starting_balance
    assert get_open_position(conn, symbol_id) is None


def test_buy_succeeds_when_expected_return_clears_cost_plus_margin(conn):
    """Aynı cfg altında yeterli beklenen getiri (%), alım BAŞARILI olmalı."""
    edge_cfg = replace(CFG, min_expected_edge_pct=0.01)
    ensure_portfolio(conn, edge_cfg.starting_balance)
    symbol_id = _symbol(conn)

    # %2 → 100 TL beklenen kâr > 55.24 TL gereken
    ok, reason = buy(
        conn, symbol_id, price=100.0, prob_up=0.7,
        decision_date=date(2026, 9, 10), cfg=edge_cfg, expected_return=0.02,
    )

    assert ok is True, reason
    assert get_open_position(conn, symbol_id) is not None
    assert get_state(conn).balance == 10000.0 - 5000.0 - 2.62


def test_buy_rejected_when_expected_return_omitted_while_edge_enabled(conn):
    """Edge kontrolü açıkken beklenen getiri 0 (bilinmiyor) → sessizce
    geçmemeli, reddedilmeli: motor NULL expected_return'ı 0.0'a düşürür."""
    edge_cfg = replace(CFG, min_expected_edge_pct=0.01)
    ensure_portfolio(conn, edge_cfg.starting_balance)
    symbol_id = _symbol(conn)

    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=edge_cfg)

    assert ok is False
    assert "beklenen kâr" in reason
