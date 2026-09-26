from dataclasses import replace
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

# Faz 6 risk yönetimi varyantları — CFG'ye eklenen alanlar (stop_loss_pct,
# take_profit_pct, cooldown_days_after_exit) varsayılan 0.0/0 yani KAPALI.
# CFG'nin kendisi değiştirilmiyor; risk alanları açık senaryolar burada
# `replace` ile türetiliyor (frozen dataclass, test_portfolio.py ile aynı desen).
CFG_RISK = replace(CFG, stop_loss_pct=0.10, take_profit_pct=0.15)
CFG_COOLDOWN = replace(CFG, stop_loss_pct=0.10, cooldown_days_after_exit=3)
CFG_NO_COOLDOWN = replace(CFG, stop_loss_pct=0.10, cooldown_days_after_exit=0)


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


# --- Faz 6: risk yönetimi (stop_loss / take_profit / cooldown) ----------------


def test_run_once_sells_with_stop_loss_reason_when_price_falls_below_entry(conn):
    """stop_loss_pct=%10, entry=100 → 90'ın altı stop tetikler. prob_up ve
    max_hold_days tetikte DEĞİL, yani yalnızca stop kurtarabilir."""
    ensure_portfolio(conn, CFG_RISK.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, CFG_RISK, decision_date=date(2026, 9, 10))
    position = get_open_position(conn, symbol_id)
    assert position is not None

    upsert_prices(conn, symbol_id, [("2026-09-11", 88.0, 88.0, 88.0, 88.0, 88.0, 1000.0)])
    # prob_up hâlâ satış eşiğinin üstünde, tek gün sonra (max_hold_days=3 dolmadı)
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.80, 1, "1.0")
    stats = run_once(conn, CFG_RISK, decision_date=date(2026, 9, 11))

    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None
    row = conn.execute(
        "SELECT exit_reason, exit_price FROM trades WHERE symbol_id = ?", (symbol_id,)
    ).fetchone()
    assert row["exit_reason"] == "stop_loss"
    assert float(row["exit_price"]) == 88.0


def test_run_once_sells_with_take_profit_reason_when_price_rises_above_entry(conn):
    """take_profit_pct=%15, entry=100 → 115'in üstü kâr satışı tetikler."""
    ensure_portfolio(conn, CFG_RISK.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, CFG_RISK, decision_date=date(2026, 9, 10))
    assert get_open_position(conn, symbol_id) is not None

    upsert_prices(conn, symbol_id, [("2026-09-11", 120.0, 120.0, 120.0, 120.0, 120.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.80, 1, "1.0")
    stats = run_once(conn, CFG_RISK, decision_date=date(2026, 9, 11))

    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None
    row = conn.execute(
        "SELECT exit_reason, exit_price FROM trades WHERE symbol_id = ?", (symbol_id,)
    ).fetchone()
    assert row["exit_reason"] == "take_profit"
    assert float(row["exit_price"]) == 120.0


def test_run_once_does_not_trigger_stop_loss_when_disabled(conn):
    """stop_loss_pct=0 (kapalı): fiyat -%15'e düşse bile stop tetiklenmez,
    pozisyon yalnızca prob_up<sell_threshold ile kapanır (geriye uyumluluk)."""
    assert CFG.stop_loss_pct == 0.0
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, CFG, decision_date=date(2026, 9, 10))
    assert get_open_position(conn, symbol_id) is not None

    upsert_prices(conn, symbol_id, [("2026-09-11", 85.0, 85.0, 85.0, 85.0, 85.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.30, 0, "1.0")
    stats = run_once(conn, CFG, decision_date=date(2026, 9, 11))

    assert stats["sold"] == 1
    row = conn.execute(
        "SELECT exit_reason FROM trades WHERE symbol_id = ?", (symbol_id,)
    ).fetchone()
    # satış olur ama sebebi stop_loss DEĞİL — eşik mantığına düşmüş
    assert row["exit_reason"] == "prob_düştü"


def test_run_once_rebuys_after_cooldown_expires_but_rejects_before(conn):
    """cooldown_days_after_exit=3: stop-loss ile satılan sembol aynı gün ve
    cooldown dolmadan yeniden ALINAMAZ; 3 gün sonra tekrar alınır."""
    ensure_portfolio(conn, CFG_COOLDOWN.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, CFG_COOLDOWN, decision_date=date(2026, 9, 10))
    assert get_open_position(conn, symbol_id) is not None

    # 1) stop-loss ile sat (kapanış 10 Eylül)
    upsert_prices(conn, symbol_id, [("2026-09-11", 85.0, 85.0, 85.0, 85.0, 85.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.30, 0, "1.0")
    stats = run_once(conn, CFG_COOLDOWN, decision_date=date(2026, 9, 11))
    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None

    # 2) satıştan 2 gün sonra hâlâ aday (prob_up yüksek) ama cooldown'da
    upsert_prices(conn, symbol_id, [("2026-09-13", 105.0, 105.0, 105.0, 105.0, 105.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-13", "2026-09-14", 0.90, 1, "1.0")
    stats = run_once(conn, CFG_COOLDOWN, decision_date=date(2026, 9, 13))

    assert stats["bought"] == 0
    assert stats["rejected"] >= 1
    assert get_open_position(conn, symbol_id) is None
    # redden başka bir sebep (pozisyon limiti, bakiye) değil, cooldown olmalı
    reason_row = conn.execute(
        "SELECT reason FROM trade_decisions WHERE symbol_id = ? AND action = 'red' "
        "AND decision_date = ?",
        (symbol_id, date(2026, 9, 13)),
    ).fetchone()
    assert "cooldown" in reason_row["reason"]

    # 3) cooldown 3 gün dolunca (11 Eylül + 3 = 14 Eylül) tekrar alınır
    upsert_prices(conn, symbol_id, [("2026-09-14", 105.0, 105.0, 105.0, 105.0, 105.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-14", "2026-09-15", 0.90, 1, "1.0")
    stats = run_once(conn, CFG_COOLDOWN, decision_date=date(2026, 9, 14))

    assert stats["bought"] == 1
    assert get_open_position(conn, symbol_id) is not None


def test_run_once_rebuys_immediately_when_cooldown_disabled(conn):
    """cooldown_days_after_exit=0 (kapalı): stop-loss sonrası bir sonraki
    gün sembol yeniden adaysa ALINABİLİR (geriye uyumluluk)."""
    assert CFG_NO_COOLDOWN.cooldown_days_after_exit == 0
    ensure_portfolio(conn, CFG_NO_COOLDOWN.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, CFG_NO_COOLDOWN, decision_date=date(2026, 9, 10))
    assert get_open_position(conn, symbol_id) is not None

    # stop-loss ile sat
    upsert_prices(conn, symbol_id, [("2026-09-11", 85.0, 85.0, 85.0, 85.0, 85.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.30, 0, "1.0")
    stats = run_once(conn, CFG_NO_COOLDOWN, decision_date=date(2026, 9, 11))
    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None

    # ertesi gün tekrar aday → cooldown yok, hemen alınır
    upsert_prices(conn, symbol_id, [("2026-09-12", 95.0, 95.0, 95.0, 95.0, 95.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-12", "2026-09-13", 0.90, 1, "1.0")
    stats = run_once(conn, CFG_NO_COOLDOWN, decision_date=date(2026, 9, 12))

    assert stats["bought"] == 1
    assert get_open_position(conn, symbol_id) is not None
