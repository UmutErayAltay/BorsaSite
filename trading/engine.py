"""Günlük al/sat/tut karar motoru — sadece BIST sembolleri, eşik-bazlı.
Her karar (alım, satım, tutma, red) `trade_decisions`'a insan-okunur bir
gerekçeyle loglanır — `kripto-trading-bot`'taki 'reddedilen işlemler
nedensel loglanmalı' disiplininin aynısı."""

from __future__ import annotations

from datetime import date

from trading import portfolio as pf
from trading.config import TradingConfig


def _latest_prediction(conn, symbol_id: int):
    return conn.execute(
        "SELECT prob_up FROM predictions WHERE symbol_id = ? ORDER BY feature_date DESC LIMIT 1",
        (symbol_id,),
    ).fetchone()


def _latest_close(conn, symbol_id: int) -> float | None:
    row = conn.execute(
        "SELECT close FROM prices_daily WHERE symbol_id = ? ORDER BY date DESC LIMIT 1",
        (symbol_id,),
    ).fetchone()
    return float(row["close"]) if row else None


def _log_decision(
    conn,
    decision_date: date,
    symbol_id: int | None,
    action: str,
    reason: str,
    prob_up: float | None,
) -> None:
    conn.execute(
        "INSERT INTO trade_decisions (decision_date, symbol_id, action, reason, prob_up) "
        "VALUES (?, ?, ?, ?, ?)",
        (decision_date.isoformat(), symbol_id, action, reason, prob_up),
    )


def run_once(conn, cfg: TradingConfig, decision_date: date | None = None) -> dict:
    decision_date = decision_date or date.today()
    pf.ensure_portfolio(conn, cfg.starting_balance)

    stats = {"bought": 0, "sold": 0, "held": 0, "rejected": 0}
    sold_symbol_ids = set()

    # 1) Açık pozisyonları değerlendir: sat ya da tut
    state = pf.get_state(conn)
    for position in list(state.open_positions):
        pred = _latest_prediction(conn, position.symbol_id)
        current_prob = float(pred["prob_up"]) if pred else None
        opened = date.fromisoformat(position.opened_at)
        held_days = (decision_date - opened).days

        if held_days >= cfg.max_hold_days:
            price = _latest_close(conn, position.symbol_id)
            if price is not None:
                trade = pf.sell(conn, position.symbol_id, price, "max_hold_süresi", decision_date, cfg)
                if trade:
                    stats["sold"] += 1
                    sold_symbol_ids.add(position.symbol_id)
                    _log_decision(
                        conn, decision_date, position.symbol_id, "sat",
                        "max_hold_süresi doldu", current_prob,
                    )
            continue

        if current_prob is not None and current_prob < cfg.sell_threshold:
            price = _latest_close(conn, position.symbol_id)
            if price is not None:
                trade = pf.sell(conn, position.symbol_id, price, "prob_düştü", decision_date, cfg)
                if trade:
                    stats["sold"] += 1
                    sold_symbol_ids.add(position.symbol_id)
                    _log_decision(
                        conn, decision_date, position.symbol_id, "sat",
                        f"prob_up {current_prob:.3f} < eşik {cfg.sell_threshold}", current_prob,
                    )
                continue

        stats["held"] += 1
        _log_decision(conn, decision_date, position.symbol_id, "tut", "eşiklerin içinde", current_prob)

    # 2) Yeni alım adayları — sadece BIST, eşik üstü, henüz açık pozisyonu olmayanlar
    open_symbol_ids = {p.symbol_id for p in pf.get_state(conn).open_positions}
    candidates = conn.execute(
        """
        SELECT p.symbol_id, p.prob_up
        FROM predictions p
        JOIN symbols s ON s.id = p.symbol_id
        WHERE s.market = 'BIST'
          AND p.prob_up > ?
          AND p.feature_date = (
              SELECT MAX(p2.feature_date) FROM predictions p2 WHERE p2.symbol_id = p.symbol_id
          )
        ORDER BY p.prob_up DESC
        """,
        (cfg.buy_threshold,),
    ).fetchall()

    for row in candidates:
        symbol_id = int(row["symbol_id"])
        prob_up = float(row["prob_up"])
        if symbol_id in open_symbol_ids or symbol_id in sold_symbol_ids:
            continue

        price = _latest_close(conn, symbol_id)
        if price is None:
            stats["rejected"] += 1
            _log_decision(conn, decision_date, symbol_id, "red", "güncel fiyat yok", prob_up)
            continue

        ok, reason = pf.buy(conn, symbol_id, price, prob_up, decision_date, cfg)
        if ok:
            stats["bought"] += 1
            open_symbol_ids.add(symbol_id)
            _log_decision(conn, decision_date, symbol_id, "al", reason, prob_up)
        else:
            stats["rejected"] += 1
            _log_decision(conn, decision_date, symbol_id, "red", reason, prob_up)

    return stats
