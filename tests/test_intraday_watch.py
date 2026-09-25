"""`react_to_negative_news` ağ çağrısı yapmaz (bkz. pipeline/intraday_watch.py
dokstring'i) — bu yüzden gerçek DB'ye karşı, FinBERT'i indirmeden/çalıştırmadan
test edilebilir: haberi ÖNCEDEN skorlanmış gibi doğrudan `news_sentiment`'e
yazıyoruz, `analyze_batch` hiç çağrılmıyor (pending sorgusu zaten skorlanmış
haberi atlar)."""
from datetime import date, timedelta

from pipeline.db import upsert_symbol
from pipeline.intraday_watch import react_to_negative_news
from trading.config import TradingConfig
from trading.portfolio import buy, ensure_portfolio, get_open_position

CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.55,
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


def _open_position(conn, opened_at: date) -> int:
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=opened_at, cfg=CFG)
    assert ok, reason
    return symbol_id


def _insert_scored_news(conn, symbol_id: int, score: float, published_at: str) -> None:
    row = conn.execute(
        "INSERT INTO news_raw (source, title, url, published_at, language, fetched_at) "
        "VALUES ('KAP', 'test bildirimi', ?, ?, 'tr', NOW()) RETURNING id",
        (f"https://kap.org.tr/test/{score}/{published_at}", published_at),
    ).fetchone()
    news_id = int(row["id"])
    conn.execute(
        "INSERT INTO news_symbol_links (news_id, symbol_id, match_reason) VALUES (?, ?, 'test')",
        (news_id, symbol_id),
    )
    conn.execute(
        "INSERT INTO news_sentiment (news_id, score, label, positive_prob, negative_prob, "
        "neutral_prob, model, analyzed_at) VALUES (?, ?, ?, ?, ?, ?, 'test-model', NOW())",
        (news_id, score, "negative" if score < 0 else "positive", max(score, 0), max(-score, 0), 0.0),
    )


def test_no_open_positions_does_nothing(conn):
    # `react_to_negative_news` her zaman `run_intraday_watch`'ın zaten bir
    # kez `get_state` çağırdığı (dolayısıyla portfolio satırının var olduğu)
    # bir bağlamda çalışır — bu testte de aynı ön koşulu kuruyoruz.
    ensure_portfolio(conn, CFG.starting_balance)
    stats = react_to_negative_news(conn, set(), CFG, negative_threshold=-0.5)
    assert stats == {"analyzed": 0, "sold": 0, "sold_symbols": []}


def test_sells_position_with_strong_negative_news(conn):
    opened_at = date(2026, 9, 1)
    symbol_id = _open_position(conn, opened_at)
    _insert_scored_news(conn, symbol_id, score=-0.8, published_at="2026-09-02 10:00:00")

    stats = react_to_negative_news(conn, {symbol_id}, CFG, negative_threshold=-0.5, decision_date=date(2026, 9, 2))

    assert stats["sold"] == 1
    assert stats["sold_symbols"] == [symbol_id]
    assert get_open_position(conn, symbol_id) is None


def test_does_not_sell_on_mild_negative_news(conn):
    opened_at = date(2026, 9, 1)
    symbol_id = _open_position(conn, opened_at)
    _insert_scored_news(conn, symbol_id, score=-0.2, published_at="2026-09-02 10:00:00")

    stats = react_to_negative_news(conn, {symbol_id}, CFG, negative_threshold=-0.5, decision_date=date(2026, 9, 2))

    assert stats["sold"] == 0
    assert get_open_position(conn, symbol_id) is not None


def test_does_not_sell_on_positive_news(conn):
    opened_at = date(2026, 9, 1)
    symbol_id = _open_position(conn, opened_at)
    _insert_scored_news(conn, symbol_id, score=0.9, published_at="2026-09-02 10:00:00")

    stats = react_to_negative_news(conn, {symbol_id}, CFG, negative_threshold=-0.5, decision_date=date(2026, 9, 2))

    assert stats["sold"] == 0
    assert get_open_position(conn, symbol_id) is not None


def test_ignores_news_published_before_position_opened(conn):
    opened_at = date(2026, 9, 5)
    symbol_id = _open_position(conn, opened_at)
    # Pozisyon açılmadan ÖNCE gelmiş, çoktan fiyatlanmış olması gereken haber.
    before = (opened_at - timedelta(days=1)).isoformat()
    _insert_scored_news(conn, symbol_id, score=-0.9, published_at=f"{before} 10:00:00")

    stats = react_to_negative_news(conn, {symbol_id}, CFG, negative_threshold=-0.5, decision_date=opened_at)

    assert stats["sold"] == 0
    assert get_open_position(conn, symbol_id) is not None


def test_only_reacts_to_held_symbols(conn):
    opened_at = date(2026, 9, 1)
    symbol_id = _open_position(conn, opened_at)
    other_symbol_id = upsert_symbol(conn, "AKBNK.IS", "BIST", "TRY")
    _insert_scored_news(conn, other_symbol_id, score=-0.9, published_at="2026-09-02 10:00:00")

    # held_symbol_ids'te sadece THYAO.IS var, AKBNK.IS'teki kötü haber yok sayılmalı.
    stats = react_to_negative_news(conn, {symbol_id}, CFG, negative_threshold=-0.5, decision_date=date(2026, 9, 2))

    assert stats["sold"] == 0
    assert get_open_position(conn, symbol_id) is not None
