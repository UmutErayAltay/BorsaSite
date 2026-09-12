from pipeline.db import (
    count_symbols,
    get_news_id_by_url,
    insert_news,
    upsert_prices,
    upsert_symbol,
)


def test_init_schema_creates_all_tables(conn):
    tables = conn.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        """
    ).fetchall()
    names = {t["table_name"] for t in tables}
    expected = {
        "symbols", "prices_daily", "news_raw", "news_symbol_links",
        "news_sentiment", "sentiment_daily", "predictions",
        "portfolio", "positions", "trades", "trade_decisions",
    }
    assert expected.issubset(names)


def test_upsert_symbol_roundtrip(conn):
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY", "Türk Hava Yolları")
    assert count_symbols(conn) == 1
    same_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY", "Türk Hava Yolları (güncel)")
    assert same_id == symbol_id
    assert count_symbols(conn) == 1


def test_upsert_prices_reports_rowcount(conn):
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    changed = upsert_prices(
        conn, symbol_id,
        [("2026-09-10", 100.0, 105.0, 99.0, 104.0, 104.0, 1000.0)],
    )
    assert changed == 1


def test_insert_news_ignores_duplicate_url(conn):
    news_id = insert_news(
        conn, "test-source", None, "Başlık", "Özet",
        "https://example.com/haber-1", "2026-09-10T10:00:00", "tr",
    )
    assert news_id is not None
    duplicate_id = insert_news(
        conn, "test-source", None, "Farklı başlık", "Özet",
        "https://example.com/haber-1", "2026-09-10T10:00:00", "tr",
    )
    assert duplicate_id is None
    assert get_news_id_by_url(conn, "https://example.com/haber-1") == news_id
