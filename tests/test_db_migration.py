from pipeline.db import (
    count_symbols,
    get_news_id_by_url,
    init_schema,
    insert_news,
    upsert_prediction,
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


def test_init_schema_is_idempotent(committed_conn):
    """Faz 7 migration'ı `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` — init_schema
    her açılışta çalıştığı için ikinci (ve üçüncü) çağrı hata vermemeli."""
    init_schema(committed_conn)
    init_schema(committed_conn)
    committed_conn.commit()

    columns = {
        c["column_name"]
        for c in committed_conn.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'predictions'
            """
        ).fetchall()
    }
    assert "expected_return" in columns


def test_expected_return_migration_preserves_existing_rows(committed_conn):
    """Migration kolonu predictions'a SONRADAN ekler; kolonu düşürmeden
    (yıkıcı olurdu) doğrulama şu: migration sonrası da bir satır yazılıp
    init_schema tekrar çalıştırıldığında o satır BOZULMAZ — eski alanları
    aynen korunur, expected_return NULL olarak kalır."""
    symbol_id = upsert_symbol(committed_conn, "THYAO.IS", "BIST", "TRY")
    upsert_prediction(
        committed_conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0"
    )
    committed_conn.commit()

    init_schema(committed_conn)  # migration ikinci kez çalışıyor

    row = committed_conn.execute(
        "SELECT * FROM predictions WHERE symbol_id = ? AND feature_date = '2026-09-10'",
        (symbol_id,),
    ).fetchone()
    assert row is not None, "migration mevcut predictions satırını sildi"
    assert row["prob_up"] == 0.75
    assert row["predicted_up"] == 1
    assert row["model_version"] == "1.0"
    assert row["expected_return"] is None


def test_upsert_prediction_expected_return_roundtrip(committed_conn):
    """expected_return default'lu: eski imzayla çağıranlar bozulmamalı, yeni
    değer INSERT ve ON CONFLICT DO UPDATE yollarında da yazılmalı."""
    symbol_id = upsert_symbol(committed_conn, "THYAO.IS", "BIST", "TRY")

    upsert_prediction(committed_conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    row = committed_conn.execute(
        "SELECT expected_return FROM predictions WHERE feature_date = '2026-09-10'"
    ).fetchone()
    assert row["expected_return"] is None  # verilmeyen parametre NULL

    upsert_prediction(
        committed_conn, symbol_id, "2026-09-10", "2026-09-11", 0.80, 1, "1.0",
        expected_return=0.0123,
    )
    row = committed_conn.execute(
        "SELECT prob_up, expected_return FROM predictions WHERE feature_date = '2026-09-10'"
    ).fetchone()
    assert row["prob_up"] == 0.80
    assert abs(row["expected_return"] - 0.0123) < 1e-9
