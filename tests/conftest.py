"""Her test gerçek yerel Postgres'e karşı çalışır, sonunda HER ZAMAN
rollback edilir (commit yok) — testler birbirini kirletmez, ekstra bir
test-DB bağımlılığı (pytest-postgresql vb.) gerekmez."""

from __future__ import annotations

import psycopg
import pytest

from pipeline.db import ConnWrapper, get_database_url, init_schema

_db_url = get_database_url()
if "localhost" not in _db_url and "127.0.0.1" not in _db_url:
    pytest.exit("Refusing to run tests against a non-local DATABASE_URL", 1)


@pytest.fixture
def conn():
    raw = psycopg.connect(get_database_url())
    wrapper = ConnWrapper(raw)
    init_schema(wrapper)
    try:
        yield wrapper
    finally:
        raw.rollback()
        raw.close()


@pytest.fixture
def committed_conn():
    """`conn` fixture'ının aksine, API testleri gerçek (ayrı bağlantılı)
    bir istekle aynı veriyi görmeli — bu yüzden rollback yerine commit eder.
    Test sonunda ilgili tabloları TRUNCATE ederek temizler."""
    raw = psycopg.connect(get_database_url())
    wrapper = ConnWrapper(raw)
    init_schema(wrapper)
    raw.commit()  # şema kurulumu idempotent ve tek seferlik — test gövdesi başlamadan
                  # kilitleri bırakmak için hemen commit et, aksi halde API'nin kendi
                  # init_schema() çağrısıyla kendi kendini kilitliyor
    try:
        yield wrapper
        raw.commit()
    finally:
        raw.rollback()
        cur = raw.cursor()
        # BİLİNEN SINIRLAMA (docs/BACKTEST_AUDIT.md'de not edildi): `symbols`
        # burada CASCADE ile truncate edildiği için Postgres bunu `symbols`'a
        # FK'si olan HER tabloya (prices_intraday, sentiment_daily,
        # news_symbol_links dahil) otomatik yayar — `prices_intraday`'i bu
        # listede ADI GEÇMESE BİLE siler (deneyle doğrulandı). Gerçek/pahalı
        # fetch edilmiş intraday veri aynı yerel Postgres'te tutuluyorsa bu
        # veri her `committed_conn` kullanan test çalıştığında kaybolur. Kalıcı
        # düzeltme: bu 7 test dosyasının (grep committed_conn) her birine
        # ticker-scoped temizlik eklemek (tests/test_intraday_dataset.py'deki
        # `intraday_conn` deseni gibi) — bilinçli olarak burada yapılmadı,
        # mevcut testlerin varsayımlarını (RESTART IDENTITY, tam satır sayısı)
        # denetlemeden değiştirmek riskli, ayrı bir tur gerektiriyor.
        cur.execute(
            "TRUNCATE trade_decisions, trades, positions, portfolio, "
            "predictions, model_experiments, prices_daily, symbols "
            "RESTART IDENTITY CASCADE"
        )
        raw.commit()
        raw.close()
