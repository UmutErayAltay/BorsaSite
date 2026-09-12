"""Her test gerçek yerel Postgres'e karşı çalışır, sonunda HER ZAMAN
rollback edilir (commit yok) — testler birbirini kirletmez, ekstra bir
test-DB bağımlılığı (pytest-postgresql vb.) gerekmez."""

from __future__ import annotations

import psycopg
import pytest

from pipeline.db import ConnWrapper, get_database_url, init_schema


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
    try:
        yield wrapper
        raw.commit()
    finally:
        cur = raw.cursor()
        cur.execute(
            "TRUNCATE trade_decisions, trades, positions, portfolio, "
            "predictions, prices_daily, symbols RESTART IDENTITY CASCADE"
        )
        raw.commit()
        raw.close()
