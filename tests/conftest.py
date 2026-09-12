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
