"""`build_intraday_dataset` integration tests against the real local Postgres
(synthetic bars, no network) — mirrors tests/test_dataset.py.

DİKKAT — `tests/conftest.py::committed_conn` bu dosyada KULLANILMIYOR: onun
teardown'ı tüm tabloları TRUNCATE eder, yani veritabanındaki gerçek (ve
pahalı) intraday verisini siler. Aşağıdaki `intraday_conn` aynı "commit'li,
ayrı bağlantıdan görünür" davranışı verir ama teardown'da YALNIZCA bu
dosyanın kullandığı test ticker'larını temizler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg
import pytest

from pipeline.db import (
    ConnWrapper,
    get_database_url,
    init_schema,
    upsert_intraday_prices,
    upsert_symbol,
)
from pipeline.intraday_dataset import INTRADAY_FEATURE_COLUMNS, build_intraday_dataset

INTERVAL = "1h"

# Yalnız bu dosyada üretilen ticker'lar — temizlik bunlarla sınırlıdır.
_SEED_TICKERS = ("TEST1.IS", "TEST2.IS", "KEEP.IS", "SHORT.IS", "MULTI.IS")


@pytest.fixture
def intraday_conn():
    raw = psycopg.connect(get_database_url())
    wrapper = ConnWrapper(raw)
    init_schema(wrapper)
    raw.commit()  # kilitleri bırak, gövde başlasın
    try:
        yield wrapper
        raw.commit()
    finally:
        raw.rollback()
        cur = raw.cursor()
        cur.execute("SELECT id FROM symbols WHERE ticker = ANY(%s)", (list(_SEED_TICKERS),))
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            cur.execute("DELETE FROM prices_intraday WHERE symbol_id = ANY(%s)", (ids,))
            cur.execute("DELETE FROM symbols WHERE id = ANY(%s)", (ids,))
        raw.commit()
        raw.close()


def _seed_symbol_with_intraday_bars(conn, ticker: str, *, days: int, bars_per_day: int = 5) -> int:
    """Her gün aynı saat diliminde `bars_per_day` bar — `min_history_bars`
    eşiğini tek bir değişkenle geçebilmek için gün sayısı dışarıdan verilir."""
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    rows = []
    price = 100.0
    for d in range(days):
        day = pd.Timestamp("2026-01-05", tz="Europe/Istanbul") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            price += 0.1 if b % 2 == 0 else -0.05
            ts = day + pd.Timedelta(hours=10 + b)
            rows.append((ts.isoformat(), price, price + 0.5, price - 0.5, price, 1000.0 + b))
    upsert_intraday_prices(conn, symbol_id, INTERVAL, iter(rows))
    return symbol_id


def test_dataset_returns_all_feature_columns(intraday_conn):
    symbol_id = _seed_symbol_with_intraday_bars(intraday_conn, "TEST1.IS", days=60)
    intraday_conn.commit()  # build_intraday_dataset() opens its own connection

    df = build_intraday_dataset(interval=INTERVAL, min_bars=100)
    rows = df[df["symbol_id"] == symbol_id]

    assert not rows.empty
    # 300 bar; özellik ısınması (sma20/volume_ratio) bar 20'de biter -> 281
    # özellik-tam bar. Bunların 57'si bir GÜNÜN SON barı (60 günden 3'ü ısınma
    # penceresine denk geliyor, ısınma gün 0-3'te bitiyor) -> 224 etiketli.
    assert len(rows) == 300 - 19 - 57
    for col in INTRADAY_FEATURE_COLUMNS:
        assert col in df.columns
        assert rows[col].notna().all(), f"{col} NaN içeriyor"
    assert rows["is_bist"].eq(1.0).all()
    assert rows["ticker"].eq("TEST1.IS").all()
    # `feature_date` değil `feature_ts`: aynı günün birden fazla bar'ı aynı
    # zaman damgasını taşımaz.
    assert "feature_ts" in df.columns
    assert rows["feature_ts"].is_unique


def test_require_target_true_drops_day_boundary_bars(intraday_conn):
    symbol_id = _seed_symbol_with_intraday_bars(intraday_conn, "TEST2.IS", days=60)
    intraday_conn.commit()

    df = build_intraday_dataset(interval=INTERVAL, min_bars=100, require_target=False)
    rows = df[df["symbol_id"] == symbol_id]
    assert not rows.empty
    # Her günün son barı etiketsizdir — sadece serinin son gününün değil.
    unlabeled = rows[rows["target_up"].isna()]
    assert len(unlabeled) > 0
    assert len(unlabeled) == rows.groupby(rows["ts"].dt.date).ngroups
    assert unlabeled["bar_of_day"].eq(4).all()  # 5 bar/gün -> son bar 4
    assert unlabeled["target_return"].isna().all()

    train = build_intraday_dataset(interval=INTERVAL, min_bars=100, require_target=True)
    train_rows = train[train["symbol_id"] == symbol_id]
    assert not train_rows.empty
    assert train_rows["target_up"].notna().all()
    assert len(train_rows) == len(rows) - len(unlabeled)


def test_symbol_below_min_bars_is_skipped(intraday_conn):
    kept_id = _seed_symbol_with_intraday_bars(intraday_conn, "KEEP.IS", days=60)
    short_id = _seed_symbol_with_intraday_bars(intraday_conn, "SHORT.IS", days=5)
    intraday_conn.commit()

    df = build_intraday_dataset(interval=INTERVAL, min_bars=100)

    assert not df[df["symbol_id"] == kept_id].empty
    assert df[df["symbol_id"] == short_id].empty


def test_other_intervals_are_ignored(intraday_conn):
    """`interval` filtresi SQL'e giriyor: sembol 15m'de var ama 1h'de yok."""
    symbol_id = upsert_symbol(intraday_conn, "MULTI.IS", "BIST", "TRY")
    rows_in = []
    price = 50.0
    for d in range(25):
        for b in range(5):
            price += 0.1 if b % 2 == 0 else -0.05
            ts = pd.Timestamp("2026-01-05 10:00", tz="Europe/Istanbul") + pd.Timedelta(days=d, hours=b)
            rows_in.append((ts.isoformat(), price, price + 0.5, price - 0.5, price, 1000.0 + b))
    upsert_intraday_prices(intraday_conn, symbol_id, "15m", iter(rows_in))
    intraday_conn.commit()

    df_1h = build_intraday_dataset(interval=INTERVAL, min_bars=1)
    df_15m = build_intraday_dataset(interval="15m", min_bars=1, require_target=False)

    # 1h sonucu gerçek sembollerle dolu OLABİLİR; veritabanında 1h verisi
    # yoksa (ör. test paketinin `committed_conn` teardown'ı TRUNCATE ettiyse)
    # boş da gelebilir. Her iki durumda da kanıt aynı: MULTI.IS 1h'de YOK.
    if not df_1h.empty:
        assert (df_1h["symbol_id"] == symbol_id).sum() == 0
    rows_15m = df_15m[df_15m["symbol_id"] == symbol_id]
    assert not rows_15m.empty
    # Her günün son barı etiketsiz — 15m'de de aynı gün-sınırı disiplini.
    unlabeled = rows_15m[rows_15m["target_up"].isna()]
    assert len(unlabeled) == rows_15m.groupby(rows_15m["ts"].dt.date).ngroups
    assert unlabeled["bar_of_day"].eq(4).all()
