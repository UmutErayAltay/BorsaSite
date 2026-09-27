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
_SEED_TICKERS = ("TEST1.IS", "TEST2.IS", "KEEP.IS", "SHORT.IS", "MULTI.IS", "XS1.IS", "XS2.IS")


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


def _seed_symbol_with_intraday_bars(
    conn,
    ticker: str,
    *,
    days: int,
    bars_per_day: int = 5,
    direction: int = 1,
    price: float = 100.0,
) -> int:
    """Her gün aynı saat diliminde `bars_per_day` bar — `min_history_bars`
    eşiğini tek bir değişkenle geçebilmek için gün sayısı dışarıdan verilir.

    `direction`: +1 yükselen, -1 düşen seri (kesitsel testlerde iki sembolün
    zıt yönde hareket etmesi için)."""
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    rows = []
    for d in range(days):
        day = pd.Timestamp("2026-01-05", tz="Europe/Istanbul") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            price += direction * (0.1 if b % 2 == 0 else -0.05)
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
    # 300 bar (60 gün x 5). 2026-09-27 v2 dalga, yeni sembol-bazlı feature'lar
    # (`ret_since_open`, `gap`, `prev_day_ret`, `d_ret_5`) günlük geçmiş
    # istiyor ve bu geçmiş barlık ısınma penceresinden BAĞIMSIZ:
    #
    #   feature            ilk tam gün (0-based)   yanan gün sayısı
    #   rsi_14                     3                    3
    #   close_sma20_ratio          4                    4   (20 barlık pencere)
    #   volume_ratio               4                    4   (20 barlık pencere)
    #   vol_20                     4                    4   (20 barlık pencere)
    #   ret_since_open             0                    0   (gün içi, hep tanımlı)
    #   gap                        1                    1   (1 gün)
    #   prev_day_ret               2                    2   (pct_change + shift(1))
    #   d_ret_5                    6                    6   (5 gün + shift(1))
    #
    # Bağlayıcı olan `d_ret_5`: ilk 6 gün yanar -> 300 - 30 = 270 özellik-tam
    # bar (gün 7-60, 54 gün). Bu 54 günün her birinde son bar etiketsiz
    # (rest-of-day hedefi için gün sonu kapanışı gerekli) -> 270 - 54 = 216.
    #
    # Kesitsel kolonlar (`mkt_*`, `ex_*`, `rank_*`) ısınmaya KATKI YAPMIYOR:
    # aynı `ts`'deki diğer semboller her zaman hazır; kesitte tek sembol
    # kalsa bile `mean`/`rank` NaN üretmez (sıralama 1.0 olur).
    assert len(rows) == 300 - 30 - 54
    for col in INTRADAY_FEATURE_COLUMNS:
        assert col in df.columns
        assert rows[col].notna().all(), f"{col} NaN içeriyor"
    # 2026-09-27: `is_bist` kaldırıldı — dataset artık zaten sadece BIST
    # döndürüyor (bkz. pipeline/intraday_dataset.py modül docstring'i).
    assert "is_bist" not in df.columns
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


def test_cross_sectional_market_context(intraday_conn):
    """Kesitsel piyasa bağlamı: aynı `ts`'deki TÜM semboller üzerinden
    hesaplanır (bar t kapanışında bilinen bilgi, look-ahead yok).

    ÖNEMLİ: gerçek BIST evreninde dataset'e ~50 sembol girer, bu yüzden
    `mkt_ret_1` yalnız bu iki test sembolünün ortalaması DEĞİLDİR. Test
    buna dayanıklı yazıldı: beklenen ortalama, `dropna` SONRASI kalan
    satırların o `ts`'deki `return_1` ortalamasıdır.

    İki test sembolü farklı yönlerde hareket ediyor ki `ex_ret_1`'in
    işaretinin sembole göre değiştiği görülsün."""
    id_a = _seed_symbol_with_intraday_bars(intraday_conn, "XS1.IS", days=60, direction=1)
    id_b = _seed_symbol_with_intraday_bars(
        intraday_conn, "XS2.IS", days=60, direction=-1, price=250.0
    )
    intraday_conn.commit()

    df = build_intraday_dataset(interval=INTERVAL, min_bars=100)
    a = df[df["symbol_id"] == id_a]
    b = df[df["symbol_id"] == id_b]
    assert not a.empty and not b.empty
    # İki sembol de eşit uzunlukta (aynı takvim) -> kesişen `ts` var.
    shared_ts = sorted(set(a["ts"]) & set(b["ts"]))
    assert shared_ts

    for ts in shared_ts[:20] + shared_ts[-5:]:
        rows_at_ts = df[df["ts"] == ts]
        assert len(rows_at_ts) >= 2
        # Piyasa ortalaması = o `ts`'deki TÜM satırların `return_1` ortalaması.
        expected_mkt = rows_at_ts["return_1"].mean()
        ra = a[a["ts"] == ts].iloc[0]
        rb = b[b["ts"] == ts].iloc[0]
        for r in (ra, rb):
            assert r["mkt_ret_1"] == pytest.approx(expected_mkt)
            assert r["mkt_ret_since_open"] == pytest.approx(
                rows_at_ts["ret_since_open"].mean()
            )
            # Fazlalık (idiosinkratik) getiri = sembolün kendi getirisi
            # eksi piyasa ortalaması.
            assert r["ex_ret_1"] == pytest.approx(r["return_1"] - r["mkt_ret_1"])
            assert r["ex_ret_since_open"] == pytest.approx(
                r["ret_since_open"] - r["mkt_ret_since_open"]
            )
            # Genişlik: o `ts`'de yükselen sembollerin oranı.
            assert r["mkt_breadth"] == pytest.approx((rows_at_ts["return_1"] > 0).mean())
        # Sıralama [0, 1] aralığında ve bu `ts`'deki kesitin permütasyonu.
        ranks = rows_at_ts["rank_ret_5"]
        assert ranks.between(0.0, 1.0, inclusive="both").all()
        assert sorted(ranks.tolist()) == pytest.approx(
            sorted(ranks.rank(pct=True).tolist())
        )
        for r in (ra, rb):
            assert 0.0 <= r["rank_since_open"] <= 1.0
        # XS1 yukarı giderken XS2 aşağı -> sıralamada biri diğerinin üstünde.
        assert ra["rank_ret_5"] > rb["rank_ret_5"]

    # İki sembolün fazlalık getirileri zıt işaretli olmalı (ortak piyasa
    # hareketi çıkarıldığında sembole özgü bileşen kalmalı).
    common = a.merge(b, on="ts", suffixes=("_a", "_b"))
    assert len(common) > 0
    assert np.sign(common["ex_ret_1_a"]).ne(np.sign(common["ex_ret_1_b"])).mean() > 0.9


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
