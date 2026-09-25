"""`build_dataset()` kendi Postgres bağlantısını açar (bkz. pipeline/dataset.py
::build_dataset -> get_connection) — bu yüzden `conn` (rollback-only) fixture'ı
DEĞİL, `committed_conn` kullanılmalı: aksi halde test verisi hiçbir zaman
commit edilmez ve build_dataset()'in kendi init_schema() çağrısı `conn`
fixture'ının hâlâ açık transaction'ıyla kilitlenir (bkz. tests/conftest.py::
committed_conn docstring'i, aynı desen)."""
from datetime import date, timedelta

from pipeline.dataset import build_dataset
from pipeline.db import upsert_prices, upsert_symbol


def _insert_sentiment_daily(committed_conn, symbol_id: int, rows) -> None:
    for d, avg_score, news_count in rows:
        committed_conn.execute(
            "INSERT INTO sentiment_daily (symbol_id, date, avg_score, news_count, updated_at) "
            "VALUES (?, ?, ?, ?, NOW())",
            (symbol_id, d, avg_score, news_count),
        )
    committed_conn.commit()


def _seed_prices(committed_conn, symbol_id: int, n_days: int = 72, start: date = date(2025, 1, 1)) -> list[str]:
    """Fiyatlar hem yükselip hem düşmeli — tekdüze artan bir seri RSI'nin
    avg_loss'unu hep 0 bırakıp RS'yi tanımsız (NaN) yapar, bu da
    dropna(FEATURE_COLUMNS)'ta TÜM satırları siler."""
    rows = []
    d = start
    for i in range(n_days):
        close = 100.0 + (i % 7) * 2.0 + i * 0.05
        volume = 1000.0 + (i % 5) * 50.0
        rows.append((d.isoformat(), close, close, close, close, close, volume))
        d += timedelta(days=1)
    upsert_prices(committed_conn, symbol_id, rows)
    committed_conn.commit()
    return [r[0] for r in rows]


def test_sentiment_is_not_visible_on_its_own_day(committed_conn):
    """Bir haber D gününün kapanışından SONRA yayınlanmış olabilir; bu yüzden
    D'nin sentiment agregasyonu D'nin kendi feature satırında görünmemeli."""
    symbol_id = upsert_symbol(committed_conn, "THYAO.IS", "BIST", "TRY", sector="Ulaştırma")
    dates = _seed_prices(committed_conn, symbol_id)
    spike_date = dates[-2]
    _insert_sentiment_daily(committed_conn, symbol_id, [(spike_date, 0.9, 5)])

    df = build_dataset(require_target=False)

    spike_row = df[df["feature_date"] == spike_date]
    assert not spike_row.empty
    assert spike_row.iloc[0]["sentiment_avg_3d"] == 0.0
    assert spike_row.iloc[0]["sentiment_news_3d"] == 0.0


def test_sentiment_becomes_visible_after_availability_lag(committed_conn):
    """Varsayılan `sentiment_availability_lag_days: 1` ile, D'nin sentiment'i
    D+1 satırında görünür olmalı — reindex+shift bunu haberli gün sayısına
    göre değil, gerçek işlem günü sayısına göre yapmalı."""
    symbol_id = upsert_symbol(committed_conn, "THYAO.IS", "BIST", "TRY", sector="Ulaştırma")
    dates = _seed_prices(committed_conn, symbol_id)
    spike_date = dates[-2]
    next_date = dates[-1]
    _insert_sentiment_daily(committed_conn, symbol_id, [(spike_date, 0.9, 5)])

    df = build_dataset(require_target=False)

    next_row = df[df["feature_date"] == next_date]
    assert not next_row.empty
    assert next_row.iloc[0]["sentiment_avg_3d"] > 0.0
    assert next_row.iloc[0]["sentiment_news_3d"] == 5.0


def test_sparse_sentiment_shifts_by_trading_days_not_by_news_rows(committed_conn):
    """İki haberli gün arasında haber olmayan günler varken, shift her iki
    kaydı da BİRBİRİNE göre değil, kendi işlem gününe göre kaydırmalı."""
    symbol_id = upsert_symbol(committed_conn, "THYAO.IS", "BIST", "TRY", sector="Ulaştırma")
    dates = _seed_prices(committed_conn, symbol_id)
    first_news_date = dates[-10]
    second_news_date = dates[-2]  # aradaki günlerde hiç haber yok
    _insert_sentiment_daily(
        committed_conn,
        symbol_id,
        [(first_news_date, 0.5, 1), (second_news_date, 0.9, 5)],
    )

    df = build_dataset(require_target=False)

    # first_news_date'ten hemen sonraki gün sentiment'i görmeli...
    idx = dates.index(first_news_date)
    day_after_first = dates[idx + 1]
    row = df[df["feature_date"] == day_after_first]
    assert not row.empty
    assert row.iloc[0]["sentiment_avg_3d"] > 0.0

    # ...ama ikinci haberin kendisinden ÖNCEKİ günde henüz ikinci haber
    # görünmemeli (henüz kaymamış).
    row_before_second = df[df["feature_date"] == second_news_date]
    assert not row_before_second.empty
    assert row_before_second.iloc[0]["sentiment_avg_3d"] == 0.0
