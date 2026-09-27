"""`run()` ağ'a çıkar (yfinance); testler sadece saf ve DB parçalarını kapsar —
`parse_intraday_bars` elle kurulmuş SENTETİK bir DataFrame ile, DB testleri
gerçek yerel Postgres'e karşı (bkz. tests/conftest.py)."""

import pandas as pd
import pytest

from pipeline.db import count_intraday_prices, upsert_intraday_prices, upsert_symbol
from pipeline.fetch_intraday_prices import fetch_intraday_history, parse_intraday_bars

# Volume=0 bar'ı BILEREK ORTAYA koyuluyor: gerçek veride artefaktın konumu
# interval'e gore degisiyor (15m'de ~09:45, 5m'de ~09:55, 1h'de ~09:30) —
# filtre "hep ilk/son bar" diye POZISYONEL olsaydi bu test yanlis gecerdi.
_INDEX = pd.DatetimeIndex(
    [
        "2026-09-24T09:30:00+03:00",
        "2026-09-24T09:45:00+03:00",
        "2026-09-24T10:00:00+03:00",
        "2026-09-24T10:15:00+03:00",
        "2026-09-24T10:30:00+03:00",
    ]
)
_ZERO_VOLUME_IX = 2
_NAN_CLOSE_IX = 4


def _synthetic_df() -> pd.DataFrame:
    """Acilis muzayedesi anlik-goruntu bar'i (O=H=L=C, Volume=0) ortada,
    Close=NaN olan bar sonda."""
    df = pd.DataFrame(
        {
            "Open": [10.0, 10.1, 10.2, 10.3, 10.4],
            "High": [10.5, 10.6, 10.2, 10.8, 10.9],
            "Low": [9.5, 9.6, 10.2, 9.8, 9.9],
            "Close": [10.0, 10.2, 10.2, 10.4, float("nan")],
            "Volume": [1_000.0, 2_000.0, 0.0, 3_000.0, 4_000.0],
        },
        index=_INDEX,
    )
    return df


def test_parse_keeps_normal_bars():
    rows = parse_intraday_bars(_synthetic_df())

    assert len(rows) == 3
    assert [r[0] for r in rows] == [
        "2026-09-24T09:30:00+03:00",
        "2026-09-24T09:45:00+03:00",
        "2026-09-24T10:15:00+03:00",
    ]
    assert rows[0] == ("2026-09-24T09:30:00+03:00", 10.0, 10.5, 9.5, 10.0, 1_000.0)


def test_parse_drops_zero_volume_bar_wherever_it_sits():
    # Pozisyonel degil, DEGER kosuluyla eleniyor: Volume=0 olan bar ORTAYDA.
    rows = parse_intraday_bars(_synthetic_df())

    assert all(r[5] != 0.0 for r in rows)
    assert _INDEX[_ZERO_VOLUME_IX].isoformat() not in {r[0] for r in rows}
    assert len(rows) == len(_synthetic_df()) - 2  # sifir hacimli bar + NaN close'lu bar


def test_parse_drops_nan_close_bar():
    rows = parse_intraday_bars(_synthetic_df())

    assert _INDEX[_NAN_CLOSE_IX].isoformat() not in {r[0] for r in rows}


def test_parse_empty_dataframe():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    assert parse_intraday_bars(empty) == []


def test_fetch_rejects_unknown_interval():
    with pytest.raises(ValueError):
        fetch_intraday_history("THYAO.IS", "7m")


def test_upsert_inserts_and_counts(conn):
    before = count_intraday_prices(conn)
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")

    rows = [
        ("2026-09-24T10:00:00+03:00", 10.0, 10.5, 9.5, 10.2, 1_000.0),
        ("2026-09-24T10:15:00+03:00", 10.2, 10.8, 9.8, 10.4, 2_000.0),
    ]
    upsert_intraday_prices(conn, symbol_id, "15m", iter(rows))

    assert count_intraday_prices(conn) == before + 2


def test_upsert_updates_on_conflict(conn):
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    ts = "2026-09-24T10:00:00+03:00"
    upsert_intraday_prices(conn, symbol_id, "15m", iter([(ts, 10.0, 10.5, 9.5, 10.2, 1_000.0)]))
    before = count_intraday_prices(conn)

    upsert_intraday_prices(conn, symbol_id, "15m", iter([(ts, 11.0, 11.5, 10.5, 11.2, 9_000.0)]))

    assert count_intraday_prices(conn) == before  # yeni satir degil
    row = conn.execute(
        "SELECT close, volume FROM prices_intraday WHERE symbol_id = ? AND interval = ? AND ts = ?",
        (symbol_id, "15m", ts),
    ).fetchone()
    assert row["close"] == 11.2
    assert row["volume"] == 9_000.0


def test_same_ts_different_interval_are_distinct_rows(conn):
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    ts = "2026-09-24T10:00:00+03:00"
    before = count_intraday_prices(conn)

    upsert_intraday_prices(conn, symbol_id, "15m", iter([(ts, 10.0, 10.5, 9.5, 10.2, 1_000.0)]))
    upsert_intraday_prices(conn, symbol_id, "1h", iter([(ts, 10.0, 10.5, 9.5, 10.2, 1_000.0)]))

    assert count_intraday_prices(conn) == before + 2
