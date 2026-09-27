"""Yahoo Finance üzerinden intraday (1m/5m/15m/30m/1h) OHLCV bar verisi çeker.

Gerçek sınırlar (docs/BACKTEST_AUDIT.md §8, 2026-09-27 gerçek ağ isteğiyle
doğrulandı): 1h 730 gün geriye, 5m/15m/30m 60 gün, 1m ~7 gün. Ticker(...).
history() kullanılıyor (download() DEĞİL) - tek-seviyeli kolon, mevcut
fetch_prices.py ile aynı desen.

Her BIST sembolünde günde 1 açılış-müzayedesi anlık-görüntü bar'ı
(Volume=0, O=H=L=C) geliyor - konumu sabit değil, Volume==0 koşuluyla
atılıyor (pozisyona göre DEĞİL - bkz. audit)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yfinance as yf

from pipeline.db import (
    count_intraday_prices,
    get_connection,
    init_schema,
    upsert_intraday_prices,
    upsert_symbol,
)
from pipeline.fetch_prices import _float_or_none, load_symbols_config

logger = logging.getLogger(__name__)

MAX_PERIOD_BY_INTERVAL: dict[str, str] = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
}


def parse_intraday_bars(df) -> list[tuple]:
    """Yfinance intraday DataFrame'ini DB satır formatına dönüştürür.

    Volume==0 olan bar'lar açılış müzayedesi anlık-görüntü artefaktıdır,
    gerçek işlem değildir — konumları sabit olmadığı için (interval'e göre
    değişir) POZİSYONA göre değil DEĞER koşuluna göre elenir."""
    if df is None or df.empty:
        return []

    rows: list[tuple] = []
    for idx, row in df.iterrows():
        close = _float_or_none(row.get("Close"))
        if close is None:
            continue
        volume = _float_or_none(row.get("Volume"))
        if volume is None or volume == 0.0:
            continue
        rows.append(
            (
                idx.isoformat(),
                _float_or_none(row.get("Open")),
                _float_or_none(row.get("High")),
                _float_or_none(row.get("Low")),
                close,
                volume,
            )
        )
    return rows


def fetch_intraday_history(ticker: str, interval: str) -> list[tuple]:
    """Yfinance'tan intraday barlar; DB satır formatına dönüştürür."""
    if interval not in MAX_PERIOD_BY_INTERVAL:
        raise ValueError(
            f"Bilinmeyen interval: {interval!r}. "
            f"Geçerli değerler: {sorted(MAX_PERIOD_BY_INTERVAL)}"
        )
    df = yf.Ticker(ticker).history(
        period=MAX_PERIOD_BY_INTERVAL[interval],
        interval=interval,
        auto_adjust=False,
    )
    return parse_intraday_bars(df)


def run(
    interval: str = "15m",
    symbols_path: Path | None = None,
) -> dict[str, int]:
    """Tüm semboller için intraday fiyat çek ve veritabanına yaz."""
    if interval not in MAX_PERIOD_BY_INTERVAL:
        raise ValueError(
            f"Bilinmeyen interval: {interval!r}. "
            f"Geçerli değerler: {sorted(MAX_PERIOD_BY_INTERVAL)}"
        )

    entries = load_symbols_config(symbols_path)
    stats = {"symbols": 0, "bar_rows": 0, "errors": 0, "skipped": 0}

    with get_connection() as conn:
        init_schema(conn)

        for entry in entries:
            ticker = entry["ticker"]
            try:
                rows = fetch_intraday_history(ticker, interval)
                if not rows:
                    logger.warning("Veri yok: %s (%s)", ticker, interval)
                    stats["skipped"] += 1
                    continue

                symbol_id = upsert_symbol(
                    conn,
                    ticker=ticker,
                    market=entry["market"],
                    currency=entry["currency"],
                )
                upsert_intraday_prices(conn, symbol_id, interval, iter(rows))
                stats["symbols"] += 1
                stats["bar_rows"] += len(rows)
                logger.info("%s (%s): %d bar kaydedildi", ticker, interval, len(rows))
            except Exception as e:
                logger.exception("Hata %s: %s", ticker, e)
                stats["errors"] += 1

        stats["total_intraday_db"] = count_intraday_prices(conn)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(interval=sys.argv[1] if len(sys.argv) > 1 else "15m")
    print("Tamamlandı:", result)
