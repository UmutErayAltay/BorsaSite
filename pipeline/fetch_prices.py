"""Yahoo Finance (yfinance) üzerinden günlük OHLCV fiyat verisi çeker."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
import yfinance as yf

from pipeline.db import (
    count_prices,
    count_symbols,
    get_connection,
    init_schema,
    upsert_prices,
    upsert_symbol,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYMBOLS_PATH = PROJECT_ROOT / "config" / "symbols.yaml"


def load_symbols_config(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or SYMBOLS_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entries: list[dict[str, Any]] = []
    for key in ("bist", "us"):
        block = cfg[key]
        for ticker in block["symbols"]:
            entries.append(
                {
                    "ticker": ticker,
                    "market": block["market"],
                    "currency": block["currency"],
                }
            )
    return entries


def fetch_history(
    ticker: str,
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
) -> list[tuple]:
    """Yfinance'tan günlük barlar; DB satır formatına dönüştürür."""
    stock = yf.Ticker(ticker)
    if start and end:
        df = stock.history(start=start, end=end, auto_adjust=False)
    else:
        df = stock.history(period=period, auto_adjust=False)

    if df.empty:
        return []

    rows: list[tuple] = []
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
        adj = row.get("Adj Close")
        if adj is None or (hasattr(adj, "__float__") and str(adj) == "nan"):
            adj = row["Close"]
        close = _float_or_none(row.get("Close"))
        if close is None:
            continue
        rows.append(
            (
                date_str,
                _float_or_none(row.get("Open")),
                _float_or_none(row.get("High")),
                _float_or_none(row.get("Low")),
                close,
                _float_or_none(adj),
                _float_or_none(row.get("Volume")),
            )
        )
    return rows


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def fetch_metadata(ticker: str) -> tuple[str | None, str | None]:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName"), info.get("sector")
    except Exception:
        return None, None


def run(
    period: str = "2y",
    incremental_days: int | None = None,
    symbols_path: Path | None = None,
) -> dict[str, int]:
    """
    Tüm semboller için fiyat çek ve veritabanına yaz.

    incremental_days: Son N günü güncelle (günlük cron için). None ise tam period.
    """
    entries = load_symbols_config(symbols_path)
    stats = {"symbols": 0, "price_rows": 0, "errors": 0, "skipped": 0}

    start = end = None
    if incremental_days is not None:
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=incremental_days + 5)).strftime("%Y-%m-%d")

    with get_connection() as conn:
        init_schema(conn)

        for entry in entries:
            ticker = entry["ticker"]
            try:
                rows = fetch_history(
                    ticker,
                    period=period if incremental_days is None else "1mo",
                    start=start,
                    end=end,
                )
                if not rows:
                    logger.warning("Veri yok: %s", ticker)
                    stats["skipped"] += 1
                    continue

                name, sector = fetch_metadata(ticker)
                symbol_id = upsert_symbol(
                    conn,
                    ticker=ticker,
                    market=entry["market"],
                    currency=entry["currency"],
                    name=name,
                    sector=sector,
                )
                upsert_prices(conn, symbol_id, iter(rows))
                stats["symbols"] += 1
                stats["price_rows"] += len(rows)
                logger.info("%s: %d bar kaydedildi", ticker, len(rows))
            except Exception as e:
                logger.exception("Hata %s: %s", ticker, e)
                stats["errors"] += 1

        stats["total_symbols_db"] = count_symbols(conn)
        stats["total_prices_db"] = count_prices(conn)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run()
    print("Tamamlandı:", result)
