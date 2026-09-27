"""Portföy eğrisini BIST 100 endeksine göre normalize edip karşılaştırır."""

from __future__ import annotations

import logging
import time
from bisect import bisect_right
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 6 * 60 * 60
_cache: dict[tuple[str, str, str], tuple[float, pd.Series]] = {}


def fetch_index_closes(start: date, end: date, ticker: str = "XU100.IS") -> pd.Series:
    """Endeks günlük kapanışları — index'i `date` olan float Series. Hata veya boş
    yanıtta loglar ve BOŞ Series döner (sessizce yutmaz)."""
    key = (ticker, start.isoformat(), end.isoformat())
    now = time.monotonic()
    cached = _cache.get(key)
    if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    series = pd.Series(dtype=float)
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if raw is not None and len(raw) > 0:
            close = raw["Close"]
            # yfinance tek ticker'da bile MultiIndex sütun döndürebiliyor
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            series = pd.to_numeric(close, errors="coerce").astype(float)
            series = series[np.isfinite(series)]
            series.index = pd.to_datetime(series.index).date
            series = series[~series.index.duplicated(keep="last")].sort_index()
        else:
            logger.warning("Endeks verisi boş döndü: %s %s..%s", ticker, start, end)
    except Exception as exc:
        logger.warning("Endeks verisi alınamadı (%s %s..%s): %s", ticker, start, end, exc)

    if series.empty:
        return pd.Series(dtype=float)
    # Yalnızca başarılı sonuç önbelleklenir; aksi halde tek bir ağ hatası
    # 6 saat boyunca endpoint'i boş veriyle kilitlerdi.
    _cache[key] = (now, series)
    return series


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _sorted_closes(index_closes: pd.Series | None) -> tuple[list[date], list[float]]:
    if index_closes is None or len(index_closes) == 0:
        return [], []
    pairs: list[tuple[date, float]] = []
    for stamp, value in index_closes.items():
        try:
            pairs.append((_as_date(stamp), float(value)))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda pair: pair[0])
    return [day for day, _ in pairs], [close for _, close in pairs]


def _close_as_of(days: list[date], closes: list[float], target: date) -> float | None:
    """Hedef tarihteki, yoksa ondan önceki son kapanış (tatil/eksik gün → önceki gün)."""
    pos = bisect_right(days, target) - 1
    return closes[pos] if pos >= 0 else None


def _return_pct(first: float | None, last: float | None) -> float | None:
    if first in (None, 0) or last is None:
        return None
    return round((last - first) / first * 100, 2)


def build_benchmark_payload(
    snapshots: list[dict],
    index_closes: pd.Series | None,
    label: str = "BIST 100",
    ticker: str = "XU100.IS",
) -> dict[str, Any]:
    """Saf fonksiyon (I/O yok): portföy ve endeks eğrilerini ilk değere normalize
    edip getiri farkını hesaplar."""
    points = sorted(
        ((_as_date(s["snapshot_date"]), float(s["total_value"])) for s in snapshots),
        key=lambda p: p[0],
    )
    if not points:
        return {
            "items": [],
            "benchmark_label": label,
            "benchmark_ticker": ticker,
            "benchmark_available": False,
            "portfolio_return_pct": None,
            "benchmark_return_pct": None,
            "difference_pct_points": None,
            "note": "Portföy verisi yok",
        }

    close_days, close_values = _sorted_closes(index_closes)
    base_close = _close_as_of(close_days, close_values, points[0][0])
    available = base_close is not None
    base_value = points[0][1]

    items: list[dict[str, Any]] = []
    last_close: float | None = None
    for day, total_value in points:
        close = _close_as_of(close_days, close_values, day) if available else None
        last_close = close
        items.append({
            "date": day.isoformat(),
            "portfolio": round(total_value, 2),
            "benchmark": round(base_value * close / base_close, 2) if close is not None else None,
        })

    portfolio_return = _return_pct(base_value, points[-1][1])
    benchmark_return = _return_pct(base_close, last_close) if available else None
    difference = (
        round(portfolio_return - benchmark_return, 2)
        if portfolio_return is not None and benchmark_return is not None
        else None
    )

    return {
        "items": items,
        "benchmark_label": label,
        "benchmark_ticker": ticker,
        "benchmark_available": available,
        "portfolio_return_pct": portfolio_return,
        "benchmark_return_pct": benchmark_return,
        "difference_pct_points": difference,
        "note": None if available else "Endeks verisi alınamadı",
    }
