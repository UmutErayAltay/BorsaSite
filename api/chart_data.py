"""Grafik verisi: saatlik (yfinance) + günlük/haftalık/aylık/yıllık (DB)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from pipeline.db import get_connection

INTERVALS = {
    "1h": {"label": "Saatlik", "source": "yfinance"},
    "1d": {"label": "Günlük", "source": "db", "days": 365},
    "1w": {"label": "Haftalık", "source": "db", "days": 1825},
    "1m": {"label": "Aylık", "source": "db", "days": 3650},
    "1y": {"label": "Yıllık", "source": "db", "days": 7300},
}


def _load_daily_from_db(symbol_id: int, days: int) -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM prices_daily
            WHERE symbol_id = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (symbol_id, days),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def _fetch_hourly(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period="60d", interval="1h", auto_adjust=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["close"])


def _series_payload(df: pd.DataFrame, interval: str, label: str) -> dict[str, Any]:
    if df.empty:
        return {
            "interval": interval,
            "interval_label": label,
            "dates": [],
            "close": [],
            "open": [],
            "high": [],
            "low": [],
            "change_pct": 0.0,
            "direction": "neutral",
            "period_changes": [],
        }

    close = df["close"]
    period_returns = close.pct_change().fillna(0) * 100

    first = float(close.iloc[0])
    last = float(close.iloc[-1])
    change_pct = ((last - first) / first * 100) if first else 0.0
    direction = "up" if change_pct > 0.05 else "down" if change_pct < -0.05 else "neutral"

    def fmt_index(ts) -> str:
        if interval == "1h":
            return ts.strftime("%Y-%m-%d %H:%M")
        if interval == "1y":
            return ts.strftime("%Y")
        if interval == "1m":
            return ts.strftime("%Y-%m")
        return ts.strftime("%Y-%m-%d")

    dates = [fmt_index(ts) for ts in df.index]
    period_changes = [
        {
            "date": dates[i],
            "change_pct": round(float(period_returns.iloc[i]), 2),
            "up": float(period_returns.iloc[i]) >= 0,
            "close": round(float(close.iloc[i]), 4),
        }
        for i in range(len(df))
        if i > 0
    ]

    return {
        "interval": interval,
        "interval_label": label,
        "dates": dates,
        "close": [round(float(v), 4) for v in close],
        "open": [round(float(v), 4) for v in df["open"]],
        "high": [round(float(v), 4) for v in df["high"]],
        "low": [round(float(v), 4) for v in df["low"]],
        "change_pct": round(change_pct, 2),
        "direction": direction,
        "period_changes": period_changes[-120:],
    }


def get_chart_data(ticker: str, symbol_id: int, interval: str) -> dict[str, Any]:
    if interval not in INTERVALS:
        raise ValueError(f"Geçersiz interval: {interval}")

    meta = INTERVALS[interval]
    label = meta["label"]

    if interval == "1h":
        df = _fetch_hourly(ticker)
        return _series_payload(df, interval, label)

    days = int(meta.get("days", 365))
    daily = _load_daily_from_db(symbol_id, days)
    if daily.empty:
        return _series_payload(daily, interval, label)

    if interval == "1d":
        df = daily
    elif interval == "1w":
        df = _resample_ohlc(daily, "W-FRI")
    elif interval == "1m":
        df = _resample_ohlc(daily, "ME")
    elif interval == "1y":
        df = _resample_ohlc(daily, "YE")
    else:
        df = daily

    return _series_payload(df, interval, label)
