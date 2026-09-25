"""Fiyat + sentiment birleşik eğitim veri seti.

Sentiment günlük agregasyondur ve `sentiment_availability_lag_days` ile D
kapanışına kaydırılır: D gününün agregasyonu, o günün haberlerinin kapanıştan
SONRA yayınlanmış olabileceği ihtimaliyle D-kararında henüz mevcut sayılmaz
(bkz. config/model.yaml::features, `backtest/walk_forward.py` docstring'i).
Tam intraday zamanlama güvenliği için ayrıca published_at/available_at
timestamp kontrolü gerekir — bu, günlük agregasyonun tutucu bir yaklaşımıdır.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pipeline.db import get_connection, init_schema
from pipeline.features import add_technical_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG = PROJECT_ROOT / "config" / "model.yaml"

FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "return_1d",
    "return_5d",
    "volume_ratio",
    "sentiment_avg_3d",
    "sentiment_news_3d",
    # is_bist bilerek YOK: walk-forward hep bist_only=True filtreliyor, o
    # zaman bu sütun sabit 1.0 oluyor (bkz. plan: Faz 1 madde 5). Kolon
    # kendisi (filtreleme için) df'te duruyor, sadece model feature'ı değil.
]


def load_model_config() -> dict[str, Any]:
    with open(MODEL_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_sentiment(conn) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT symbol_id, date, avg_score, news_count
        FROM sentiment_daily
        """
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["symbol_id", "date", "avg_score", "news_count"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_dataset(min_days: int | None = None, require_target: bool = True) -> pd.DataFrame:
    """`require_target=False`: walk-forward ve canlı tahmin gibi en sondaki
    (henüz next-day kapanışı bilinmeyen, `target_up=NaN` olan — bkz.
    pipeline/features.py) satırın da lazım olduğu çağrılar için o satırları
    ATMAZ. Eğitim yolu (`pipeline/train_model.py`) varsayılanı kullanmaya
    devam eder."""
    cfg = load_model_config()
    feat_cfg = cfg["features"]
    min_days = min_days or int(feat_cfg.get("min_history_days", 60))
    lag = int(feat_cfg.get("sentiment_lag_days", 3))
    availability_lag = int(feat_cfg.get("sentiment_availability_lag_days", 1))

    frames: list[pd.DataFrame] = []

    with get_connection() as conn:
        init_schema(conn)
        sentiment = _load_sentiment(conn)
        symbols = conn.execute(
            "SELECT id, ticker, market FROM symbols ORDER BY ticker"
        ).fetchall()

        for sym in symbols:
            symbol_id = int(sym["id"])
            prices = conn.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM prices_daily
                WHERE symbol_id = ?
                ORDER BY date
                """,
                (symbol_id,),
            ).fetchall()
            if len(prices) < min_days + 5:
                continue

            pdf = pd.DataFrame([dict(r) for r in prices])
            pdf["date"] = pd.to_datetime(pdf["date"])
            pdf = pdf.set_index("date").sort_index()
            pdf = add_technical_features(pdf, feat_cfg)

            pdf["symbol_id"] = symbol_id
            pdf["ticker"] = sym["ticker"]
            pdf["is_bist"] = 1.0 if sym["market"] == "BIST" else 0.0

            if not sentiment.empty:
                s = sentiment[sentiment["symbol_id"] == symbol_id].copy()
                s = s.set_index("date")[["avg_score", "news_count"]].sort_index()
                # `sentiment_daily`, haberi olmayan günler için hiç satır
                # üretmez (bkz. pipeline/db.py::rebuild_sentiment_daily) —
                # önce fiyat takviminin ÜZERİNE reindex edip sonra kaydırmak
                # gerekiyor, aksi halde shift() haberli günler arasındaki
                # SIRAYLA kayar, aradaki işlem günü sayısına göre değil.
                s = s.reindex(pdf.index).fillna(0.0)
                # D'nin günlük agregasyonu D-kararında henüz mevcut sayılmaz;
                # tam işlem günü kadar geriye kaydır (bkz. modül docstring'i).
                s = s.shift(availability_lag)
                merged = pdf.join(s, how="left")
                merged["avg_score"] = merged["avg_score"].fillna(0)
                merged["news_count"] = merged["news_count"].fillna(0)
                merged["sentiment_avg_3d"] = merged["avg_score"].rolling(lag, min_periods=1).mean()
                merged["sentiment_news_3d"] = merged["news_count"].rolling(lag, min_periods=1).sum()
            else:
                merged = pdf.copy()
                merged["sentiment_avg_3d"] = 0.0
                merged["sentiment_news_3d"] = 0.0

            merged = merged.reset_index()
            merged["feature_date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
            frames.append(merged)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Cross-sectional (piyasa-göreli) binary etiket: mutlak yön yerine o günün
    # BIST medyanının ÜSTÜNDE getiri. Tüm semboller birleştikten SONRA kurulur,
    # yoksa medyan sembol başına ayrı ayrı hesaplanır (bkz. features.py yorumu).
    bist_med = (
        df[df["is_bist"] == 1.0]
        .groupby("feature_date")["forward_return"]
        .median()
    )
    day_med = df["feature_date"].map(bist_med)
    df["target_up"] = np.where(
        df["forward_return"].isna() | day_med.isna(),
        np.nan,
        (df["forward_return"] > day_med).astype(float),
    )

    df = df.dropna(subset=FEATURE_COLUMNS)
    # Training needs a known label; live prediction and walk-forward need only
    # features and must keep the most recent row even though its target_up is
    # unknown until tomorrow (see pipeline/features.py — that row now
    # legitimately has target_up=NaN instead of a silently-wrong 0.0).
    if require_target:
        df = df[df["target_up"].notna()]
    return df
