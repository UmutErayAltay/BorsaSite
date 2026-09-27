"""Intraday (1h) eğitim veri seti — `pipeline/dataset.py`'nin intraday karşılığı.

BİLİNÇLİ SINIRLAMA (v1): SENTİMENT KULLANILMIYOR. `sentiment_daily` günlük
granülaritededir ve günlük pipeline'daki `sentiment_availability_lag_days`
mantığı gün-içi zamanlamayı çözemez — bir günün agregasyonunu barlara
forward-fill etmek, haberin hangi bar içinde yayınlandığını bilmeden yanlış
zamanlama (look-ahead ya da gereksiz gecikme) üretir. Doğru çözüm
published_at/available_at timestamp'ine dayanan ayrı bir birleştirme
katmanıdır; bu, kendi dalgalı ve daha büyük bir iş. `INTRADAY_FEATURE_COLUMNS`
buna yer açıyor (sentiment kolonu şimdilik yok).

Hedef tanımı ve gün-sınırı disiplini için bkz. `pipeline/intraday_features.py`
modül docstring'i.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.db import get_connection, init_schema
from pipeline.intraday_features import add_intraday_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTRADAY_MODEL_CONFIG = PROJECT_ROOT / "config" / "intraday_model.yaml"

INTRADAY_FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "close_sma20_ratio",
    "return_1",
    "return_5",
    "volume_ratio",
    "bar_of_day",
    "is_bist",
]


def load_intraday_model_config() -> dict[str, Any]:
    """config/intraday_model.yaml'i yükler (config/model.yaml DEĞİL)."""
    with open(INTRADAY_MODEL_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_intraday_dataset(
    interval: str | None = None,
    min_bars: int | None = None,
    require_target: bool = True,
) -> pd.DataFrame:
    """`require_target=False`: canlı tahmin gibi en sondaki (henüz gün içi
    sonraki barının kapanışı bilinmeyen, `target_up=NaN` olan) satırları
    ATMAZ. Eğitim yolu varsayılanı kullanmaya devam eder.

    Günlük pipeline'dan farklı olarak "en son satır" tek bir satır değildir:
    hedef tanımı gereği HER GÜN'ün son barı etiketsizdir (bkz.
    `pipeline/intraday_features.py`)."""
    cfg = load_intraday_model_config()
    feat_cfg = cfg["features"]
    interval = interval or cfg.get("interval", "1h")
    min_bars = min_bars or int(feat_cfg.get("min_history_bars", 200))

    frames: list[pd.DataFrame] = []

    with get_connection() as conn:
        init_schema(conn)
        symbols = conn.execute(
            "SELECT id, ticker, market FROM symbols ORDER BY ticker"
        ).fetchall()

        for sym in symbols:
            symbol_id = int(sym["id"])
            prices = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM prices_intraday
                WHERE symbol_id = ? AND interval = ?
                ORDER BY ts
                """,
                (symbol_id, interval),
            ).fetchall()
            if len(prices) < min_bars:
                continue

            pdf = pd.DataFrame([dict(r) for r in prices])
            # `ts` TIMESTAMPTZ olduğu için `pd.to_datetime` tz-aware bir index
            # üretir; gün sınırları bu index'ten tespit edilir.
            pdf["ts"] = pd.to_datetime(pdf["ts"])
            pdf = pdf.set_index("ts").sort_index()
            pdf = add_intraday_features(pdf, feat_cfg)

            pdf["symbol_id"] = symbol_id
            pdf["ticker"] = sym["ticker"]
            pdf["is_bist"] = 1.0 if sym["market"] == "BIST" else 0.0

            merged = pdf.reset_index()
            # Intraday'de tarih değil zaman damgası anlamlı: aynı güne ait iki
            # farklı bar aynı "feature_date"ye düşerdi (günlük pipeline'daki
            # `feature_date` yerine).
            merged["feature_ts"] = merged["ts"].astype(str)
            frames.append(merged)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=INTRADAY_FEATURE_COLUMNS)
    # Training needs a known label. Bu filtre, seri sonu satırının yanı sıra
    # GÜN SINIRINI GEÇEN barları da otomatik olarak eler — onların
    # `target_up`'ı `add_intraday_features` içinde zaten NaN yapıldı (gece
    # sıçraması hedeflenmez).
    if require_target:
        df = df[df["target_up"].notna()]
    return df
