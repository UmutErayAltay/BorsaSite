"""Intraday (1h) eğitim veri seti — `pipeline/dataset.py`'nin intraday karşılığı.

BİLİNÇLİ SINIRLAMA (v1): SENTİMENT KULLANILMIYOR. `sentiment_daily` günlük
granülaritededir ve günlük pipeline'daki `sentiment_availability_lag_days`
mantığı gün-içi zamanlamayı çözemez — bir günün agregasyonunu barlara
forward-fill etmek, haberin hangi bar içinde yayınlandığını bilmeden yanlış
zamanlama (look-ahead ya da gereksiz gecikme) üretir. Doğru çözüm
published_at/available_at timestamp'ine dayanan ayrı bir birleştirme
katmanıdır; bu, kendi dalgalı ve daha büyük bir iş. `INTRADAY_FEATURE_COLUMNS`
buna yer açıyor (sentiment kolonu şimdilik yok).

**2026-09-27 — SADECE BIST (mimari düzeltme, gerçek backtest sonrası):**
`trading/engine.py`'nin canlı alım sorgusu zaten SADECE `market = 'BIST'`
sembollerini alıyor (US sembolleri hiç ticaret edilmiyor, sadece izleniyor)
— ama bu modül önceden BIST+US karışık eğitiyordu ve `is_bist` feature
importance'ın %76.5'ini yiyordu (gerçek sinyalin üstünü örtüyordu, bkz.
docs/BACKTEST_AUDIT.md §11 madde 10). BIST-only yeniden eğitim ROC AUC'u
İYİLEŞTİRMEDİ (0.547 → 0.525, gerçekte biraz kötüleşti) — yani karışım
tek sorun değildi, ama artık en azından ticaret edilmeyen sembollerle
eğitim yapılmıyor ve `is_bist` gibi sabit/anlamsız bir feature yok.

Hedef tanımı ve gün-sinırı disiplini için bkz. `pipeline/intraday_features.py`
modül docstring'i.

**2026-09-27 — HEDEF + KESİTSEL PİYASA BAĞLAMI (v2 dalga):**
Hedef artık "bir sonraki bar" değil REST-OF-DAY (gün sonuna kadar getiri) —
v1'in gerçek backtest'teki %-43.75 kaybının nedeni hedef/strateji
uyumsuzluğuydu: model bir sonraki saati, strateji ise gün boyu pozisyonu
taşımayı hedefliyordu, sonuç saatlik churn ve saf ücret kaybıydı (bkz.
`docs/BACKTEST_AUDIT.md` §11 madde 10).

Sembol-bazlı teknik feature'lar tek başına "piyasa bugün ne yapıyor"
bilgisini taşımaz; BIST'te hisselerin çoğu aynı gün aynı yöne gittiği için
ham `return_1`/`return_5` büyük ölçüde ORTAK hareketi, sembole özgü (idiosinkratik)
bileşeni gizler. Bu yüzden, `pd.concat` sonrası ve `dropna` ÖNCESİ, aynı `ts`
damgasındaki TÜM semboller üzerinden kesitsel (cross-sectional) sütunlar
hesaplanır: piyasa ortalaması (`mkt_*`), fazlalık getiri (`ex_*`), kesitsel
sıralama (`rank_*`) ve piyasa genişliği (`mkt_breadth`).

Look-ahead YOKTUR: hepsi AYNI `ts`'de — yani o bar kapandığında zaten
bilinen — bilgilerden türetilir; `build_intraday_dataset(require_target=True)`
zaten bu `ts` için kapanışların hepsini içerir. `dropna` kesitsel sütunlar
için de çalıştığı için (canlı tahminde kısmen eksik bir gün gelirse yarım
kalmış kesit kullanılmaz).
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
    "vol_20",
    "bar_of_day",
    # Sembol-bazlı gün içi / günlük bağlam (pipeline/intraday_features.py)
    "ret_since_open",
    "gap",
    "prev_day_ret",
    "d_ret_5",
    # Kesitsel piyasa bağlamı (aşağıda `pd.concat` sonrası hesaplanır)
    "mkt_ret_1",
    "mkt_ret_since_open",
    "ex_ret_1",
    "ex_ret_since_open",
    "rank_ret_5",
    "rank_since_open",
    "mkt_breadth",
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
    """`require_target=False`: canlı tahmin gibi en sondaki (gün sonu
    kapanışı henüz gerçekleşmemiş, `target_up=NaN` olan) satırları ATMAZ.
    Eğitim yolu varsayılanı kullanmaya devam eder.

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
            "SELECT id, ticker, market FROM symbols WHERE market = 'BIST' ORDER BY ticker"
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

            merged = pdf.reset_index()
            # Intraday'de tarih değil zaman damgası anlamlı: aynı güne ait iki
            # farklı bar aynı "feature_date"ye düşerdi (günlük pipeline'daki
            # `feature_date` yerine).
            merged["feature_ts"] = merged["ts"].astype(str)
            frames.append(merged)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Kesitsel piyasa bağlamı — aynı `ts` damgasındaki TÜM semboller üzerinden
    # (bar t kapanışında bilinen bilgi, look-ahead yok). Sembol-bazlı ham
    # getirilerin ortak piyasa bileşenini ayırmak için (modül docstring'i).
    ts_g = df.groupby("ts")
    df["mkt_ret_1"] = ts_g["return_1"].transform("mean")
    df["mkt_ret_since_open"] = ts_g["ret_since_open"].transform("mean")
    df["ex_ret_1"] = df["return_1"] - df["mkt_ret_1"]
    df["ex_ret_since_open"] = df["ret_since_open"] - df["mkt_ret_since_open"]
    df["rank_ret_5"] = ts_g["return_5"].rank(pct=True)
    df["rank_since_open"] = ts_g["ret_since_open"].rank(pct=True)
    df["mkt_breadth"] = ts_g["return_1"].transform(lambda s: (s > 0).mean())

    df = df.dropna(subset=INTRADAY_FEATURE_COLUMNS)
    # Training needs a known label. Bu filtre, seri sonu satırının yanı sıra
    # HER GÜN'ün son barını da otomatik olarak eler — onların `target_up`'ı
    # `add_intraday_features` içinde zaten NaN yapıldı (gün sonu kapanışına
    # girilemez; gece sıçraması hedeflenmez).
    if require_target:
        df = df[df["target_up"].notna()]
    return df
