"""Intraday (1h) teknik özellikler ve GÜN-İÇİ hedef.

KRİTİK TASARIM KARARI — bu modülün var olma sebebi:
Hedef SADECE gün içi bir sonraki bara bakar. GECE (seans-arası) sıçrama
hedef olarak KULLANILMAZ; bir günün SON bar'ının hedefi NaN'dir. Bu,
günlük pipeline'daki "serinin son elemanının hedefi NaN olmalı, sessizce
yanlış 0.0 olmamalı" disiplininin (bkz. `pipeline/features.py` docstring'i ve
`docs/BACKTEST_AUDIT.md` §2) gün sınırlarına genellenmesidir: burada "seri
sonu" tek bir satır değil, HER GÜN'ün son bar'ıdır.

Bunun doğal sonucu: bu model gün içi açılan pozisyon için karar verir, gece
pozisyon tutmaz. Model gece riski hakkında hiçbir fikir üretmediği için gece
tutmak sorumsuz olur — bu kuralın backtest/trading katmanında uygulanması
bu dalganın (yalnızca feature + dataset) konusu DEĞİLDİR.

Gün tespiti `index.date`'e dayanır; seans sınırları (BIST 10:00-18:00, ABD
09:30-16:00) hard-coded değildir — `prices_intraday` UTC saklar, sembolün
takvimi sembol başına değişir ve bu dalgada buna karar verilmedi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features import compute_macd, compute_rsi


def add_intraday_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """df: DatetimeIndex (tz-aware), kolonlar close, volume. Tek sembolun
    çoklu-günlük barları olabilir - gün sınırlarını index.date'ten tespit eder."""
    out = df.copy()
    close = out["close"]
    volume = out["volume"].fillna(0)

    out["rsi_14"] = compute_rsi(close, int(cfg.get("rsi_period", 14)))

    macd, sig, hist = compute_macd(
        close,
        int(cfg.get("macd_fast", 12)),
        int(cfg.get("macd_slow", 26)),
        int(cfg.get("macd_signal", 9)),
    )
    out["macd"] = macd
    out["macd_signal"] = sig
    out["macd_hist"] = hist

    sma_s = int(cfg.get("sma_short", 20))
    out["close_sma20_ratio"] = close / close.rolling(sma_s).mean() - 1

    out["return_1"] = close.pct_change(1)
    out["return_5"] = close.pct_change(5)

    vol_ma = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    # Gün tespiti: tz-aware index'i önce naive'e çevir, sonra normalize et —
    # aksi halde `normalize()` UTC gün sınırında keser ve gün içi barları
    # yanlış güne yazardı.
    out["_day"] = (
        out.index.tz_convert(None).normalize()
        if out.index.tz is not None
        else out.index.normalize()
    )
    # Günün kaçıncı bar'ı (0'dan başlar) — günlük pipeline'da KARŞILIĞI OLMAYAN
    # bir özellik: saat dilimi etkisini (açılış oynaklığı, öğlen sakinliği,
    # kapanış rampası) yakalar.
    out["bar_of_day"] = out.groupby("_day").cumcount()

    # Hedef (modül docstring'indeki tasarım kararı): yalnızca AYNI GÜN içindeki
    # bir sonraki bar. `same_day` son bar için ve gün sınırını geçen her bar
    # için False olur — NaN böylece doğru şekilde yayılır. Boolean indeksleme
    # yerine `np.where` bilinçli seçildi: `same_day` ndarray, `.loc[mask]=`
    # maskeyi yeniden hizalamak zorunda kalır ve sınır hatasına açıktır.
    next_close = close.shift(-1)
    next_day = out["_day"].shift(-1)
    same_day = next_day == out["_day"]
    out["target_return"] = np.where(same_day, next_close / close - 1, np.nan)
    out["target_up"] = np.where(same_day, (next_close > close).astype(float), np.nan)

    return out.drop(columns=["_day"])
