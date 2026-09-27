"""Intraday (1h) teknik özellikler ve GÜN-SONU hedefi.

KRİTİK TASARIM KARARI — bu modülün var olma sebebi:
Hedef GECE (seans-arası) sıçramayı KULLANMAZ; bir günün SON bar'ının hedefi
NaN'dir. Bu, günlük pipeline'daki "serinin son elemanının hedefi NaN olmalı,
sessizce yanlış 0.0 olmamalı" disiplininin (bkz. `pipeline/features.py`
docstring'i ve `docs/BACKTEST_AUDIT.md` §2) gün sınırlarına genellenmesidir:
burada "seri sonu" tek bir satır değil, HER GÜN'ün son bar'ıdır.

Bunun doğal sonucu: bu model gün içi açılan pozisyon için karar verir, gece
pozisyon tutmaz. Model gece riski hakkında hiçbir fikir üretmediği için gece
tutmak sorumsuz olur — bu kuralın backtest/trading katmanında uygulanması
bu dalganın (yalnızca feature + dataset) konusu DEĞİLDİR.

**HEDEF: REST-OF-DAY (2026-09-27, v2).** Hedef artık "bir sonraki bar" değil,
"gün sonuna kadar getiri"dir: `day_close / close - 1`. NEDEN? Gerçek
backtest'te v1 model %-43.75 kaybetti ve teshis netti: BRÜT kâr POZİTİF
(+4.305 TL), kaybın tamamı ücretten (8.680 TL) — 868 işlemin 552'si yalnızca
1 saat tutulmuş (bkz. `docs/BACKTEST_AUDIT.md` §11 madde 10). Sebep bir
hedef/strateji UYUMSUZLUĞUYDU: model "bir sonraki saat" tahmin ediyordu,
strateji ise her saat fikir değiştirip 10 TL ücret ödüyordu. Model hedefi
artık stratejinin ZATEN gerçekleştirdiği şeyi (gün boyunca pozisyonu taşı)
tahmin eder; gün sonu kapanışına kadar bekleyen pozisyon artık modele
"doğru" görünür, saatlik churn ise yanlış sinyal olarak cezalandırılır.

Günün SON bar'ının hedefi yine de NaN'dır — iki sebepten: orada girilirse
"anında gün sonu kapanışı" elde edilebilir, yani getiri saf ücret kaybıdır;
ve model o bari hiç görmemelidir. `day_close` bir FEATURE DEĞİLDİR (gelecek
bilgi), yalnızca hedefin hesabı içindir ve çıktıdan düşürülür.

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

    out["vol_20"] = out["return_1"].rolling(20).std()

    # Gün tespiti: tz-aware index'i önce naive'e çevir, sonra normalize et —
    # aksi halde `normalize()` UTC gün sınırında keser ve gün içi barları
    # yanlış güne yazardı.
    out["_day"] = (
        out.index.tz_convert(None).normalize()
        if out.index.tz is not None
        else out.index.normalize()
    )
    day_group = out.groupby("_day")
    # Günün kaçıncı bar'ı (0'dan başlar) — günlük pipeline'da KARŞILIĞI OLMAYAN
    # bir özellik: saat dilimi etkisini (açılış oynaklığı, öğlen sakinliği,
    # kapanış rampası) yakalar. Günün toplam bar sayısı (`bars_left`) bilinçli
    # olarak EKLENMEZ: yarım günlerde (bayram arifesi) sonradan hesaplanan
    # "bugün kaç bar kaldı" o günü sonradan haber verirdi.
    out["bar_of_day"] = day_group.cumcount()

    # Gün içi kümülatif getiri — "bugün şu ana kadar ne yaptı", rest-of-day
    # hedefinin doğal eşleniği (günün geri kalanı bu farkla döner).
    day_open = day_group["open"].transform("first")
    out["ret_since_open"] = out["close"] / day_open - 1

    # Günlük agregasyon -> barlara yayma. `.shift(1)` KRİTİK: bugünün
    # kapanışı bugünün feature'ına SIZMAMALI (aksi halde look-ahead).
    daily = out.groupby("_day").agg(d_open=("open", "first"), d_close=("close", "last"))
    daily["gap"] = daily["d_open"] / daily["d_close"].shift(1) - 1
    daily["prev_day_ret"] = daily["d_close"].pct_change().shift(1)
    daily["d_ret_5"] = daily["d_close"].pct_change(5).shift(1)
    out["gap"] = out["_day"].map(daily["gap"])
    out["prev_day_ret"] = out["_day"].map(daily["prev_day_ret"])
    out["d_ret_5"] = out["_day"].map(daily["d_ret_5"])

    # Hedef (modül docstring): GÜN SONU — günün son kapanışına kadar getiri.
    # `not_last_bar` son bar için (ve dolayısıyla serinin son barı için) False
    # olur — NaN böylece doğru şekilde yayılır. Boolean indeksleme yerine
    # `np.where` bilinçli seçildi: maske ndarray, `.loc[mask]=` maskeyi yeniden
    # hizalamak zorunda kalır ve sınır hatasına açıktır.
    day_close = day_group["close"].transform("last")
    next_day = out["_day"].shift(-1)
    not_last_bar = next_day == out["_day"]
    out["target_return"] = np.where(not_last_bar, day_close / close - 1, np.nan)
    out["target_up"] = np.where(not_last_bar, (day_close > close).astype(float), np.nan)

    # `day_open`/`day_close` yalnızca yerel ara hesaplardır (biri gelecek
    # bilgi, diğeri `ret_since_open`'ın girdisi) — hiçbir zaman kolon
    # olmadıkları için çıktıdan düşürülecek bir şey yoktur.
    return out.drop(columns=["_day"])
