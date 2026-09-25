"""Teknik analiz göstergeleri (pandas)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_technical_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """df: date index, columns open, close, volume."""
    out = df.copy()
    close = out["close"]
    volume = out["volume"].fillna(0)

    rsi_p = int(cfg.get("rsi_period", 14))
    out["rsi_14"] = compute_rsi(close, rsi_p)

    macd, sig, hist = compute_macd(
        close,
        int(cfg.get("macd_fast", 12)),
        int(cfg.get("macd_slow", 26)),
        int(cfg.get("macd_signal", 9)),
    )
    # Fiyata göre ölçeklenir — ham TRY biriminde havuzlanmış bir modelde 2 TL'lik
    # ve 500 TL'lik hisseyi karşılaştırılamaz kılıyordu (bkz. plan: Faz 1).
    out["macd"] = macd / close
    out["macd_signal"] = sig / close
    out["macd_hist"] = hist / close

    sma_s = int(cfg.get("sma_short", 20))
    sma_l = int(cfg.get("sma_long", 50))
    sma20 = close.rolling(sma_s).mean()
    sma50 = close.rolling(sma_l).mean()
    out["close_sma20_ratio"] = close / sma20 - 1
    out["close_sma50_ratio"] = close / sma50 - 1

    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)

    vol_ma = volume.rolling(20).mean()
    out["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    # Etiket artık gerçek trade'i yansıtıyor: giriş D+1 açılışında (execution_
    # delay_days=1'e uygun), çıkış D+h kapanışında — eskiden D+1 kapanışına göre
    # etiketlenip D+1 açılışında alınıp 10 güne kadar tutuluyordu (uyumsuzluk,
    # bkz. plan: Faz 1 madde 1). Mutlak yön yerine BINARY etiket dataset.py'de,
    # tüm semboller birleştikten sonra, o günün BIST medyanına göre kurulur —
    # burada sadece piyasa-geneli hareketten arındırılmamış ham forward return
    # üretilir.
    h = int(cfg.get("horizon", 5))
    entry_price = out["open"].shift(-1)
    exit_price = close.shift(-h)
    out["forward_return"] = exit_price / entry_price - 1
    target_date = out.index.to_series().shift(-h)
    out["target_date"] = target_date.dt.strftime("%Y-%m-%d")

    return out
