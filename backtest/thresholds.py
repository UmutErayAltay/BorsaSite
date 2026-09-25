"""Alım eşiği seçimi — SADECE validation etiketlerine karşı, OOS etiketleri
asla eşik seçiminde kullanılmaz (bkz. backtest/walk_forward.py)."""
from __future__ import annotations

from typing import Any

import numpy as np


def select_buy_threshold(
    probabilities,
    y_true,
    thresholds=None,
    min_trades: int = 1,
) -> dict[str, Any]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("Threshold verisi boş veya olasılık/etiket uzunlukları farklı.")
    if thresholds is None:
        thresholds = np.arange(0.50, 0.801, 0.01)

    base_rate = float(y.mean())
    candidates: list[tuple[float, float, float, float, int]] = []
    for t in thresholds:
        mask = p >= float(t)
        n = int(mask.sum())
        if n < min_trades:
            continue
        precision = float(y[mask].mean())
        # (precision üstünlüğü, precision, -eşik) sıralaması: eşit precision'da
        # daha düşük (daha çok işlem yapan) eşik tercih edilir.
        candidates.append((precision - base_rate, precision, -float(t), float(t), n))

    if not candidates:
        raise ValueError("Minimum işlem sayısını sağlayan bir threshold bulunamadı.")

    _, precision, _, threshold, trade_count = max(candidates)
    return {
        "threshold": threshold,
        "validation_precision": precision,
        "base_rate": base_rate,
        "trade_count": trade_count,
    }
