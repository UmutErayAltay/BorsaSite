"""Regression tests for the intraday day-boundary discipline
(docs/BACKTEST_AUDIT.md §2, generalized to day boundaries — see
`pipeline/intraday_features.py`): a day's LAST bar must have a NaN target, not
a false label, because the model never targets the overnight gap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.intraday_features import add_intraday_features


def _two_day_bars() -> pd.DataFrame:
    """Tek sembol, iki günlük, 5'er bar — Europe/Istanbul tz-aware."""
    idx = pd.date_range("2026-01-05 10:00", periods=5, freq="1h", tz="Europe/Istanbul").append(
        pd.date_range("2026-01-06 10:00", periods=5, freq="1h", tz="Europe/Istanbul")
    )
    # Gün 1 monotonik YUKARI, gün 2 monotonik AŞAĞI: gün 1'in son barı ile
    # gün 2'nin ilk barı arasındaki gece sıçraması DİREN yönde olsun ki test,
    # gün-sınırı maskesi olmadan sessizce yanlış bir etiket üretirdi.
    close = np.concatenate([np.arange(100.0, 105.0), np.arange(200.0, 195.0, -1.0)])
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000.0},
        index=idx,
    )


def test_last_bar_of_each_day_has_nan_target():
    """EN ÖNEMLİ test: gün 1'in SON bar'ının hedefi NaN olmalı. Gece sıçraması
    (gün 2'nin ilk bar'ı) aşağı yönlü olsa bile `target_up` NaN kalmalı —
    0.0 değil."""
    out = add_intraday_features(_two_day_bars(), {})

    day1_last = out.index[4]  # 2026-01-05 14:00
    assert np.isnan(out["target_up"].loc[day1_last])
    assert np.isnan(out["target_return"].loc[day1_last])
    # NaN, 0.0'a sessizce düşmemeli (günlük pipeline'daki §2 regresyonu).
    assert out["target_up"].loc[day1_last] != 0.0


def test_mid_day_bar_targets_next_bar_of_same_day():
    out = add_intraday_features(_two_day_bars(), {})

    row = out.index[2]  # gün 1'in orta barı
    expected_return = out["close"].iloc[3] / out["close"].iloc[2] - 1
    assert out["target_return"].loc[row] == expected_return
    assert out["target_up"].loc[row] == 1.0  # gün 1 yükseliyor

    # Gün 2'de yön tersine döner — hedef yine de aynı günün bir sonraki barı.
    row2 = out.index[7]  # gün 2'nin 3. barı
    assert out["target_up"].loc[row2] == 0.0
    assert out["target_return"].loc[row2] == (
        out["close"].iloc[8] / out["close"].iloc[7] - 1
    )


def test_true_last_bar_of_series_has_nan_target():
    """Serinin GERÇEK son bar'ı (gün 2'nin son bar'ı) da NaN — hem 'gün
    sınırı' hem 'hiç sonraki yok' durumu aynı `same_day` mantığıyla kapanır."""
    out = add_intraday_features(_two_day_bars(), {})

    last = out.index[-1]
    assert np.isnan(out["target_up"].loc[last])
    assert np.isnan(out["target_return"].loc[last])


def test_first_bar_of_each_day_is_bar_zero():
    out = add_intraday_features(_two_day_bars(), {})

    assert out["bar_of_day"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert out["bar_of_day"].loc[out.index[5]] == 0  # gün 2'nin ilk barı


def test_target_never_spans_a_day_boundary():
    """Hiçbir etiketli satırın hedefi bir sonraki GÜNün barına bakamaz."""
    out = add_intraday_features(_two_day_bars(), {})

    day = out.index.tz_convert(None).normalize()
    next_day = pd.Series(day, index=out.index).shift(-1)
    assert out["target_up"].notna().equals(next_day == day)
