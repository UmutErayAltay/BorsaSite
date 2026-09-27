"""Regression tests for the intraday day-boundary discipline
(docs/BACKTEST_AUDIT.md §2, generalized to day boundaries — see
`pipeline/intraday_features.py`): a day's LAST bar must have a NaN target, not
a false label, because the model never targets the overnight gap.

The target itself is REST-OF-DAY (day close vs. this bar's close) as of
2026-09-27: the v1 "next bar" target made the hourly strategy churn against the
model (docs/BACKTEST_AUDIT.md §11 madde 10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def _zigzag_bars(days: int = 10, bars_per_day: int = 5) -> pd.DataFrame:
    """Yeterince uzun ZİG-ZAG seri (varsayılan 10 gün x 5 bar).

    Monoton seriden daha güçlü bir test zemini: yön değiştirdiği için
    `return_1`/`return_5` işaretli hem pozitif hem negatif değerler alır,
    `vol_20` sıfırdan uzaklaşır, ve `rsi_14` gerçek bir değere oturur.
    `open`, `close`'in bir bar gerisinden gelen uyarlamasıdır → `ret_since_open`
    ve `gap` testleri de anlamlı olur."""
    # Barlar 10:00-14:00 Europe/Istanbul'de — gerçek BIST seansıyla aynı
    # saatler. ÖNEMLİ: `add_intraday_features` gün tespitini
    # `index.tz_convert(None).normalize()` ile yapar, yani gün sınırı index'in
    # TZ'sinin UTC'ye çevrilmiş halinden türetilir. 03:00'ten ÖNCEKİ
    # Istanbul barları bu çevrimde bir ÖNCEKİ güne düşer. Gerçek veri
    # 10:00-18:00 olduğu için üretimde sorun yok; test serisi de aynı seans
    # saatlerini kullanmalı (mevcut `_two_day_bars` da öyle).
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-05 10:00", tz="Europe/Istanbul") + pd.Timedelta(days=d, hours=b)
            for d in range(days)
            for b in range(bars_per_day)
        ]
    )
    step = np.tile(np.array([1.0, 1.0, -0.6, -0.6, 0.8]), days)[: len(idx)]
    close = 100.0 + np.cumsum(step)
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0 + (np.arange(len(close)) % 7),
        },
        index=idx,
    )


def test_last_bar_of_each_day_has_nan_target():
    """EN ÖNEMLİ test: gün 1'in SON bar'ının hedefi NaN olmalı. Gece sıçraması
    (gün 2'nin ilk barı) aşağı yönlü olsa bile `target_up` NaN kalmalı —
    0.0 değil."""
    out = add_intraday_features(_two_day_bars(), {})

    day1_last = out.index[4]  # 2026-01-05 14:00
    assert np.isnan(out["target_up"].loc[day1_last])
    assert np.isnan(out["target_return"].loc[day1_last])
    # NaN, 0.0'a sessizce düşmemeli (günlük pipeline'daki §2 regresyonu).
    assert out["target_up"].loc[day1_last] != 0.0


def test_mid_day_bar_targets_day_close():
    """REST-OF-DAY: gün 1'in orta barının hedefi, gün 1'in SON kapanışı /
    o barın kapanışı - 1. Bir sonraki bar DEĞİL (bu, v1'den bu dalgadaki
    tek kasıtlı kırılma noktası)."""
    out = add_intraday_features(_two_day_bars(), {})

    day1_close = out["close"].iloc[4]  # gün 1'in son barı
    row = out.index[2]  # gün 1'in orta barı
    assert out["target_return"].loc[row] == day1_close / out["close"].iloc[2] - 1
    assert out["target_up"].loc[row] == 1.0  # gün 1 yükseliyor

    # Gün 1'in 0. ve 1. barı da aynı kapanışı hedefler — hedef bar'ın
    # konumuna değil, günün sonuna bağlıdır.
    for i in (0, 1):
        assert out["target_return"].iloc[i] == day1_close / out["close"].iloc[i] - 1

    # Gün 2'de yön tersine döner — hedef yine de gün 2'nin son kapanışı.
    row2 = out.index[7]  # gün 2'nin 3. barı
    day2_close = out["close"].iloc[9]
    assert out["target_up"].loc[row2] == 0.0
    assert out["target_return"].loc[row2] == day2_close / out["close"].iloc[7] - 1


def test_true_last_bar_of_series_has_nan_target():
    """Serinin GERÇEK son barı (gün 2'nin son barı) da NaN — hem 'gün
    sınırı' hem 'gün sonu kapanışı daha yok' durumu aynı `not_last_bar`
    mantığıyla kapanır."""
    out = add_intraday_features(_two_day_bars(), {})

    last = out.index[-1]
    assert np.isnan(out["target_up"].loc[last])
    assert np.isnan(out["target_return"].loc[last])


def test_first_bar_of_each_day_is_bar_zero():
    out = add_intraday_features(_two_day_bars(), {})

    assert out["bar_of_day"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert out["bar_of_day"].loc[out.index[5]] == 0  # gün 2'nin ilk barı


def test_target_never_spans_a_day_boundary():
    """Etiketli satır, günün SON barı dışındaki her bardır (serinin gerçek
    son barı da `shift(-1)` ile NaN'a düştüğü için zaten burada)."""
    out = add_intraday_features(_two_day_bars(), {})

    day = out.index.tz_convert(None).normalize()
    next_day = pd.Series(day, index=out.index).shift(-1)
    assert out["target_up"].notna().equals(next_day == day)


def test_target_equals_day_close_over_close_in_a_long_series():
    """Daha uzun, yön değiştiren seride: her etiketli satırın hedefi KENDİ
    GÜNÜNÜN son kapanışı / kendi kapanışı - 1 (yani hedef asla bir sonraki
    bara değil, asla bir sonraki güne bakmıyor)."""
    out = add_intraday_features(_zigzag_bars(), {})
    day = out.index.tz_convert(None).normalize()
    day_close = out["close"].groupby(day).transform("last")

    labeled = out["target_up"].notna()
    assert labeled.sum() == 10 * 4  # 10 gün x (5 - 1 son bar)
    assert np.allclose(
        out.loc[labeled, "target_return"],
        day_close[labeled] / out["close"][labeled] - 1,
    )
    assert np.allclose(
        out.loc[labeled, "target_up"],
        (day_close[labeled] > out["close"][labeled]).astype(float),
    )
    # Etiketli satırların HİÇBİRİ kendi gününün son barı değil.
    assert not out.loc[labeled].index.isin(out.index[out["bar_of_day"] == 4]).any()


def test_features_have_no_lookahead_when_series_is_truncated():
    """EN KRİTİK yeni test — gelecek, geçmişe SIZMAZ.

    Aynı serinin TAM hali ile, k. günden sonrası KESİLMİŞ hali üzerinde
    `add_intraday_features` çalıştırılır; kesim noktasına kadarki barlarda
    HER özellik kolonu BİREBİR aynı olmalı. `groupby(...).transform("last")`
    veya `pct_change().shift(0)` kayması bu testi kırar.

    Kapsam dışı: hedef kolonları (`target_return`/`target_up` — kesilen günün
    son kapanışı kesmede bilinmez, tanım gereği böyledir) ve `day_close`/
    `day_open` (yalnızca hedef için tutulan ara hesaplar, çıktıdan düşer).
    Kesitsel kolonlar (`ex_*`, `mkt_*`, `rank_*`, `mkt_breadth`) semboller
    arasıdır ve tek sembollü seride tanımsızdır; `pipeline/intraday_dataset.py`
    tarafında hesaplanır, o katmanın testine aittir."""
    full = _zigzag_bars()
    cut = 32  # 7. günün ortası

    full_out = add_intraday_features(full, {})
    trunc_out = add_intraday_features(full.iloc[:cut], {})
    assert trunc_out.index.equals(full_out.index[:cut])

    excluded = {"target_return", "target_up", "day_close", "day_open"}
    feature_cols = [
        c
        for c in full_out.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(full_out[c])
    ]
    # Isınma pencereleri (sma20, vol_20) ve günlük geçmişi `d_ret_5`/`gap`
    # her ikisi de birlikte kırpıldığı için NaN maskeleri de birebir aynı
    # olmalı — aksi halde kırpma bir özelliği "ısıtmış" olurdu.
    assert full_out[feature_cols].notna().iloc[:cut].equals(trunc_out[feature_cols].notna())

    head_full = full_out.iloc[:cut][feature_cols].dropna()
    head_trunc = trunc_out[feature_cols].dropna()
    pd.testing.assert_frame_equal(
        head_full, head_trunc, check_exact=False, rtol=1e-12, atol=1e-12
    )
    # Test gerçekten bir şey karşılaştırıyor olsun (hepsi NaN olup boş
    # kalmasın).
    assert len(head_trunc) > 0

    # `day_close`/`day_open` çıktıda YOK — biri gelecek bilgi, diğeri
    # yalnızca `ret_since_open` hesabının girdisi.
    assert "day_close" not in full_out.columns
    assert "day_open" not in full_out.columns


def test_gap_is_today_open_over_previous_close():
    """`gap` = bugünkü açılış / DÜNKÜ kapanış - 1. İlk gün NaN'dir (önceki
    gün yok). `d_ret_5`/`prev_day_ret` de bugünün kapanışını içermez."""
    df = _zigzag_bars(days=8)
    out = add_intraday_features(df, {})

    assert out["gap"].iloc[0:5].isna().all()  # gün 1'in önceki günü yok
    for d in range(1, 8):
        first = d * 5
        assert out["gap"].iloc[first] == pytest.approx(
            out["open"].iloc[first] / out["close"].iloc[first - 1] - 1
        )
    # Gün içi sabit: ertesi gün açılışını bekleyen bar'lar aynı `gap`'i taşır.
    assert out["gap"].iloc[5:10].nunique() == 1

    # Gün 1'de 5 günlük geçmiş yok -> ilk 5 gün NaN, 6. günden itibaren var.
    assert out["d_ret_5"].iloc[0:30].isna().all()
    assert out["d_ret_5"].iloc[30:].notna().all()
    # `prev_day_ret` = d_close / önceki d_close'nin BİR GÜN GERİSİ; iki günlük
    # geçmiş ister (pct_change'in kendi NaN'ı + shift(1)) -> 3. günden itibaren.
    assert out["prev_day_ret"].iloc[0:10].isna().all()
    assert out["prev_day_ret"].iloc[10:].notna().all()
    # Gün 3'teki `prev_day_ret` = GÜN 2'nin getirisi (kendisi değil):
    # `pct_change` zaten bir gün geriden bakıyor, `shift(1)` ise bugünün
    # kapanışını da feature'dan çıkarıyor -> iki günlük geçmiş gerekiyor.
    assert out["prev_day_ret"].iloc[10] == pytest.approx(
        out["close"].iloc[9] / out["close"].iloc[4] - 1
    )


def test_ret_since_open_is_cumulative_from_day_open():
    """`ret_since_open` = o günün açılışından beri kümülatif getiri; günün
    ilk barında tam olarak `close/open - 1` (ve bar'lar arasında sürekli)."""
    out = add_intraday_features(_zigzag_bars(days=8), {})

    day = out.index.tz_convert(None).normalize()
    day_open = out["open"].groupby(day).transform("first")
    assert np.allclose(out["ret_since_open"], out["close"] / day_open - 1)

    for d in range(8):
        first = d * 5
        assert out["ret_since_open"].iloc[first] == pytest.approx(
            out["close"].iloc[first] / out["open"].iloc[first] - 1
        )
    # Gün 1'in 2. bar'ı: gün 1'in AÇILIŞINDAN (bar 0'ın açılışı) birikmiş getiri.
    assert out["ret_since_open"].iloc[1] == pytest.approx(
        out["close"].iloc[1] / out["open"].iloc[0] - 1
    )
