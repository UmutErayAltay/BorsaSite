"""Intraday v2 için WALK-FORWARD (genişleyen pencere) backtest.

Tek seferlik `backtest/intraday_engine.py` backtest'i dürüst DEĞİLDİR:
model tüm veriyle eğitilir, eşik val'den türetilir, sonra o model tüm
geçmişte oynatılır — yani model görmediği bir dönem yoktur. Bu modül
GERÇEK out-of-sample zinciri kurar:

    fold k:  train = gün[0 : t0]                    (GENİŞLEYEN, ilk günden)
             val   = gün[t0 : t0+val_days]
             test  = gün[t0+val_days : ...+test_days]

`test` pencereleri ARDIŞIK ve ÇAKIŞMASIZdır, dolayısıyla bir fold'un
val+test'i bir sonraki fold'un train'ine GİRER — pencereler birbirini
besler, veri tekrar tekrar kullanılmaz ve tek bir sürekli portföy kurulur.

SIZDIRMAZLIĞIN ÜÇ KURALI:

1. Model yalnızca `train`'de eğitilir; `val` yalnızca eğitim içi
   `eval_set` (erken durdurma) ve `entry_threshold` üretimi içindir.
2. `entry_threshold` = regresörün **validation** tahminlerinin kantili
   (`entry_quantile`). TEST'E BAKILMAZ — aynı gerekçe
   `pipeline/intraday_train_model.py` ve docs/BACKTEST_AUDIT.md §11/10.
3. Bir fold'un test satırları OYNATILMAZ, yalnızca model çıktısı ve eşiği
   taşır; oynatma yalnızca tüm test satırları birleştikten sonra TEK bir
   `simulate_intraday` çağrısında yapılır.

Ayrıntılı gerekçeler modül docstring'lerinde; burada tek istisna notu:
`_day` dönüşümü `backtest/intraday_engine.py` ile BİREBİR aynıdır
(`ts.dt.tz_convert(None).dt.normalize()`), çünkü fold sınırları ile
oynatılan günler aynı gün tanımını paylaşmalıdır — biri İstanbul
gece yarısında, diğeri UTC'de keserse barlar iki farklı güne dağılır ve
portföy izdüşümü bozulur.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backtest.costs import BacktestCostConfig
from backtest.engine import BacktestResult
from backtest.intraday_engine import simulate_intraday
from pipeline.intraday_dataset import (
    INTRADAY_FEATURE_COLUMNS,
    build_intraday_dataset,
    load_intraday_model_config,
)
from pipeline.intraday_train_model import fit_intraday_models
from trading.config import TradingConfig, load_trading_config

logger = logging.getLogger(__name__)


def _normalize_days(df: pd.DataFrame) -> pd.Series:
    """Gün etiketi — `backtest/intraday_engine.py` ile AYNI dönüşüm."""
    return df["ts"].dt.tz_convert(None).dt.normalize()


def _spearman_ic(pred: pd.Series, actual: pd.Series) -> float | None:
    """Fold'un test dönemi bilgi katsayısı: `pred_rod` ile `target_return`
    arasındaki SPEARMAN sıra korelasyonu.

    Sıra korelasyonu Pearson'dan daha uygun: eşik kararı tahminin
    MUTLAK değerine değil, barlar arası SIRALAMASINA bakar. Sabit tahmin
    veya tek örnek korelasyon tanımsızdır -> `None` (aynı söz
    `pipeline/intraday_train_model.py::_information_coefficient`)."""
    if len(pred) < 2:
        return None
    corr = spearmanr(pred.to_numpy(dtype=float), actual.to_numpy(dtype=float)).statistic
    return None if np.isnan(corr) else float(corr)


def run_intraday_walk_forward(
    trading_cfg: TradingConfig | None = None,
    cost_cfg: BacktestCostConfig | None = None,
    initial_train_days: int = 250,
    val_days: int = 60,
    test_days: int = 60,
    entry_quantile: float | None = None,
) -> tuple[BacktestResult, list[dict[str, Any]]]:
    """Walk-forward backtest'i çalıştırır.

    Döner: `(BacktestResult, fold_ozetleri)`. Sonuç, TÜM fold'ların test
    satırlarının tek bir portföyde oynatılmasıyla oluşur — bakiye fold'lar
    arasında ZİNCİRLENİR, dolayısıyla her fold kendi başına değil, gerçek
    sermaye dağılımıyla değerlendirilir.

    `entry_quantile=None` ise config'deki `strategy.entry_quantile` kullanılır
    (0.998). Test dönemi küçük olduğunda bu kantil neredeyse hiç giriş
    üretmeyebilir; fold eşiği kendi VALIDATION tahminlerinden türer, yani
    val penceresi küçüldükçe eşik küçülür ve orantılı sayıda giriş bırakır
    (test'e bakılmadığı için bu bir kusur değil, val tarafındaki örneklem
    etkisinin doğal sonucudur).

    Veri yetersizse (tek bir test penceresi dolmuyor) fold'lar boş döner ve
    `BacktestResult` boştur — çağıran taraf "yeterli veri yok" durumunu
    ayrıca ele alabilsin diye exception fırlatmıyoruz."""
    trading_cfg = trading_cfg or load_trading_config()
    cost_cfg = cost_cfg or BacktestCostConfig()
    cfg = load_intraday_model_config()
    quantile = float(
        entry_quantile if entry_quantile is not None else cfg["strategy"]["entry_quantile"]
    )

    # Dataset BİR KEZ kurulur: her fold için yeniden hesaplamak, rolling
    # feature'ların ısınma penceresini kaydırıp aynı güne farklı barlar
    # yazmasına yol açardı.
    df = build_intraday_dataset(require_target=False)
    if df.empty:
        return BacktestResult([], [], [], None, None), []

    df = df.copy()
    df["_day"] = _normalize_days(df)
    # Sıralı benzersiz günler — fold sınırları DİZİ İNDİSİ üzerinden
    # kurulur, tarih karşılaştırmasıyla değil.
    days = sorted(df["_day"].unique())

    day_index = pd.Series(np.arange(len(days)), index=pd.DatetimeIndex(days))
    # Her satırın gün indisini `df`'in kendi indeksine hizalar; sıralama
    # korunur ama maske `isin` ile değil `between` ile kurulabilir.
    row_day_pos = day_index.reindex(pd.DatetimeIndex(df["_day"])).to_numpy()

    sim_parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []

    for k, t0 in enumerate(range(initial_train_days, len(days), test_days)):
        val_end = t0 + val_days
        # Son fold'da eldeki kadar gün test edilir; test penceresi boşsa dur.
        test_end = min(val_end + test_days, len(days))
        if test_end <= val_end:
            break

        train_pos = row_day_pos < t0
        val_pos = (row_day_pos >= t0) & (row_day_pos < val_end)
        test_pos = (row_day_pos >= val_end) & (row_day_pos < test_end)

        train_df = df[train_pos]
        val_df = df[val_pos]
        test_df = df[test_pos]

        # `require_target` mantığı: eğitim/val yalnızca ETİKETİ BİLİNEN
        # satırlar (her günün son barının hedefi NaN'dır), test ise TÜM
        # satırlar — son bar da karar verebilir, sadece dolum yoktur.
        train_df = train_df[train_df["target_return"].notna()]
        val_df = val_df[val_df["target_return"].notna()]
        if train_df.empty or val_df.empty or test_df.empty:
            logger.warning("Fold %d atlandı: eğitim/val/test satırı yetersiz", k)
            continue

        classifier, regressor = fit_intraday_models(train_df, val_df, cfg)

        val_pred_ret = regressor.predict(val_df[INTRADAY_FEATURE_COLUMNS])
        # Eşik YALNIZCA val'den; test'e bakılmıyor.
        entry_threshold = float(np.quantile(val_pred_ret, quantile))

        test_df = test_df.copy()
        test_df["pred_rod"] = regressor.predict(test_df[INTRADAY_FEATURE_COLUMNS])
        test_df["prob_up"] = classifier.predict_proba(test_df[INTRADAY_FEATURE_COLUMNS])[:, 1]
        test_df["entry_threshold"] = entry_threshold
        sim_parts.append(test_df)

        scored = test_df[test_df["target_return"].notna()]
        folds.append(
            {
                "fold": k,
                "train_start": days[0].date().isoformat(),
                "train_end": days[t0 - 1].date().isoformat(),
                "val_start": days[t0].date().isoformat(),
                "val_end": days[val_end - 1].date().isoformat(),
                "test_start": days[val_end].date().isoformat(),
                "test_end": days[test_end - 1].date().isoformat(),
                "n_train_rows": int(len(train_df)),
                "entry_threshold": entry_threshold,
                "test_ic": _spearman_ic(scored["pred_rod"], scored["target_return"]),
            }
        )
        logger.info(
            "Fold %d: train %s..%s (%d satır), val %d gün, test %s..%s, "
            "eşik=%.6f, test_ic=%s",
            k,
            folds[-1]["train_start"],
            folds[-1]["train_end"],
            len(train_df),
            val_days,
            folds[-1]["test_start"],
            folds[-1]["test_end"],
            entry_threshold,
            "yok" if folds[-1]["test_ic"] is None else f"{folds[-1]['test_ic']:.4f}",
        )

    if not sim_parts:
        return BacktestResult([], [], [], None, None), []

    # Fold'lar ARDIŞIK olduğu için test satırları zaten kronolojik; sıralama
    # yine de açıkça verilir (maskeler kaynak sırasını korur, `sort_values`
    # bunu garantiler).
    combined = pd.concat(sim_parts, axis=0).sort_values("ts", kind="stable")
    return simulate_intraday(combined, trading_cfg, cost_cfg), folds
