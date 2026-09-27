"""Intraday (1h) XGBoost model eğitimi — `pipeline/train_model.py`'nin
gün-içi karşılığı.

Günlük pipeline'dan bilinçli olarak AYRI tutulur: config, model dosyası,
metrics dosyası ve `model_experiments` tablosundaki `model_version` string'i
hep farklıdır — iki model birbirinin çıktısını asla ezmez. Tablo şeması
DEĞİŞMEZ, sadece satır içeriği ayrışır.

Hedef tanımı ve gün-sinırı disiplini `pipeline/intraday_features.py`
modül docstring'inde anlatılır; burada tekrar etmeye gerek yoktur.

**v2 (2026-09-27) — GÜN BAZLI ÜÇ PARÇALI AYRIM.** v1 barları `feature_ts`
sırasına göre ikiye bölüyordu; aynı günün barları iki parçaya ayrılabiliyordu
ve model/strateji kararı "bu gün" kavramını hiç bilmiyordu. Artık train/val/test
GÜNler üzerinden kesiliyor: aynı günün barları asla iki parçaya bölünmez. Model
SADECE train'de eğitilir; val yalnızca `entry_quantile` eşiğini türetmek ve
erken durdurmak (eval_set) için kullanılır — val ve test SADE SEÇİM YAPILMAZ.
`entry_threshold` = regresorun VALIDATION tahminlerinin `entry_quantile`
kantile; `backtest/intraday_engine.py` giriş kararını bu eşikten verir.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

from pipeline.db import get_connection, init_schema, insert_model_experiment
from pipeline.intraday_dataset import (
    INTRADAY_FEATURE_COLUMNS,
    build_intraday_dataset,
    load_intraday_model_config,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _day_based_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gün bazlı kronolojik train/val/test ayrımı.

    `ts` TIMESTAMPTZ olduğu için gün tespiti İstanbul yereline çevrilerek
    yapılır (UTC gün sınırında kesmek yanlış güne yazardı — aynı dönüşüm
    `pipeline/intraday_features.py` ve `backtest/intraday_engine.py`'de de
    var). Sıralama anahtarı BAR değil GÜN: böylece aynı güne ait barlar
    asla iki parçaya bölünmez."""
    days = sorted(
        df["ts"].dt.tz_convert("Europe/Istanbul").dt.date.unique()
    )
    n_days = len(days)
    n_test = int(round(n_days * test_ratio))
    n_val = int(round(n_days * val_ratio))
    # Üç parça da en az bir gün istiyor; küçük veri kümelerinde oranlar
    # yuvarlama ile 0'a düşebilir.
    if n_days < 3:
        raise RuntimeError(
            f"Üç parçalı ayrım için en az 3 gün gerekir, veride {n_days} gün var."
        )
    n_test = max(1, min(n_test, n_days - 2))
    n_val = max(1, min(n_val, n_days - n_test - 1))

    train_days = days[: n_days - n_val - n_test]
    val_days = days[n_days - n_val - n_test : n_days - n_test]
    test_days = days[n_days - n_test :]

    day_col = df["ts"].dt.tz_convert("Europe/Istanbul").dt.date
    return (
        df[day_col.isin(train_days)],
        df[day_col.isin(val_days)],
        df[day_col.isin(test_days)],
    )


def _information_coefficient(pred: np.ndarray, actual: np.ndarray) -> float | None:
    """Regresyonun bilgi katsayısı: tahmin ile gerçek getirinin Pearson
    korelasyonu. Sınıflandırma metriklerinden FARKLIDIR — modelin
    yönü değil, getiri BÜYÜKLÜĞÜNü ne kadar iyi sıraladığını ölçer, ki
    strateji kararı (`pred_rod > entry_threshold`) tam olarak bunu kullanır.
    Sabit tahmin veya tek örnek korelasyon tanımsızdır -> None."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if pred.size < 2 or np.std(pred) == 0 or np.std(actual) == 0:
        return None
    ic = float(np.corrcoef(pred, actual)[0, 1])
    return None if np.isnan(ic) else ic


def run(save_metrics: bool = True) -> dict:
    cfg = load_intraday_model_config()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    model_path = PROJECT_ROOT / model_cfg["path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_intraday_dataset()
    if df.empty or len(df) < 100:
        raise RuntimeError(
            f"Yetersiz veri: {len(df)} satır. Önce intraday fiyat çekin: "
            f"python scripts/run_fetch_intraday_prices.py --interval 1h"
        )

    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    test_ratio = float(train_cfg.get("test_ratio", 0.2))
    train_df, val_df, test_df = _day_based_split(df, val_ratio, test_ratio)

    X_train = train_df[INTRADAY_FEATURE_COLUMNS]
    y_train = train_df["target_up"].astype(int)
    y_train_ret = train_df["target_return"]  # kesintisiz, .astype(int) YOK
    X_val = val_df[INTRADAY_FEATURE_COLUMNS]
    y_val = val_df["target_up"].astype(int)
    y_val_ret = val_df["target_return"]
    X_test = test_df[INTRADAY_FEATURE_COLUMNS]
    y_test = test_df["target_up"].astype(int)
    y_test_ret = test_df["target_return"]

    xgb_params = dict(train_cfg.get("xgb", {}))
    xgb_params.setdefault("objective", "binary:logistic")
    xgb_params.setdefault("random_state", int(train_cfg.get("random_state", 42)))

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Aynı ayrım, hedef `target_return` (gün sonuna kadar getiri büyüklüğü).
    reg_params = dict(train_cfg.get("xgb_regressor", {}))
    reg_params.setdefault("random_state", int(train_cfg.get("random_state", 42)))

    reg_model = xgb.XGBRegressor(**reg_params)
    reg_model.fit(
        X_train,
        y_train_ret,
        eval_set=[(X_val, y_val_ret)],
        verbose=False,
    )

    # Giriş eşiği: regresorun VALIDATION tahminlerinin kantili. TEST'E
    # BAKILMAZ — eşik seçimi ve backtest, aynı gün görülmemiş veriyle
    # yapılan dürüst bir seçimdir (bkz. docs/BACKTEST_AUDIT.md §11 madde 10).
    entry_quantile = float(cfg["strategy"]["entry_quantile"])
    val_pred_ret = reg_model.predict(X_val)
    entry_threshold = float(np.quantile(val_pred_ret, entry_quantile))

    val_start: date = min(val_df["ts"].dt.tz_convert("Europe/Istanbul").dt.date)
    test_start: date = min(test_df["ts"].dt.tz_convert("Europe/Istanbul").dt.date)

    joblib.dump(
        {
            "model": model,
            "magnitude_model": reg_model,
            "features": INTRADAY_FEATURE_COLUMNS,
            "version": model_cfg.get("version", "2.0"),
            "entry_threshold": entry_threshold,
            "val_start": val_start.isoformat(),
            "test_start": test_start.isoformat(),
        },
        model_path,
    )
    logger.info("Intraday model kaydedildi: %s", model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= float(cfg["prediction"].get("direction_threshold", 0.5))).astype(int)
    y_pred_ret = reg_model.predict(X_test)

    metrics = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "symbols": int(df["symbol_id"].nunique()),
        "val_start": val_start.isoformat(),
        "test_start": test_start.isoformat(),
        "entry_threshold": entry_threshold,
    }
    val_ic = _information_coefficient(val_pred_ret, y_val_ret)
    test_ic = _information_coefficient(y_pred_ret, y_test_ret)
    metrics["val_ic"] = None if val_ic is None else round(val_ic, 6)
    metrics["test_ic"] = None if test_ic is None else round(test_ic, 6)

    test_accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    metrics["test_accuracy"] = test_accuracy
    # `accuracy` anahtarı model_experiments tablosunun şeması ve
    # `pipeline/predict_model.py` uyumluluğu için AYNI değeri taşır.
    metrics["accuracy"] = test_accuracy
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
    except ValueError:
        metrics["roc_auc"] = None
    try:
        val_prob = model.predict_proba(X_val)[:, 1]
        metrics["val_auc"] = round(float(roc_auc_score(y_val, val_prob)), 4)
    except ValueError:
        metrics["val_auc"] = None
    metrics["mae"] = round(float(mean_absolute_error(y_test_ret, y_pred_ret)), 6)
    try:
        metrics["r2"] = round(float(r2_score(y_test_ret, y_pred_ret)), 6)
    except ValueError:
        metrics["r2"] = None

    # Özellik önemliliği: artık REGRESORDEN gelir — strateji kararlarını
    # `pred_rod > entry_threshold` veren model odur (sınıflandırıcı yalnızca
    # karar logunun prob_up alanını besler). xgboost varsayılan "gain"
    # importance'ı, feature_importances_ INTRADAY_FEATURE_COLUMNS ile aynı
    # sırada ve uzunlukta.
    metrics["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in sorted(
            zip(INTRADAY_FEATURE_COLUMNS, reg_model.feature_importances_),
            key=lambda pair: -pair[1],
        )
    }

    # Deney takibi: günlük deneylerle karışmasın diye `model_version`
    # "intraday-" önekiyle ayrılır (tablo şeması aynı kalır).
    with get_connection() as conn:
        init_schema(conn)
        insert_model_experiment(
            conn,
            model_version=f"intraday-{model_cfg.get('version', '2.0')}",
            train_rows=metrics["train_rows"],
            test_rows=metrics["test_rows"],
            symbols=metrics["symbols"],
            accuracy=metrics["accuracy"],
            roc_auc=metrics.get("roc_auc"),
            mae=metrics.get("mae"),
            r2=metrics.get("r2"),
            params={"xgb": xgb_params, "xgb_regressor": reg_params},
        )

    if save_metrics:
        # Günlük `metrics.json`'un ÜZERİNE yazmıyoruz — iki ayrı model, iki
        # ayrı metrics dosyası.
        metrics_path = model_path.parent / f"{model_path.stem}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run())
