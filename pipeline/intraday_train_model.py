"""Intraday (1h) XGBoost model eğitimi — `pipeline/train_model.py`'nin
gün-içi karşılığı.

Günlük pipeline'dan bilinçli olarak AYRI tutulur: config, model dosyası,
metrics dosyası ve `model_experiments` tablosundaki `model_version` string'i
hep farklıdır — iki model birbirinin çıktısını asla ezmez. Tablo şeması
DEĞİŞMEZ, sadece satır içeriği ayrışır.

Hedef tanımı ve gün-sınırı disiplini `pipeline/intraday_features.py`
modül docstring'inde anlatılır; burada tekrar etmeye gerek yoktur.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
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


def _temporal_split(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Günlük pipeline'ın kopyası — tek fark: sıralama anahtarı `feature_date`
    değil `feature_ts`'tir (aynı güne ait birden fazla bar olabilir)."""
    timestamps = sorted(df["feature_ts"].unique())
    split_idx = int(len(timestamps) * (1 - test_ratio))
    if split_idx < 1 or split_idx >= len(timestamps):
        split_idx = max(1, len(timestamps) - 1)
    train_cutoff = timestamps[split_idx - 1]
    train = df[df["feature_ts"] <= train_cutoff]
    test = df[df["feature_ts"] > train_cutoff]
    return train, test


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

    train_df, test_df = _temporal_split(df, float(train_cfg.get("test_ratio", 0.2)))
    X_train = train_df[INTRADAY_FEATURE_COLUMNS]
    y_train = train_df["target_up"].astype(int)
    y_train_ret = train_df["target_return"]  # kesintisiz, .astype(int) YOK
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
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Aynı temporal split, hedef `target_return` (kesintisiz getiri büyüklüğü).
    reg_params = dict(train_cfg.get("xgb_regressor", {}))
    reg_params.setdefault("random_state", int(train_cfg.get("random_state", 42)))

    reg_model = xgb.XGBRegressor(**reg_params)
    reg_model.fit(
        X_train,
        y_train_ret,
        eval_set=[(X_test, y_test_ret)],
        verbose=False,
    )

    joblib.dump(
        {
            "model": model,
            "magnitude_model": reg_model,
            "features": INTRADAY_FEATURE_COLUMNS,
            "version": model_cfg.get("version", "1.0"),
        },
        model_path,
    )
    logger.info("Intraday model kaydedildi: %s", model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= float(cfg["prediction"].get("direction_threshold", 0.5))).astype(int)
    y_pred_ret = reg_model.predict(X_test)

    metrics = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "symbols": int(df["symbol_id"].nunique()),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
    except ValueError:
        metrics["roc_auc"] = None
    metrics["mae"] = round(float(mean_absolute_error(y_test_ret, y_pred_ret)), 6)
    try:
        metrics["r2"] = round(float(r2_score(y_test_ret, y_pred_ret)), 6)
    except ValueError:
        metrics["r2"] = None

    # Özellik önemliliği: gerçekten eğitilmiş sınıflandırıcıdan gelir
    # (xgboost varsayılan "gain" importance'ı), feature_importances_
    # INTRADAY_FEATURE_COLUMNS ile aynı sırada ve uzunlukta.
    metrics["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in sorted(
            zip(INTRADAY_FEATURE_COLUMNS, model.feature_importances_),
            key=lambda pair: -pair[1],
        )
    }

    # Deney takibi: günlük deneylerle karışmasın diye `model_version`
    # "intraday-" önekiyle ayrılır (tablo şeması aynı kalır).
    with get_connection() as conn:
        init_schema(conn)
        insert_model_experiment(
            conn,
            model_version=f"intraday-{model_cfg.get('version', '1.0')}",
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
