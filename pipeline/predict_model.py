"""Eğitilmiş model ile güncel tahminler."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from pipeline.dataset import FEATURE_COLUMNS, build_dataset, load_model_config
from pipeline.db import get_connection, init_schema, upsert_prediction

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_trained_model():
    cfg = load_model_config()
    path = PROJECT_ROOT / cfg["model"]["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {path}\nÖnce: python scripts/run_train.py"
        )
    return joblib.load(path), cfg


def run(latest_only: bool = True) -> dict:
    bundle, cfg = load_trained_model()
    model = bundle["model"]
    version = bundle.get("version", cfg["model"].get("version", "1.0"))
    threshold = float(cfg["prediction"].get("direction_threshold", 0.5))

    df = build_dataset(require_target=False)
    if df.empty:
        raise RuntimeError("Tahmin için veri yok.")

    if latest_only:
        df = (
            df.sort_values("feature_date")
            .groupby("symbol_id", as_index=False)
            .tail(1)
        )

    X = df[FEATURE_COLUMNS]
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["prob_up"] = probs
    df["predicted_up"] = (probs >= threshold).astype(int)

    stats = {"predictions": 0, "avg_prob_up": round(float(probs.mean()), 4)}

    with get_connection() as conn:
        init_schema(conn)
        for _, row in df.iterrows():
            upsert_prediction(
                conn,
                symbol_id=int(row["symbol_id"]),
                feature_date=str(row["feature_date"]),
                target_date=str(row["target_date"]) if pd.notna(row.get("target_date")) else None,
                prob_up=float(row["prob_up"]),
                predicted_up=int(row["predicted_up"]),
                model_version=version,
            )
            stats["predictions"] += 1

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run())
