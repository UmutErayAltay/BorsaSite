"""Tek bir XGBoost model oluşturma noktası — `backtest/walk_forward.py`'nin
her pencerede fit ettiği model burada kurulur, parametreler hep
`config/model.yaml::training`'den gelir."""
from __future__ import annotations

from typing import Any

import xgboost as xgb


def build_xgb_model(training_cfg: dict[str, Any], seed: int) -> xgb.XGBClassifier:
    params = dict(training_cfg.get("xgb", {}))
    params.setdefault("objective", "binary:logistic")
    params.setdefault("random_state", seed)
    return xgb.XGBClassifier(**params)
