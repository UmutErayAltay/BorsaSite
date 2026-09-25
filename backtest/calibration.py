"""Olasılık kalibrasyonu — SADECE validation setinde fit edilir, OOS asla
kalibratörün fit'inde kullanılmaz (bkz. backtest/walk_forward.py)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class ProbabilityCalibrator:
    method: str = "none"
    _model: object | None = None

    def fit(self, probabilities, y_true) -> "ProbabilityCalibrator":
        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(y_true, dtype=int)
        if len(p) != len(y) or len(p) < 2:
            raise ValueError("Calibration verisi yetersiz.")
        if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Olasılıklar 0..1 aralığında olmalı.")
        if len(np.unique(y)) < 2:
            raise ValueError("Calibration için iki sınıf gerekir.")
        if self.method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError("method: none, sigmoid veya isotonic olmalı.")

        if self.method == "sigmoid":
            eps = 1e-6
            logits = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
            self._model = LogisticRegression(random_state=42).fit(logits, y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Olasılıklar 0..1 aralığında olmalı.")
        if self.method == "none":
            return p.copy()
        if self._model is None:
            raise RuntimeError("Calibrator önce fit edilmelidir.")
        if self.method == "sigmoid":
            eps = 1e-6
            logits = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
            return self._model.predict_proba(logits)[:, 1]
        return np.asarray(self._model.predict(p), dtype=float)

    def evaluate(self, probabilities, y_true) -> dict[str, float]:
        before = np.asarray(probabilities, dtype=float)
        after = self.transform(before)
        y = np.asarray(y_true, dtype=int)
        return {
            "brier_before": float(brier_score_loss(y, before)),
            "brier_after": float(brier_score_loss(y, after)),
        }
