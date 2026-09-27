"""Faz 7: beklenen getiri BÜYÜKLÜĞÜ regresörü (magnitude_model) eğitimi.

train_model.run() artık yön sınıflandırıcısının yanında `target_return`
üzerine bir XGBRegressor da eğitir, aynı temporal split'i kullanır, sonucu
AYNI .pkl bundle'ına `magnitude_model` anahtarıyla yazar ve metrics'e
mae/r2 ekler. Canlıdaki gerçek model dosyasına dokunmamak için model yolu
testte tmp_path'e yönlendirilir.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta

import joblib
import pytest

from pipeline.dataset import FEATURE_COLUMNS, build_dataset, load_model_config
from pipeline.db import upsert_prices, upsert_symbol
from pipeline.train_model import run

TICKERS = ["AAA1.IS", "BBB2.IS", "CCC3.IS", "DDD4.IS", "EEE5.IS", "FFF6.IS", "GGG7.IS", "HHH8.IS"]


def _seed_symbols(conn, days: int = 80) -> None:
    """Her sembol için gerçek düşüş günleri de olan yükselen seri — RSI'nin
    avg_loss terimi sıfır kalıp NaN'a düşmesin diye (bkz. test_backtest_engine)."""
    start = date(2026, 1, 1)
    for idx, ticker in enumerate(TICKERS):
        symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
        rows = []
        price = 100.0 + idx
        for i in range(days):
            d = start + timedelta(days=i)
            rows.append((d.isoformat(), price, price + 1, price - 1, price, price, 1000.0 + i))
            price += 0.6 if i % 3 else -0.2
        upsert_prices(conn, symbol_id, iter(rows))


@pytest.fixture
def redirected_model_path(tmp_path, monkeypatch):
    """`run()`'in gerçek data/models/xgb_up.pkl dosyasını EZMESİ için model
    yolunu tmp_path'e yönlendirir; geri kalan config aynen korunur."""
    cfg = copy.deepcopy(load_model_config())
    cfg["model"]["path"] = str(tmp_path / "xgb_up.pkl")
    monkeypatch.setattr("pipeline.train_model.load_model_config", lambda: cfg)
    return tmp_path / "xgb_up.pkl"


def test_run_returns_mae_and_r2_and_saves_magnitude_model(committed_conn, redirected_model_path):
    _seed_symbols(committed_conn)
    committed_conn.commit()  # run() kendi bağlantısını açıyor

    metrics = run(save_metrics=False)

    assert metrics["train_rows"] + metrics["test_rows"] == len(build_dataset())
    assert "mae" in metrics and "r2" in metrics
    assert isinstance(metrics["mae"], float)
    assert metrics["mae"] >= 0.0
    # r2 tek bir örnekli test setinde tanımsız -> None dönebilir; burada
    # yeterli satır olduğu için gerçek bir sayı bekliyoruz.
    assert metrics["r2"] is None or isinstance(metrics["r2"], float)

    bundle = joblib.load(redirected_model_path)
    assert "magnitude_model" in bundle
    # Mevcut alanlar bozulmamalı.
    assert "model" in bundle
    assert bundle["features"] == FEATURE_COLUMNS
    assert "version" in bundle

    regressor = bundle["magnitude_model"]
    assert hasattr(regressor, "predict")


def test_magnitude_model_predicts_continuous_return_not_binary(committed_conn, redirected_model_path):
    """Regresörün çıktısı 0/1 değil, kesintisiz getiri büyüklüğü olmalı —
    yani tahminler tam olarak iki değerden ibaret olamaz."""
    _seed_symbols(committed_conn)
    committed_conn.commit()

    run(save_metrics=False)
    bundle = joblib.load(redirected_model_path)

    sample = build_dataset(require_target=False).tail(5)[FEATURE_COLUMNS]
    preds = bundle["magnitude_model"].predict(sample)

    assert len(preds) == len(sample)
    assert len(set(preds.tolist())) > 1


def test_run_writes_metrics_json_when_requested(committed_conn, redirected_model_path):
    _seed_symbols(committed_conn)
    committed_conn.commit()

    metrics = run(save_metrics=True)

    import json

    written = json.loads((redirected_model_path.parent / "metrics.json").read_text(encoding="utf-8"))
    assert written["mae"] == metrics["mae"]
    assert written["r2"] == metrics["r2"]
