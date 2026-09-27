"""`pipeline/intraday_train_model.py` integration tests against the real
local Postgres (synthetic bars, no network) — mirrors tests/test_train_model.py.

DİKKAT — `tests/conftest.py::committed_conn` bu dosyada KULLANILMIYOR: onun
teardown'ı tüm tabloları TRUNCATE eder, yani veritabanındaki gerçek (ve
pahalı) 547K satırlık intraday verisini siler. Aşağıdaki `intraday_conn`,
`tests/test_intraday_dataset.py`'deki fixture ile aynı "commit'li, ayrı
bağlantıdan görünür" davranışı verir ama teardown'da YALNIZCA bu dosyanın
kullandığı test ticker'larını ve kendi deney satırlarını temizler.
"""
from __future__ import annotations

import copy
import json

import joblib
import pandas as pd
import psycopg
import pytest

from pipeline.db import (
    ConnWrapper,
    get_connection,
    get_database_url,
    init_schema,
    list_model_experiments,
    upsert_intraday_prices,
    upsert_symbol,
)
from pipeline.intraday_dataset import (
    INTRADAY_FEATURE_COLUMNS,
    build_intraday_dataset,
    load_intraday_model_config,
)
from pipeline.intraday_train_model import run

INTERVAL = "1h"

# Yalnız bu dosyada üretilen ticker'lar — temizlik bunlarla sınırlıdır
# (test_intraday_dataset.py'nin TEST1.IS/KEEP.IS/... ile çakışmaz).
_SEED_TICKERS = ("TRAIN1.IS", "TRAIN2.IS")

# Günlük pipeline'ın metrics dosyası: intraday eğitimi bunu ASLA ezmemeli.
_DAILY_METRICS = "data/models/metrics.json"


@pytest.fixture
def intraday_conn():
    raw = psycopg.connect(get_database_url())
    wrapper = ConnWrapper(raw)
    init_schema(wrapper)
    raw.commit()  # kilitleri bırak, gövde başlasın
    try:
        yield wrapper
        raw.commit()
    finally:
        raw.rollback()
        cur = raw.cursor()
        cur.execute("SELECT id FROM symbols WHERE ticker = ANY(%s)", (list(_SEED_TICKERS),))
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            cur.execute("DELETE FROM prices_intraday WHERE symbol_id = ANY(%s)", (ids,))
            cur.execute("DELETE FROM symbols WHERE id = ANY(%s)", (ids,))
        raw.commit()
        raw.close()


def _seed_symbol_with_intraday_bars(conn, ticker: str, *, days: int, bars_per_day: int = 5) -> int:
    """tests/test_intraday_dataset.py ile aynı üretim deseni — gün sayısı
    dışarıdan verilir ki `min_history_bars` eşiği tek değişkenle geçilebilsin."""
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    rows = []
    price = 100.0
    for d in range(days):
        day = pd.Timestamp("2026-01-05", tz="Europe/Istanbul") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            price += 0.1 if b % 2 == 0 else -0.05
            ts = day + pd.Timedelta(hours=10 + b)
            rows.append((ts.isoformat(), price, price + 0.5, price - 0.5, price, 1000.0 + b))
    upsert_intraday_prices(conn, symbol_id, INTERVAL, iter(rows))
    return symbol_id


@pytest.fixture
def redirected_model_path(tmp_path, monkeypatch):
    """`run()`'in gerçek data/models/xgb_intraday_1h.pkl dosyasını EZMESİ
    için model yolunu tmp_path'e yönlendirir; geri kalan config aynen korunur.

    `build_intraday_dataset()` config'i KENDİ modülünden okuduğu için
    `pipeline.intraday_dataset.load_intraday_model_config` de override
    edilmek zorunda — aksi halde eşiği küçülttüğümüz `min_history_bars`
    test sembollerini eler ve gerçek sembollerden (>=5023 bar) eğitim yapılır."""
    cfg = copy.deepcopy(load_intraday_model_config())
    cfg["model"]["path"] = str(tmp_path / "xgb_intraday_1h.pkl")
    cfg["features"]["min_history_bars"] = 100
    cfg["training"]["xgb"]["n_estimators"] = 20
    cfg["training"]["xgb_regressor"]["n_estimators"] = 20
    for target in (
        "pipeline.intraday_train_model.load_intraday_model_config",
        "pipeline.intraday_dataset.load_intraday_model_config",
    ):
        monkeypatch.setattr(target, lambda: cfg)
    return tmp_path / "xgb_intraday_1h.pkl"


@pytest.fixture
def experiment_watermark():
    """`run()` kendi bağlantısını COMMIT ettiği için testin kendi
    transaction'ı yeni satırları görmez. Temizlikte yalnız BU testten
    sonra eklenen satırları silebilmek için başlangıçtaki en yüksek id'yi
    tutuyoruz — gerçek deney geçmişine dokunmadan."""
    with get_connection() as conn:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM model_experiments").fetchone()
        yield int(row["m"])
    with get_connection() as conn:
        conn.execute("DELETE FROM model_experiments WHERE id > %s", (int(row["m"]),))


def _seed_and_commit(conn) -> None:
    for ticker in _SEED_TICKERS:
        _seed_symbol_with_intraday_bars(conn, ticker, days=80)
    conn.commit()  # run() kendi bağlantısını açıyor


def test_run_returns_mae_r2_and_feature_importance_and_saves_magnitude_model(
    intraday_conn, redirected_model_path, experiment_watermark
):
    _seed_and_commit(intraday_conn)

    metrics = run(save_metrics=False)

    assert "mae" in metrics and "r2" in metrics
    assert isinstance(metrics["mae"], float)
    assert metrics["mae"] >= 0.0
    # r2 tek örnekli test setinde tanımsız -> None dönebilir.
    assert metrics["r2"] is None or isinstance(metrics["r2"], float)

    importance = metrics["feature_importance"]
    assert set(importance) == set(INTRADAY_FEATURE_COLUMNS)
    values = list(importance.values())
    assert values == sorted(values, reverse=True)

    bundle = joblib.load(redirected_model_path)
    assert "magnitude_model" in bundle
    assert "model" in bundle
    assert bundle["features"] == INTRADAY_FEATURE_COLUMNS
    assert "version" in bundle

    regressor = bundle["magnitude_model"]
    assert hasattr(regressor, "predict")

    # Önemlilik gerçekten eğitilmiş modelden gelmeli.
    for col, imp in zip(INTRADAY_FEATURE_COLUMNS, bundle["model"].feature_importances_):
        assert importance[col] == pytest.approx(round(float(imp), 4), abs=1e-4)


def test_magnitude_model_predicts_continuous_return_not_binary(
    intraday_conn, redirected_model_path, experiment_watermark
):
    """Regresörün çıktısı 0/1 değil, kesintisiz getiri büyüklüğü olmalı."""
    _seed_and_commit(intraday_conn)

    run(save_metrics=False)
    bundle = joblib.load(redirected_model_path)

    sample = build_intraday_dataset().tail(5)[INTRADAY_FEATURE_COLUMNS]
    preds = bundle["magnitude_model"].predict(sample)

    assert len(preds) == len(sample)
    # Tam olarak iki değerden ibaret olamaz (o zaman sınıflandırıcı olurdu).
    assert len(set(preds.tolist())) > 1


def test_run_records_experiment_row_with_intraday_prefix(intraday_conn, redirected_model_path, experiment_watermark):
    _seed_and_commit(intraday_conn)

    metrics = run(save_metrics=False)

    with get_connection() as conn:
        experiments = list_model_experiments(conn, limit=100)
    rows = [e for e in experiments if str(e["model_version"]).startswith("intraday-")]
    assert rows, "model_experiments içinde 'intraday-' önekli satır yok"
    row = rows[0]

    assert row["model_version"] == f"intraday-{load_intraday_model_config()['model']['version']}"
    assert row["train_rows"] == metrics["train_rows"]
    assert row["test_rows"] == metrics["test_rows"]
    assert row["symbols"] == metrics["symbols"]
    assert row["accuracy"] == pytest.approx(metrics["accuracy"], abs=1e-4)
    assert row["mae"] == pytest.approx(metrics["mae"], abs=1e-4)
    assert row["r2"] == pytest.approx(metrics["r2"], abs=1e-4)

    params = row["params"]
    if isinstance(params, str):  # sürücü JSONB'yi dict yerine metin olarak döndürebilir
        params = json.loads(params)
    assert "xgb_regressor" in params
    assert "xgb" in params


def test_run_writes_separate_metrics_file_and_leaves_daily_metrics_untouched(
    intraday_conn, redirected_model_path, experiment_watermark
):
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    daily_metrics = project_root / _DAILY_METRICS
    before = daily_metrics.read_text(encoding="utf-8") if daily_metrics.exists() else None

    _seed_and_commit(intraday_conn)

    metrics = run(save_metrics=True)

    # AYRI dosya: günlük "metrics.json" değil.
    written_path = redirected_model_path.parent / "xgb_intraday_1h_metrics.json"
    assert written_path.exists()
    assert not (redirected_model_path.parent / "metrics.json").exists()
    written = json.loads(written_path.read_text(encoding="utf-8"))
    assert written["mae"] == metrics["mae"]
    assert written["r2"] == metrics["r2"]
    assert set(written["feature_importance"]) == set(INTRADAY_FEATURE_COLUMNS)

    after = daily_metrics.read_text(encoding="utf-8") if daily_metrics.exists() else None
    assert after == before, f"günlük {_DAILY_METRICS} değişmiş"
