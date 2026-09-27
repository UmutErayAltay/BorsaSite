"""Faz 7: predict_model.run() için beklenen getiri büyüklüğü.

Canlıdaki mevcut .pkl dosyası `magnitude_model` İÇERMEZ — eski format bundle
ile de tahmin üretilebilmeli (expected_return None, avg_expected_return None).
Yeni format bundle ile ise expected_return hesaplanıp predictions tablosunun
expected_return kolonuna yazılmalı.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta

import joblib
import numpy as np
import pytest

from pipeline.dataset import FEATURE_COLUMNS, load_model_config
from pipeline.db import get_connection, upsert_prices, upsert_symbol

TICKERS = ["AAA1.IS", "BBB2.IS", "CCC3.IS", "DDD4.IS", "EEE5.IS", "FFF6.IS", "GGG7.IS", "HHH8.IS"]


class _FixedProbModel:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.7)])


class _FixedReturnModel:
    def __init__(self, value: float = 0.0123) -> None:
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _seed_symbols(conn, days: int = 80) -> None:
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
    cfg = copy.deepcopy(load_model_config())
    cfg["model"]["path"] = str(tmp_path / "xgb_up.pkl")
    monkeypatch.setattr("pipeline.predict_model.load_model_config", lambda: cfg)
    return tmp_path / "xgb_up.pkl"


def test_old_bundle_without_magnitude_model_does_not_crash(committed_conn, redirected_model_path):
    """Eski format: bundle'da magnitude_model YOK. run() çökmemeli,
    expected_return None kalmalı ve DB'ye null yazılmalı."""
    _seed_symbols(committed_conn)
    joblib.dump(
        {"model": _FixedProbModel(), "features": FEATURE_COLUMNS, "version": "1.0"},
        redirected_model_path,
    )
    committed_conn.commit()  # run() kendi bağlantısını açıyor

    from pipeline.predict_model import run

    stats = run()

    assert stats["predictions"] > 0
    assert stats["avg_expected_return"] is None

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT expected_return FROM predictions WHERE model_version = '1.0'"
        ).fetchall()
    assert rows, "hiç tahmin yazılmadı"
    assert all(r["expected_return"] is None for r in rows)


def test_new_bundle_writes_expected_return(committed_conn, redirected_model_path):
    _seed_symbols(committed_conn)
    joblib.dump(
        {
            "model": _FixedProbModel(),
            "magnitude_model": _FixedReturnModel(0.0123),
            "features": FEATURE_COLUMNS,
            "version": "1.1",
        },
        redirected_model_path,
    )
    committed_conn.commit()

    from pipeline.predict_model import run

    stats = run()

    assert stats["predictions"] > 0
    assert stats["avg_expected_return"] == pytest.approx(0.0123, abs=1e-6)

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT expected_return FROM predictions WHERE model_version = '1.1'"
        ).fetchall()
    assert rows, "hiç tahmin yazılmadı"
    assert all(r["expected_return"] is not None for r in rows)
    assert all(abs(r["expected_return"] - 0.0123) < 1e-6 for r in rows)


def test_expected_return_is_overwritten_on_reupsert(committed_conn, redirected_model_path):
    """ON CONFLICT DO UPDATE gerçekten expected_return'ı da güncelliyor mu?"""
    _seed_symbols(committed_conn)
    joblib.dump(
        {
            "model": _FixedProbModel(),
            "magnitude_model": _FixedReturnModel(-0.05),
            "features": FEATURE_COLUMNS,
            "version": "1.1",
        },
        redirected_model_path,
    )
    committed_conn.commit()

    from pipeline.predict_model import run

    run()
    with get_connection() as conn:
        conn.execute("UPDATE predictions SET expected_return = 0.0 WHERE model_version = '1.1'")

    run()  # ikinci koşuda aynı (symbol_id, feature_date, version) satırlarına yazılır
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT expected_return FROM predictions WHERE model_version = '1.1'"
        ).fetchall()
    assert rows
    assert all(abs(r["expected_return"] - (-0.05)) < 1e-6 for r in rows)
