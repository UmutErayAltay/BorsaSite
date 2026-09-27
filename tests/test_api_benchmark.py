"""BIST 100 karşılaştırma endpoint'i — yfinance ASLA gerçek çağrılmaz,
`benchmark.fetch_index_closes` her testte monkeypatch ile sahte Series döner."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import benchmark
from api.benchmark import build_benchmark_payload
from api.main import app

client = TestClient(app)


def _series(pairs: list[tuple[str, float]]) -> pd.Series:
    s = pd.Series([v for _, v in pairs], dtype=float, index=pd.Index([d for d, _ in pairs]))
    s.index = pd.to_datetime(s.index).date
    return s


@pytest.fixture
def snapshot_rows(committed_conn):
    """`portfolio_snapshots` conftest'in TRUNCATE listesinde olmadığı için
    bu testin yazdığı satırlar kalıcı olurdu — burada temizleniyor."""
    committed_conn.execute("DELETE FROM portfolio_snapshots")
    committed_conn.commit()
    yield committed_conn
    committed_conn.execute("DELETE FROM portfolio_snapshots")
    committed_conn.commit()


# --- saf fonksiyon: build_benchmark_payload --------------------------------


def test_normalization_scales_benchmark_to_first_portfolio_value():
    payload = build_benchmark_payload(
        [
            {"snapshot_date": "2026-09-01", "total_value": 1000.0},
            {"snapshot_date": "2026-09-02", "total_value": 1100.0},
            {"snapshot_date": "2026-09-03", "total_value": 900.0},
        ],
        _series([("2026-09-01", 50.0), ("2026-09-02", 55.0), ("2026-09-03", 45.0)]),
    )

    assert payload["benchmark_available"] is True
    assert payload["benchmark_label"] == "BIST 100"
    assert payload["benchmark_ticker"] == "XU100.IS"
    # İlk nokta her iki eğride de başlangıç değerine eşitlenir
    assert payload["items"][0] == {"date": "2026-09-01", "portfolio": 1000.0, "benchmark": 1000.0}
    # 1000 * 55/50 = 1100, 1000 * 45/50 = 900
    assert [i["benchmark"] for i in payload["items"]] == [1000.0, 1100.0, 900.0]
    assert [i["portfolio"] for i in payload["items"]] == [1000.0, 1100.0, 900.0]
    assert payload["portfolio_return_pct"] == -10.0
    assert payload["benchmark_return_pct"] == -10.0
    assert payload["difference_pct_points"] == 0.0
    assert payload["note"] is None


def test_missing_index_day_falls_back_to_previous_close():
    # 2026-09-02 için (tatil) endeks kapanışı yok → 09-01'in kapanışı kullanılır
    payload = build_benchmark_payload(
        [
            {"snapshot_date": date(2026, 9, 1), "total_value": 200.0},
            {"snapshot_date": date(2026, 9, 2), "total_value": 210.0},
            {"snapshot_date": date(2026, 9, 4), "total_value": 240.0},
        ],
        _series([("2026-09-01", 100.0), ("2026-09-03", 110.0), ("2026-09-04", 120.0)]),
    )

    assert payload["benchmark_available"] is True
    # 200 * 100/100 = 200 (önceki kapanış), 200 * 110/100 = 220, 200 * 120/100 = 240
    assert [i["benchmark"] for i in payload["items"]] == [200.0, 200.0, 240.0]
    assert payload["benchmark_return_pct"] == 20.0
    assert payload["portfolio_return_pct"] == 20.0


def test_empty_index_marks_benchmark_unavailable():
    payload = build_benchmark_payload(
        [{"snapshot_date": "2026-09-01", "total_value": 500.0}],
        pd.Series(dtype=float),
    )

    assert payload["benchmark_available"] is False
    assert payload["items"][0]["benchmark"] is None
    assert payload["items"][0]["portfolio"] == 500.0
    assert payload["benchmark_return_pct"] is None
    assert payload["difference_pct_points"] is None
    assert payload["note"] == "Endeks verisi alınamadı"
    # Portföy tarafı yine hesaplanır
    assert payload["portfolio_return_pct"] == 0.0


def test_index_starting_after_first_snapshot_is_unavailable():
    payload = build_benchmark_payload(
        [
            {"snapshot_date": "2026-09-01", "total_value": 1000.0},
            {"snapshot_date": "2026-09-05", "total_value": 1200.0},
        ],
        _series([("2026-09-02", 90.0), ("2026-09-05", 110.0)]),
    )

    assert payload["benchmark_available"] is False
    assert [i["benchmark"] for i in payload["items"]] == [None, None]
    assert payload["benchmark_return_pct"] is None
    assert payload["note"] == "Endeks verisi alınamadı"


def test_empty_snapshots_returns_empty_items():
    payload = build_benchmark_payload([], _series([("2026-09-01", 100.0)]))

    assert payload["items"] == []
    assert payload["benchmark_available"] is False
    assert payload["portfolio_return_pct"] is None
    assert payload["benchmark_return_pct"] is None
    assert payload["difference_pct_points"] is None


def test_return_and_difference_calculation():
    payload = build_benchmark_payload(
        [
            {"snapshot_date": "2026-09-01", "total_value": 1000.0},
            {"snapshot_date": "2026-09-10", "total_value": 1500.0},
        ],
        _series([("2026-09-01", 100.0), ("2026-09-10", 120.0)]),
    )

    assert payload["portfolio_return_pct"] == 50.0
    assert payload["benchmark_return_pct"] == 20.0
    assert payload["difference_pct_points"] == 30.0
    # İşaretli (kötü) getiri farkı
    losing = build_benchmark_payload(
        [
            {"snapshot_date": "2026-09-01", "total_value": 1000.0},
            {"snapshot_date": "2026-09-10", "total_value": 800.0},
        ],
        _series([("2026-09-01", 100.0), ("2026-09-10", 130.0)]),
    )
    assert losing["portfolio_return_pct"] == -20.0
    assert losing["benchmark_return_pct"] == 30.0
    assert losing["difference_pct_points"] == -50.0


def test_snapshots_are_sorted_ascending_regardless_of_input_order():
    payload = build_benchmark_payload(
        [
            {"snapshot_date": "2026-09-03", "total_value": 900.0},
            {"snapshot_date": "2026-09-01", "total_value": 1000.0},
            {"snapshot_date": "2026-09-02", "total_value": 1100.0},
        ],
        _series([("2026-09-01", 100.0), ("2026-09-02", 110.0), ("2026-09-03", 90.0)]),
    )

    assert [i["date"] for i in payload["items"]] == ["2026-09-01", "2026-09-02", "2026-09-03"]
    assert [i["benchmark"] for i in payload["items"]] == [1000.0, 1100.0, 900.0]


# --- endpoint ---------------------------------------------------------------


@pytest.fixture
def fake_index(monkeypatch):
    closes = _series(
        [("2026-08-28", 95.0), ("2026-09-01", 100.0), ("2026-09-02", 102.0), ("2026-09-03", 90.0)]
    )
    calls: list[tuple] = []

    def _fake(start, end, ticker="XU100.IS"):
        calls.append((start, end, ticker))
        return closes

    monkeypatch.setattr(benchmark, "fetch_index_closes", _fake)
    return calls


def test_benchmark_endpoint_returns_normalized_series(snapshot_rows, fake_index):
    for day, total in [("2026-09-01", 1000.0), ("2026-09-02", 1100.0), ("2026-09-03", 900.0)]:
        snapshot_rows.execute(
            """
            INSERT INTO portfolio_snapshots (snapshot_date, balance, positions_value, total_value)
            VALUES (?, ?, ?, ?)
            """,
            (day, total - 100.0, 100.0, total),
        )
    snapshot_rows.commit()

    response = client.get("/api/portfolio/benchmark", params={"days": 365})

    assert response.status_code == 200
    data = response.json()
    assert data["benchmark_available"] is True
    assert data["benchmark_label"] == "BIST 100"
    assert data["benchmark_ticker"] == "XU100.IS"
    assert [i["date"] for i in data["items"]] == ["2026-09-01", "2026-09-02", "2026-09-03"]
    assert [i["portfolio"] for i in data["items"]] == [1000.0, 1100.0, 900.0]
    # 1000 * 100/100, 1000 * 102/100, 1000 * 90/100
    assert [i["benchmark"] for i in data["items"]] == [1000.0, 1020.0, 900.0]
    assert data["portfolio_return_pct"] == -10.0
    assert data["benchmark_return_pct"] == -10.0
    assert data["difference_pct_points"] == 0.0
    assert data["note"] is None
    # Endeks aralığı ilk snapshot'tan 7 gün önce, son snapshot'tan 1 gün sonra
    assert fake_index[0][0] == date(2026, 8, 25)
    assert fake_index[0][1] == date(2026, 9, 4)


def test_benchmark_endpoint_reports_missing_index(snapshot_rows, monkeypatch):
    monkeypatch.setattr(benchmark, "fetch_index_closes", lambda start, end, ticker="XU100.IS": pd.Series(dtype=float))
    snapshot_rows.execute(
        """
        INSERT INTO portfolio_snapshots (snapshot_date, balance, positions_value, total_value)
        VALUES ('2026-09-01', 900.0, 100.0, 1000.0)
        """
    )
    snapshot_rows.commit()

    response = client.get("/api/portfolio/benchmark")

    assert response.status_code == 200
    data = response.json()
    assert data["benchmark_available"] is False
    assert data["items"][0]["benchmark"] is None
    assert data["note"] == "Endeks verisi alınamadı"


def test_benchmark_endpoint_without_snapshots_skips_index_fetch(snapshot_rows, fake_index):
    response = client.get("/api/portfolio/benchmark")

    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["portfolio_return_pct"] is None
    assert fake_index == []
