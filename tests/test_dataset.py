"""Regression tests for the silent last-row mislabeling bug (docs/BACKTEST_AUDIT.md §2):
`forward_return`/`target_date` must be NaN — not a false 0.0/"NaT" — for the last
`horizon` rows of each symbol, `build_dataset(require_target=False)` must still
return those rows for live prediction, and `target_up` must be built
cross-sectionally against the daily BIST median in `build_dataset`."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from pipeline.dataset import build_dataset
from pipeline.db import upsert_prices, upsert_symbol
from pipeline.features import add_technical_features

HORIZON = 5  # config/model.yaml::features.horizon


def test_forward_return_and_target_date_are_nan_for_last_horizon_rows():
    """add_technical_features artık yön üretmiyor; ham forward return + hedef
    tarih üretiyor: close(D+h)/open(D+1) - 1, son h satır NaN olmalı."""
    h = 3
    idx = pd.date_range("2026-01-01", periods=6)
    df = pd.DataFrame(
        {
            "open": [10.0, 11.0, 9.0, 12.0, 13.0, 14.0],
            "close": [10.5, 11.5, 9.5, 12.5, 13.5, 14.5],
            "volume": [100.0] * 6,
        },
        index=idx,
    )
    out = add_technical_features(df, {"horizon": h})

    fr = out["forward_return"].to_numpy()
    assert np.isnan(fr[-h:]).all()  # son h satır: exit fiyatı henüz bilinmiyor
    assert pd.isna(out["target_date"].iloc[-1])  # string "NaT" DEĞİL, gerçek NaN
    assert out["target_date"].iloc[0] == (idx[0] + pd.Timedelta(days=h)).strftime("%Y-%m-%d")

    # Elle doğrulama: close[i+h] / open[i+1] - 1
    for i in range(len(idx) - h):
        assert fr[i] == pytest_approx(df["close"].iloc[i + h] / df["open"].iloc[i + 1] - 1)

    # target_up artık BURADA hesaplanmıyor
    assert "target_up" not in out.columns


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-9, abs=1e-12)


def _seed_symbol_with_history(conn, ticker: str = "THYAO.IS", days: int = 70, trend: float = 0.0) -> int:
    """`trend` serinin günlük eğimi (güçlü/weak sembol ayrımı için). ZOR: zigzag
    genliği ±0.05 olduğu için `abs(trend) < 0.05` olmalı — tekdüze artan bir
    seride RSI'nin avg_loss'u hep 0 kalır, RS NaN olur ve dropna TÜM satırları
    siler (aynı tuzak: tests/test_dataset_sentiment_timing.py::_seed_prices)."""
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    start = date(2026, 1, 1)
    rows = []
    price = 100.0
    for i in range(days):
        d = start + timedelta(days=i)
        price += trend + (0.1 if i % 2 == 0 else -0.05)
        rows.append((d.isoformat(), price, price + 1, price - 1, price, price, 1000.0 + i))
    upsert_prices(conn, symbol_id, iter(rows))
    return symbol_id


def test_build_dataset_drops_latest_row_by_default(committed_conn):
    symbol_id = _seed_symbol_with_history(committed_conn)
    committed_conn.commit()  # build_dataset() opens its own connection

    df = build_dataset()
    rows = df[df["symbol_id"] == symbol_id]
    assert not rows.empty
    assert rows["target_up"].isna().sum() == 0  # training set never sees an unknown label
    max_feature_date = rows["feature_date"].max()

    df_predict = build_dataset(require_target=False)
    predict_rows = df_predict[df_predict["symbol_id"] == symbol_id]
    # The prediction-time dataset must include more (later) rows than training:
    # the most recent trading days, whose targets aren't known yet.
    assert predict_rows["feature_date"].max() > max_feature_date


def test_build_dataset_require_target_false_keeps_all_feature_complete_rows(committed_conn):
    symbol_id = _seed_symbol_with_history(committed_conn)
    committed_conn.commit()

    df = build_dataset(require_target=False)
    rows = df[df["symbol_id"] == symbol_id]
    # Son `horizon` satırın forward_return'ı NaN -> etiketi de NaN.
    assert rows["target_up"].isna().sum() == HORIZON
    assert len(rows) > HORIZON + 5  # seed 70 gün, dropna'dan sonra yeterli satır kalmalı


def test_target_up_is_built_against_daily_bist_median(committed_conn):
    """Etiket piyasa-göreli: güçlü trend yapan sembol, o günün BIST medyanının
    üstünde kaldığı günlerde 1.0 almalı; zayıf olan 0.0'a düşmeli."""
    strong = _seed_symbol_with_history(committed_conn, "STRONG.IS", trend=0.04)
    weak = _seed_symbol_with_history(committed_conn, "WEAK.IS", trend=-0.04)
    committed_conn.commit()

    df = build_dataset(require_target=False)
    strong_rows = df[(df["symbol_id"] == strong) & df["target_up"].notna()]
    weak_rows = df[(df["symbol_id"] == weak) & df["target_up"].notna()]
    assert not strong_rows.empty and not weak_rows.empty

    assert strong_rows["target_up"].mean() > weak_rows["target_up"].mean()
    # Medyan-üstü tanımı birebir doğrula (güçlü sembol örneklemi üzerinden).
    med = (
        df[df["is_bist"] == 1.0].groupby("feature_date")["forward_return"].median()
    )
    sample = strong_rows.iloc[0]
    expected = 1.0 if sample["forward_return"] > med[sample["feature_date"]] else 0.0
    assert sample["target_up"] == expected
    assert strong_rows["target_up"].max() == 1.0
