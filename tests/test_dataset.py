"""Regression tests for the silent last-row mislabeling bug (docs/BACKTEST_AUDIT.md §2):
`target_up` must be NaN — not a false 0.0 — for the most recent row of each symbol,
and `build_dataset(require_target=False)` must still return that row for live prediction."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from pipeline.dataset import build_dataset
from pipeline.db import upsert_prices, upsert_symbol
from pipeline.features import add_technical_features


def test_target_up_is_nan_for_last_row_not_false():
    close = pd.Series([10.0, 11.0, 9.0, 12.0], index=pd.date_range("2026-01-01", periods=4))
    df = pd.DataFrame({"close": close, "volume": [100.0] * 4})
    out = add_technical_features(df, {})

    assert out["target_up"].iloc[:-1].tolist() == [1.0, 0.0, 1.0]
    assert np.isnan(out["target_up"].iloc[-1])


def _seed_symbol_with_history(conn, days: int = 70) -> int:
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    start = date(2026, 1, 1)
    rows = []
    price = 100.0
    for i in range(days):
        d = start + timedelta(days=i)
        price += 0.1 if i % 2 == 0 else -0.05
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
    # The prediction-time dataset must include one more (later) row than training:
    # the most recent trading day, whose target isn't known yet.
    assert predict_rows["feature_date"].max() > max_feature_date


def test_build_dataset_require_target_false_keeps_all_feature_complete_rows(committed_conn):
    symbol_id = _seed_symbol_with_history(committed_conn)
    committed_conn.commit()

    df = build_dataset(require_target=False)
    rows = df[df["symbol_id"] == symbol_id]
    assert rows["target_up"].isna().sum() == 1  # only the last row, and only that one
