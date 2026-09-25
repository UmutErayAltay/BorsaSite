import pandas as pd

from backtest.walk_forward import (
    WalkForwardConfig,
    load_walk_forward_config,
    make_windows,
    run_walk_forward,
)
from pipeline.dataset import FEATURE_COLUMNS
from trading.config import TradingConfig


def test_windows_are_chronological_and_non_overlapping_oos():
    dates = [f"2025-01-{i:02d}" for i in range(1, 31)]
    cfg = WalkForwardConfig(train_days=10, validation_days=5, oos_days=5, step_days=5)
    windows = make_windows(dates, cfg)
    assert windows
    for w in windows:
        assert w.train_end < w.validation_start < w.validation_end < w.oos_start <= w.oos_end
    for a, b in zip(windows, windows[1:]):
        assert a.oos_end < b.oos_start


def test_windows_require_full_history():
    dates = [f"2025-01-{i:02d}" for i in range(1, 10)]
    cfg = WalkForwardConfig(train_days=5, validation_days=3, oos_days=3)
    assert make_windows(dates, cfg) == []


def test_load_walk_forward_config_reads_yaml_defaults():
    cfg = load_walk_forward_config()
    assert cfg.train_days == 504
    assert cfg.validation_days == 63
    assert cfg.oos_days == 63
    assert cfg.calibration_method == "sigmoid"


def _synthetic_dataset(n_days: int, validation_days: int, train_days: int, constant_validation_target: bool) -> pd.DataFrame:
    rows = []
    for i in range(n_days):
        date = f"2025-01-{i + 1:02d}" if i < 31 else f"2025-02-{i - 30:02d}"
        # VALIDATION penceresi tek sınıflı olacak şekilde etiketi zorla — bu,
        # gerçek BIST verisiyle karşılaşılan bir vakayı taklit ediyor: bazı
        # pencerelerde model/eşik/kalibrasyon için işlem yapılabilir bir sinyal
        # bulunamayabilir, bu koşuyu ÇÖKERTMEMELİ (bkz. run_walk_forward).
        in_validation = train_days <= i < train_days + validation_days
        target = 1.0 if (constant_validation_target and in_validation) else float(i % 2)
        row = {c: 0.5 for c in FEATURE_COLUMNS}
        row.update({
            "feature_date": date,
            "ticker": "TEST.IS",
            "is_bist": 1.0,
            "open": 100.0 + i,
            "close": 100.0 + i,
            "target_up": target,
        })
        rows.append(row)
    return pd.DataFrame(rows)


TEST_TRADING_CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.55,
    sell_threshold=0.50,
    max_hold_days=5,
    max_open_positions=5,
    max_position_pct=0.3,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=1.0,
    min_position_value_try=1.0,
)


def test_window_with_single_class_validation_is_skipped_not_crashed():
    cfg = WalkForwardConfig(train_days=5, validation_days=3, oos_days=3, step_days=3, min_train_rows=1)
    df = _synthetic_dataset(n_days=11, validation_days=3, train_days=5, constant_validation_target=True)

    result = run_walk_forward(cfg=cfg, trading_cfg=TEST_TRADING_CFG, dataset=df)

    assert result.windows == []
    assert result.trades == []


def test_window_with_mixed_validation_produces_a_window():
    cfg = WalkForwardConfig(train_days=5, validation_days=3, oos_days=3, step_days=3, min_train_rows=1, min_threshold_trades=1)
    df = _synthetic_dataset(n_days=11, validation_days=3, train_days=5, constant_validation_target=False)

    result = run_walk_forward(cfg=cfg, trading_cfg=TEST_TRADING_CFG, dataset=df)

    assert len(result.windows) == 1
