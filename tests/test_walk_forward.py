from backtest.walk_forward import WalkForwardConfig, load_walk_forward_config, make_windows


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
