from pathlib import Path

from trading.config import load_trading_config

FIXTURE = Path(__file__).parent / "fixtures" / "trading_test.yaml"


def test_load_trading_config_from_project_default():
    cfg = load_trading_config()
    assert cfg.starting_balance == 10000.0
    assert cfg.buy_threshold == 0.55
    assert cfg.max_open_positions == 8


def test_load_trading_config_from_explicit_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
starting_balance: 5000.0
buy_threshold: 0.7
sell_threshold: 0.4
max_hold_days: 5
max_open_positions: 3
max_position_pct: 0.2
max_portfolio_exposure_pct: 0.8
commission_pct: 0.1
bsmv_pct_of_commission: 5.0
min_commission_try: 2.0
min_position_value_try: 100.0
""",
        encoding="utf-8",
    )
    cfg = load_trading_config(custom)
    assert cfg.starting_balance == 5000.0
    assert cfg.max_open_positions == 3
