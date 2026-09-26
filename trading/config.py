"""Trading motorunun yapılandırması — config/trading.yaml'dan yüklenir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "trading.yaml"


@dataclass(frozen=True)
class TradingConfig:
    starting_balance: float
    buy_threshold: float
    sell_threshold: float
    max_hold_days: int
    max_open_positions: int
    max_position_pct: float
    max_portfolio_exposure_pct: float
    commission_pct: float
    bsmv_pct_of_commission: float
    min_commission_try: float
    min_position_value_try: float
    # Faz 6 risk yönetimi — varsayılanlar 0.0/0 (kapalı), böylece bu alanları
    # vermeyen çağıranların (testler dahil) davranışı değişmez.
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    cooldown_days_after_exit: int = 0


def load_trading_config(path: Path | None = None) -> TradingConfig:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TradingConfig(**data)
