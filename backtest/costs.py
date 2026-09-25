"""Backtest'e özel maliyet katmanı. Komisyon/BSMV hesabı ve pozisyon-büyüklüğü
kontrolü `trading/costs.py`'den AYNEN kullanılıyor (tek finansal mantık, iki
implementasyon yok) — burada sadece backtest'e özgü olan slippage/spread var,
canlı motorun (`trading/engine.py`) hiç bilmediği bir kavram."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestCostConfig:
    slippage_bps: float = 0.0
    spread_bps: float = 0.0


def execution_price(mid_price: float, side: str, cfg: BacktestCostConfig) -> float:
    """BUY: execution_price > mid_price. SELL: execution_price < mid_price.
    Spread'in yarısı + slippage'ın tamamı işlem yönüne göre uygulanır."""
    if side not in ("buy", "sell"):
        raise ValueError(f"gecersiz side: {side!r}")
    total_bps = cfg.slippage_bps + cfg.spread_bps / 2
    adjustment = mid_price * (total_bps / 10000.0)
    return mid_price + adjustment if side == "buy" else mid_price - adjustment


SCENARIOS: dict[str, BacktestCostConfig] = {
    "base": BacktestCostConfig(slippage_bps=0.0, spread_bps=0.0),
    "conservative": BacktestCostConfig(slippage_bps=10.0, spread_bps=20.0),
    "stress": BacktestCostConfig(slippage_bps=30.0, spread_bps=50.0),
}
