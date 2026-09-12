"""Komisyon/BSMV hesabı ve pozisyon-büyüklüğü kontrolü — SAF fonksiyonlar,
hiçbir I/O yok. `risk/cost_check.py` (kripto-trading-bot) disiplininin
aynısı: finansal mantık tek, izole, kolay test edilir bir yerde durur."""

from __future__ import annotations

from dataclasses import dataclass

from trading.config import TradingConfig


@dataclass(frozen=True)
class CommissionResult:
    commission: float
    bsmv: float
    total_fee: float


def calculate_fee(trade_value: float, cfg: TradingConfig) -> CommissionResult:
    commission = trade_value * (cfg.commission_pct / 100.0)
    bsmv = commission * (cfg.bsmv_pct_of_commission / 100.0)
    total_fee = max(commission + bsmv, cfg.min_commission_try)
    return CommissionResult(
        commission=round(commission, 2),
        bsmv=round(bsmv, 2),
        total_fee=round(total_fee, 2),
    )


@dataclass(frozen=True)
class PositionSizeCheck:
    allowed: bool
    reason: str


def check_position_size(position_value: float, cfg: TradingConfig) -> PositionSizeCheck:
    """`kripto-trading-bot`'taki 'beklenen brüt kâr > toplam maliyet'
    kontrolünün DEĞİL, bu stratejinin şekline (eşik-bazlı, stop-loss/
    take-profit yok) uyarlanmış basitleştirilmiş biçimi: pozisyon o kadar
    küçükse asgari işlem ücretinin payı orantısız büyür, böyle bir işlem
    reddedilir."""
    if position_value < cfg.min_position_value_try:
        return PositionSizeCheck(
            allowed=False,
            reason=(
                f"Pozisyon çok küçük: {position_value:.2f} TL < "
                f"asgari {cfg.min_position_value_try:.2f} TL"
            ),
        )
    return PositionSizeCheck(allowed=True, reason="Pozisyon büyüklüğü yeterli")
