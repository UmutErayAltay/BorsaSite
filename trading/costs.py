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


@dataclass(frozen=True)
class ExpectedEdgeCheck:
    allowed: bool
    reason: str


def check_expected_edge(
    expected_return_pct: float,
    position_value: float,
    round_trip_fee: float,
    cfg: TradingConfig,
) -> ExpectedEdgeCheck:
    """Faz 6/7 (docs/BACKTEST_AUDIT.md §6): beklenen brut kar, round-trip
    (alim+satim) ucretini + min_expected_edge_pct guvenlik payini asmali.
    cfg.min_expected_edge_pct <= 0 ise (varsayilan) bu kontrol TAMAMEN
    KAPALI — hicbir aday reddedilmez, mevcut davranis degismez."""
    if cfg.min_expected_edge_pct <= 0:
        return ExpectedEdgeCheck(allowed=True, reason="edge kontrolü kapalı")
    expected_profit = expected_return_pct * position_value
    required = round_trip_fee + cfg.min_expected_edge_pct * position_value
    if expected_profit < required:
        return ExpectedEdgeCheck(
            allowed=False,
            reason=(
                f"beklenen kâr {expected_profit:.2f} TL < maliyet+pay "
                f"{required:.2f} TL (round-trip ücret {round_trip_fee:.2f} TL)"
            ),
        )
    return ExpectedEdgeCheck(allowed=True, reason="beklenen edge yeterli")
