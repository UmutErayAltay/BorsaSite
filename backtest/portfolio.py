"""Bellek-içi (DB'siz) portföy simülatörü — `trading/portfolio.py`'nin Postgres'e
bağlı canlı hâlinin backtest eşdeğeri. Karar mantığı (limit kontrolleri, komisyon,
pozisyon büyüklüğü) `trading/costs.py`'deki SAF fonksiyonlarla birebir aynı;
sadece durum saklama şekli (bellek-içi dataclass vs. tablolar) farklı, aynı
finansal mantığın iki farklı implementasyonu olmasın diye."""

from __future__ import annotations

from dataclasses import dataclass, field

from backtest.costs import BacktestCostConfig, execution_price
from trading.config import TradingConfig
from trading.costs import calculate_fee, check_position_size


@dataclass
class BTPosition:
    symbol: str
    entry_price: float
    quantity: float
    entry_prob_up: float
    opened_at: str
    entry_fee: float


@dataclass(frozen=True)
class BTClosedTrade:
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees_paid: float
    net_pnl: float
    exit_reason: str
    opened_at: str
    closed_at: str


@dataclass
class BacktestPortfolio:
    starting_balance: float
    balance: float = field(init=False)
    open_positions: dict[str, BTPosition] = field(default_factory=dict)
    closed_trades: list[BTClosedTrade] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.balance = self.starting_balance

    def buy(
        self,
        symbol: str,
        mid_price: float,
        prob_up: float,
        decision_date: str,
        trading_cfg: TradingConfig,
        cost_cfg: BacktestCostConfig,
    ) -> tuple[bool, str]:
        if symbol in self.open_positions:
            return False, "Zaten açık pozisyon var"
        if len(self.open_positions) >= trading_cfg.max_open_positions:
            return False, f"Açık pozisyon limiti doldu ({trading_cfg.max_open_positions})"

        position_value = self.balance * trading_cfg.max_position_pct
        size_check = check_position_size(position_value, trading_cfg)
        if not size_check.allowed:
            return False, size_check.reason

        price = execution_price(mid_price, "buy", cost_cfg)
        fee = calculate_fee(position_value, trading_cfg)
        total_cost = position_value + fee.total_fee
        if total_cost > self.balance:
            return False, (
                f"Bakiye yetersiz: gereken {total_cost:.2f} TL, mevcut {self.balance:.2f} TL"
            )

        currently_locked = sum(p.entry_price * p.quantity for p in self.open_positions.values())
        projected_pct = (currently_locked + position_value) / self.starting_balance * 100
        limit_pct = trading_cfg.max_portfolio_exposure_pct * 100
        if projected_pct > limit_pct:
            return False, (
                f"Toplam portföy riski aşılırdı: %{projected_pct:.1f} (sınır %{limit_pct:.1f})"
            )

        quantity = position_value / price
        self.balance -= total_cost
        self.open_positions[symbol] = BTPosition(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            entry_prob_up=prob_up,
            opened_at=decision_date,
            entry_fee=fee.total_fee,
        )
        return True, f"Alındı: {symbol} @ {price:.4f} x {quantity:.6f} (ücret {fee.total_fee:.2f} TL)"

    def sell(
        self,
        symbol: str,
        mid_price: float,
        exit_reason: str,
        decision_date: str,
        trading_cfg: TradingConfig,
        cost_cfg: BacktestCostConfig,
    ) -> BTClosedTrade | None:
        position = self.open_positions.get(symbol)
        if position is None:
            return None

        price = execution_price(mid_price, "sell", cost_cfg)
        gross_proceeds = price * position.quantity
        fee = calculate_fee(gross_proceeds, trading_cfg)
        net_proceeds = gross_proceeds - fee.total_fee

        entry_cost = position.entry_price * position.quantity
        gross_pnl = gross_proceeds - entry_cost
        total_fees = position.entry_fee + fee.total_fee
        net_pnl = gross_pnl - total_fees

        self.balance += net_proceeds
        del self.open_positions[symbol]

        trade = BTClosedTrade(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            gross_pnl=round(gross_pnl, 2),
            fees_paid=round(total_fees, 2),
            net_pnl=round(net_pnl, 2),
            exit_reason=exit_reason,
            opened_at=position.opened_at,
            closed_at=decision_date,
        )
        self.closed_trades.append(trade)
        return trade

    def positions_value(self, latest_prices: dict[str, float]) -> float:
        return sum(
            latest_prices.get(p.symbol, p.entry_price) * p.quantity
            for p in self.open_positions.values()
        )

    def record_snapshot(self, decision_date: str, latest_prices: dict[str, float]) -> None:
        total = self.balance + self.positions_value(latest_prices)
        self.equity_curve.append((decision_date, round(total, 2)))
