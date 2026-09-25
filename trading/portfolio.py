"""Sanal bakiye + açık/kapalı pozisyonlar — Postgres'teki `portfolio`/
`positions`/`trades` tablolarına karşı okur/yazar. `kripto-trading-bot/
engine/portfolio.py`'daki davranış sözleşmesiyle aynı (yetersiz bakiye/
limit aşımında `(False, insan-okunur sebep)` döner), ama JSON dosyası
yerine bu tablolara karşı çalışır; kilit mekanizması yok çünkü motor günde
bir kez, tek process'te sıralı çalışıyor (kripto bot'un paralel-sembol
işleme ihtiyacı burada yok)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trading.config import TradingConfig
from trading.costs import calculate_fee, check_position_size


@dataclass(frozen=True)
class Position:
    symbol_id: int
    entry_price: float
    quantity: float
    entry_prob_up: float
    opened_at: str
    entry_fee: float


@dataclass(frozen=True)
class ClosedTrade:
    symbol_id: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees_paid: float
    net_pnl: float
    exit_reason: str
    opened_at: str
    closed_at: str


@dataclass(frozen=True)
class PortfolioState:
    balance: float
    starting_balance: float
    open_positions: list[Position]


def ensure_portfolio(conn, starting_balance: float) -> None:
    """Tek satırlık portfolio kaydını yoksa oluşturur (idempotent)."""
    row = conn.execute("SELECT id FROM portfolio WHERE id = 1").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO portfolio (id, starting_balance, balance) VALUES (1, ?, ?)",
            (starting_balance, starting_balance),
        )


def _row_to_position(row) -> Position:
    return Position(
        symbol_id=int(row["symbol_id"]),
        entry_price=float(row["entry_price"]),
        quantity=float(row["quantity"]),
        entry_prob_up=float(row["entry_prob_up"]),
        opened_at=str(row["opened_at"]),
        entry_fee=float(row["entry_fee"]),
    )


def get_state(conn) -> PortfolioState:
    row = conn.execute(
        "SELECT starting_balance, balance FROM portfolio WHERE id = 1"
    ).fetchone()
    positions_rows = conn.execute(
        "SELECT symbol_id, entry_price, quantity, entry_prob_up, opened_at, entry_fee FROM positions"
    ).fetchall()
    return PortfolioState(
        balance=float(row["balance"]),
        starting_balance=float(row["starting_balance"]),
        open_positions=[_row_to_position(p) for p in positions_rows],
    )


def get_open_position(conn, symbol_id: int) -> Position | None:
    row = conn.execute(
        "SELECT symbol_id, entry_price, quantity, entry_prob_up, opened_at, entry_fee "
        "FROM positions WHERE symbol_id = ?",
        (symbol_id,),
    ).fetchone()
    return _row_to_position(row) if row else None


def current_price(conn, symbol_id: int, fallback: float) -> float:
    """En son kapanış fiyatı, yoksa `fallback`'e düşer (ör. henüz fiyat
    çekilmemiş yeni bir sembol) — `/api/portfolio` ve snapshot kaydı bunu paylaşır."""
    row = conn.execute(
        "SELECT close FROM prices_daily WHERE symbol_id = ? ORDER BY date DESC LIMIT 1",
        (symbol_id,),
    ).fetchone()
    return float(row["close"]) if row else fallback


def positions_market_value(conn, positions: list[Position]) -> float:
    return sum(current_price(conn, p.symbol_id, p.entry_price) * p.quantity for p in positions)


def record_snapshot(conn, snapshot_date: date) -> None:
    """Günün sonunda toplam portföy değerini (nakit + açık pozisyonlar) kaydeder
    — dashboard'daki 'zaman içinde ne kadar büyüdü' eğrisi bu tablodan gelir.
    Aynı günde tekrar çalıştırılırsa (elle yeniden tetikleme) o günün satırını
    üzerine yazar, yinelenen satır oluşturmaz."""
    state = get_state(conn)
    positions_value = positions_market_value(conn, state.open_positions)
    total_value = state.balance + positions_value
    conn.execute(
        """
        INSERT INTO portfolio_snapshots (snapshot_date, balance, positions_value, total_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (snapshot_date) DO UPDATE SET
            balance = excluded.balance,
            positions_value = excluded.positions_value,
            total_value = excluded.total_value
        """,
        (
            snapshot_date.isoformat(),
            round(state.balance, 2),
            round(positions_value, 2),
            round(total_value, 2),
        ),
    )


def buy(
    conn,
    symbol_id: int,
    price: float,
    prob_up: float,
    decision_date: date,
    cfg: TradingConfig,
) -> tuple[bool, str]:
    state = get_state(conn)

    if len(state.open_positions) >= cfg.max_open_positions:
        return False, f"Açık pozisyon limiti doldu ({cfg.max_open_positions})"

    position_value = state.balance * cfg.max_position_pct
    size_check = check_position_size(position_value, cfg)
    if not size_check.allowed:
        return False, size_check.reason

    fee = calculate_fee(position_value, cfg)
    total_cost = position_value + fee.total_fee
    if total_cost > state.balance:
        return False, (
            f"Sanal bakiye yetersiz: gereken {total_cost:.2f} TL, "
            f"mevcut {state.balance:.2f} TL"
        )

    currently_locked = sum(p.entry_price * p.quantity for p in state.open_positions)
    # `state.starting_balance` DEĞİL, mevcut toplam varlık (nakit + açık
    # pozisyonların maliyet bazı) — sabit bir referansa bölünürse, bakiye o
    # referansın biraz üzerine çıktığı an (strateji kâr ettiğinde) oran
    # kalıcı olarak sınırı aşar ve portföy bir daha ASLA yeni pozisyon
    # açamaz (walk-forward backtest'te gerçek veriyle tespit edildi).
    current_equity = state.balance + currently_locked
    projected_pct = (currently_locked + position_value) / current_equity * 100
    limit_pct = cfg.max_portfolio_exposure_pct * 100
    if projected_pct > limit_pct:
        return False, (
            f"Toplam portföy riski aşılırdı: %{projected_pct:.1f} "
            f"(sınır %{limit_pct:.1f})"
        )

    quantity = position_value / price
    new_balance = state.balance - total_cost

    conn.execute("UPDATE portfolio SET balance = ?, updated_at = NOW() WHERE id = 1", (new_balance,))
    conn.execute(
        "INSERT INTO positions (symbol_id, entry_price, quantity, entry_prob_up, opened_at, entry_fee) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (symbol_id, price, quantity, prob_up, decision_date.isoformat(), fee.total_fee),
    )
    return True, (
        f"Alındı: sembol {symbol_id} @ {price:.4f} x {quantity:.6f} "
        f"(ücret {fee.total_fee:.2f} TL)"
    )


def sell(
    conn,
    symbol_id: int,
    price: float,
    exit_reason: str,
    decision_date: date,
    cfg: TradingConfig,
) -> ClosedTrade | None:
    position = get_open_position(conn, symbol_id)
    if position is None:
        return None

    gross_proceeds = price * position.quantity
    fee = calculate_fee(gross_proceeds, cfg)
    net_proceeds = gross_proceeds - fee.total_fee

    entry_cost = position.entry_price * position.quantity
    gross_pnl = gross_proceeds - entry_cost
    total_fees = position.entry_fee + fee.total_fee
    net_pnl = gross_pnl - total_fees

    state = get_state(conn)
    new_balance = state.balance + net_proceeds

    conn.execute("UPDATE portfolio SET balance = ?, updated_at = NOW() WHERE id = 1", (new_balance,))
    conn.execute("DELETE FROM positions WHERE symbol_id = ?", (symbol_id,))
    conn.execute(
        """
        INSERT INTO trades
            (symbol_id, entry_price, exit_price, quantity, gross_pnl, fees_paid, net_pnl,
             exit_reason, opened_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol_id, position.entry_price, price, position.quantity,
            round(gross_pnl, 2), round(total_fees, 2), round(net_pnl, 2),
            exit_reason, position.opened_at, decision_date.isoformat(),
        ),
    )
    return ClosedTrade(
        symbol_id=symbol_id,
        entry_price=position.entry_price,
        exit_price=price,
        quantity=position.quantity,
        gross_pnl=round(gross_pnl, 2),
        fees_paid=round(total_fees, 2),
        net_pnl=round(net_pnl, 2),
        exit_reason=exit_reason,
        opened_at=position.opened_at,
        closed_at=decision_date.isoformat(),
    )
