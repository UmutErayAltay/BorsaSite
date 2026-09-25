"""Backtest performans metrikleri — SAF fonksiyonlar, hiçbir I/O yok.
`trading/costs.py` disiplininin aynısı: finansal mantık tek, izole, kolay
test edilir bir yerde durur. Boş girdide anlamsız olan metrikler `None`
döndürür, asla exception fırlatmaz."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

EquityPoint = tuple[str, float]


@dataclass(frozen=True)
class Trade:
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees_paid: float
    net_pnl: float
    opened_at: str
    closed_at: str


def _trade_field(t: Trade | dict, name: str) -> float:
    """`Trade` dataclass ya da aynı isimli alanları taşıyan dict'ten değer okur
    (`trading/portfolio.ClosedTrade` de bu sözleşmeyi karşılar)."""
    if isinstance(t, Mapping):
        return t[name]
    return getattr(t, name)


def _values(curve: Sequence[EquityPoint]) -> list[float]:
    return [v for _, v in curve]


def daily_returns(equity_curve: Sequence[EquityPoint]) -> list[float]:
    """Ardışık günlerin yüzde getirisi; ilk gün için değer üretilmez,
    sonuç listesi girdiden bir eksik uzunluktadır."""
    values = _values(equity_curve)
    if len(values) < 2:
        return []
    return [(values[i] - values[i - 1]) / values[i - 1] * 100.0 for i in range(1, len(values))]


def total_return_pct(equity_curve: Sequence[EquityPoint]) -> float | None:
    """Toplam yüzde getiri; tek veri noktası ya da boş girdide `None`."""
    values = _values(equity_curve)
    if len(values) < 2 or values[0] == 0:
        return None
    return (values[-1] - values[0]) / values[0] * 100.0


def annualized_return_pct(
    equity_curve: Sequence[EquityPoint],
    trading_days_per_year: int = 252,
) -> float | None:
    """CAGR formülüyle yıllıklandırılmış getiri; gerçek gün sayısı
    `len(equity_curve)` üzerinden yıllandırılır. İlk değer sıfır ya da
    negatifse üs alma tanımsız olduğundan `None` döner."""
    values = _values(equity_curve)
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
        return None
    years = len(values) / trading_days_per_year
    if years == 0:
        return None
    return (math.pow(values[-1] / values[0], 1.0 / years) - 1.0) * 100.0


def max_drawdown_pct(equity_curve: Sequence[EquityPoint]) -> dict:
    """En yüksek tepe noktasından en derin çöküşe kadar olan yüzde kayıp
    (pozitif sayı, örn. 12.5 = %12.5 kayıp). Boş girdide `0.0` ve `None`
    tarihlerle döner."""
    if not equity_curve:
        return {"max_drawdown_pct": 0.0, "peak_date": None, "trough_date": None}

    peak_value = equity_curve[0][1]
    peak_date = equity_curve[0][0]
    max_dd = 0.0
    # peak_date_at_max is the peak that produced the recorded max drawdown —
    # NOT necessarily the running peak_date, which a later (higher, irrelevant)
    # peak would otherwise overwrite even though it never actually drew down.
    peak_date_at_max = equity_curve[0][0]
    trough_date = equity_curve[0][0]

    for date_str, value in equity_curve:
        if value > peak_value:
            peak_value = value
            peak_date = date_str
        if peak_value > 0:
            dd = (peak_value - value) / peak_value
            if dd > max_dd:
                max_dd = dd
                trough_date = date_str
                peak_date_at_max = peak_date

    return {
        "max_drawdown_pct": max_dd * 100.0,
        "peak_date": peak_date_at_max,
        "trough_date": trough_date,
    }


def _downside_deviation(excess_returns: list[float]) -> float | None:
    """Standart downside deviation: TÜM gözlemler üzerinden `min(0, r)`'nin
    RMS'i (kayıp olmayan günler 0 katkı verir) — sadece negatif alt kümenin
    KENDİ ortalamasından sapması DEĞİL (tek bir negatif gün varsa o formül
    her zaman 0 verir, halbuki gerçek risk sıfır değildir). Negatif getiri
    hiç yoksa `None` — payda 0 olan bir oran matematiksel olarak anlamsızdır."""
    if not any(r < 0 for r in excess_returns):
        return None
    squared_downside = [min(0.0, r) ** 2 for r in excess_returns]
    return math.sqrt(statistics.fmean(squared_downside))


def _excess_returns(
    equity_curve: Sequence[EquityPoint],
    risk_free_rate: float,
    trading_days_per_year: int,
) -> list[float]:
    """Günlük risk-free oranı düşülmüş yüzde getiriler; günlük fazlalık
    getiri `risk_free_rate / trading_days_per_year`'dır."""
    daily_rf = risk_free_rate / trading_days_per_year
    return [r - daily_rf for r in daily_returns(equity_curve)]


def sharpe_ratio(
    equity_curve: Sequence[EquityPoint],
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> float | None:
    """Günlük getirilerin ortalaması/standart sapması, `sqrt(gün)` ile
    yıllıklandırılmış. Standart sapma 0 ise `None` (ZeroDivisionError yok)."""
    excess = _excess_returns(equity_curve, risk_free_rate, trading_days_per_year)
    if len(excess) < 2:
        return None
    std = statistics.pstdev(excess)
    # A near-constant series never has an exact 0.0 pstdev in float arithmetic
    # (e.g. compounding 1%/day accrues rounding noise) — an exact `== 0` check
    # lets that noise through and produces an absurd, meaningless ratio.
    if math.isclose(std, 0.0, abs_tol=1e-9):
        return None
    return statistics.fmean(excess) / std * math.sqrt(trading_days_per_year)


def sortino_ratio(
    equity_curve: Sequence[EquityPoint],
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> float | None:
    """Sharpe ile aynı ama paydada downside deviation (bkz. `_downside_deviation`).
    Negatif getiri hiç yoksa `None`."""
    excess = _excess_returns(equity_curve, risk_free_rate, trading_days_per_year)
    if len(excess) < 2:
        return None
    downside = _downside_deviation(excess)
    if downside is None or math.isclose(downside, 0.0, abs_tol=1e-9):
        return None
    return statistics.fmean(excess) / downside * math.sqrt(trading_days_per_year)


def profit_factor(trades: Sequence[Trade | dict]) -> float | None:
    """Brüt kâr toplamının brüt kayıp toplamının mutlak değerine oranı;
    kayıp toplamı 0 ise `None` (payda 0)."""
    gross = [_trade_field(t, "gross_pnl") for t in trades]
    wins = sum(g for g in gross if g > 0)
    losses = sum(g for g in gross if g < 0)
    if losses == 0:
        return None
    return wins / abs(losses)


def win_rate_pct(trades: Sequence[Trade | dict]) -> float | None:
    """net_pnl > 0 olan işlemlerin yüzdesi."""
    if not trades:
        return None
    wins = sum(1 for t in trades if _trade_field(t, "net_pnl") > 0)
    return wins / len(trades) * 100.0


def expectancy(trades: Sequence[Trade | dict]) -> float | None:
    """İşlem başına ortalama net_pnl."""
    if not trades:
        return None
    return statistics.fmean(_trade_field(t, "net_pnl") for t in trades)


def average_holding_days(trades: Sequence[Trade | dict]) -> float | None:
    """Giriş-çıkış gün farklarının ortalaması (ISO tarih string'leri
    `date.fromisoformat` ile parse edilir)."""
    if not trades:
        return None
    days = []
    for t in trades:
        opened = date.fromisoformat(_trade_field(t, "opened_at"))
        closed = date.fromisoformat(_trade_field(t, "closed_at"))
        days.append((closed - opened).days)
    return statistics.fmean(days)


def turnover_pct(
    trades: Sequence[Trade | dict],
    equity_curve: Sequence[EquityPoint],
) -> float | None:
    """Alış+satış işlem hacminin ortalama portföy değerine oranı;
    ortalama değer 0 ise `None` (payda 0)."""
    if not trades:
        return None
    values = _values(equity_curve)
    if not values:
        return None
    avg_value = statistics.fmean(values)
    if avg_value == 0:
        return None
    volume = sum(
        abs(_trade_field(t, "entry_price") * _trade_field(t, "quantity"))
        + abs(_trade_field(t, "exit_price") * _trade_field(t, "quantity"))
        for t in trades
    )
    return volume / avg_value * 100.0


def total_fees(trades: Sequence[Trade | dict]) -> float:
    """Ödenen toplam komisyon; boş girdide `0.0`."""
    return float(sum(_trade_field(t, "fees_paid") for t in trades))


def average_fee_per_trade(trades: Sequence[Trade | dict]) -> float | None:
    """İşlem başına ortalama komisyon; işlem yoksa `None`."""
    if not trades:
        return None
    return statistics.fmean(_trade_field(t, "fees_paid") for t in trades)


def summarize(
    equity_curve: Sequence[EquityPoint],
    trades: Sequence[Trade | dict],
    starting_balance: float,
) -> dict:
    """Tüm metrikleri tek dict'te toplar; anahtar adları fonksiyon
    adlarıyla birebir aynıdır (`_pct` ekleri korunur)."""
    return {
        "daily_returns": daily_returns(equity_curve),
        "total_return_pct": total_return_pct(equity_curve),
        "annualized_return_pct": annualized_return_pct(equity_curve),
        "max_drawdown_pct": max_drawdown_pct(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "sortino_ratio": sortino_ratio(equity_curve),
        "profit_factor": profit_factor(trades),
        "win_rate_pct": win_rate_pct(trades),
        "expectancy": expectancy(trades),
        "average_holding_days": average_holding_days(trades),
        "turnover_pct": turnover_pct(trades, equity_curve),
        "total_fees": total_fees(trades),
        "average_fee_per_trade": average_fee_per_trade(trades),
        "starting_balance": starting_balance,
        "ending_balance": _values(equity_curve)[-1] if equity_curve else None,
        "num_trades": len(trades),
    }