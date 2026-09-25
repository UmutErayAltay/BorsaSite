"""Karşılaştırma referansı (benchmark) eğrileri — SAF fonksiyonlar, hiçbir
I/O yok. Komisyon kullanmayan buy-and-hold eğrileri, strateji
performansının ölçüleceği altın standarttır."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

PricePoint = tuple[str, float]


def buy_and_hold_curve(
    prices: Sequence[PricePoint],
    starting_balance: float,
) -> list[PricePoint]:
    """İlk gün kapanışıyla `starting_balance`'ın tamamı hisseye çevrilmiş
    (komisyon yok) ve sadece tutulmuş gibi her günün portföy değeri."""
    if not prices:
        return []
    first_close = prices[0][1]
    if first_close == 0:
        return []
    shares = starting_balance / first_close
    return [(date_str, shares * close) for date_str, close in prices]


def equal_weight_basket_curve(
    price_series: Mapping[str, list[PricePoint]],
    starting_balance: float,
) -> list[PricePoint]:
    """Her sembole eşit pay ayrılmış buy-and-hold eğrilerinin, TÜM
    sembollerin ortak (kesişen) tarihlerinde toplamı. Bir sembolün verisi
    bir tarihte yoksa o gün kesişim dışı kalır — ekstrapolasyon yok."""
    if not price_series:
        return []
    share = starting_balance / len(price_series)

    curves = {}
    common_dates = None
    for symbol, prices in price_series.items():
        curve = dict(buy_and_hold_curve(prices, share))
        if not curve:
            return []
        curves[symbol] = curve
        dates = set(curve)
        common_dates = dates if common_dates is None else common_dates & dates

    if not common_dates:
        return []
    return [
        (date_str, sum(curves[symbol][date_str] for symbol in curves))
        for date_str in sorted(common_dates)
    ]