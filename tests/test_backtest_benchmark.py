import pytest

from backtest.benchmark import buy_and_hold_curve, equal_weight_basket_curve


def test_buy_and_hold_curve_hand_computed():
    prices = [("d1", 100.0), ("d2", 110.0), ("d3", 90.0)]
    curve = buy_and_hold_curve(prices, 1000.0)
    # 10 shares bought at 100
    assert curve == [("d1", 1000.0), ("d2", 1100.0), ("d3", 900.0)]


def test_buy_and_hold_curve_empty():
    assert buy_and_hold_curve([], 1000.0) == []


def test_buy_and_hold_curve_zero_first_price():
    assert buy_and_hold_curve([("d1", 0.0)], 1000.0) == []


def test_equal_weight_basket_curve_hand_computed():
    price_series = {
        "A": [("d1", 100.0), ("d2", 110.0), ("d3", 121.0)],
        "B": [("d1", 50.0), ("d2", 55.0), ("d3", 60.5)],
    }
    curve = equal_weight_basket_curve(price_series, 1000.0)
    # 500 TL each: A -> 5 shares, B -> 10 shares
    # d1: 5*100 + 10*50 = 500+500=1000
    # d2: 5*110 + 10*55 = 550+550=1100
    # d3: 5*121 + 10*60.5 = 605+605=1210
    assert curve == pytest.approx([("d1", 1000.0), ("d2", 1100.0), ("d3", 1210.0)])


def test_equal_weight_basket_curve_uses_only_common_dates():
    price_series = {
        "A": [("d1", 100.0), ("d2", 110.0), ("d3", 120.0)],
        "B": [("d2", 50.0), ("d3", 55.0)],  # missing d1
    }
    curve = equal_weight_basket_curve(price_series, 1000.0)
    dates = [d for d, _ in curve]
    assert dates == ["d2", "d3"]


def test_equal_weight_basket_curve_empty():
    assert equal_weight_basket_curve({}, 1000.0) == []
