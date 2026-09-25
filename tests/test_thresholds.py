import numpy as np

from backtest.thresholds import select_buy_threshold


def test_threshold_uses_validation_only_and_returns_trade_count():
    result = select_buy_threshold(
        np.array([0.51, 0.55, 0.70, 0.90]),
        np.array([0, 1, 1, 1]),
        thresholds=[0.5, 0.7, 0.9],
    )
    assert result["threshold"] in {0.7, 0.9}
    assert result["trade_count"] >= 1


def test_raises_when_no_threshold_meets_min_trades():
    try:
        select_buy_threshold(
            np.array([0.51, 0.55]),
            np.array([0, 1]),
            thresholds=[0.9],
            min_trades=5,
        )
        assert False, "beklenen ValueError fırlatılmadı"
    except ValueError:
        pass


def test_rejects_mismatched_lengths():
    try:
        select_buy_threshold(np.array([0.5, 0.6]), np.array([0]))
        assert False, "beklenen ValueError fırlatılmadı"
    except ValueError:
        pass
