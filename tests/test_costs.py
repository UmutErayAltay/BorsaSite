from dataclasses import replace

from trading.config import TradingConfig
from trading.costs import calculate_fee, check_expected_edge, check_position_size

CFG = TradingConfig(
    starting_balance=100000.0,
    buy_threshold=0.62,
    sell_threshold=0.50,
    max_hold_days=10,
    max_open_positions=8,
    max_position_pct=0.15,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=5.0,
    min_position_value_try=500.0,
)


def test_calculate_fee_normal_trade():
    result = calculate_fee(10000.0, CFG)
    # komisyon: 10000 * 0.05% = 5.0 TL; BSMV: 5.0 * %5 = 0.25 TL
    assert result.commission == 5.0
    assert result.bsmv == 0.25
    assert result.total_fee == 5.25


def test_calculate_fee_applies_minimum_commission():
    # 100 TL'lik işlemde komisyon 0.05 TL + BSMV ~0.0025 TL — asgari 5 TL'nin altında
    result = calculate_fee(100.0, CFG)
    assert result.total_fee == 5.0


def test_check_position_size_rejects_below_minimum():
    check = check_position_size(499.0, CFG)
    assert check.allowed is False
    assert "499" in check.reason


def test_check_position_size_allows_at_minimum():
    check = check_position_size(500.0, CFG)
    assert check.allowed is True


# --- Faz 6/7: beklenen edge (maliyet + güvenlik payı) kontrolü ---------------
# Pozisyon 5000 TL, round-trip ücret 2 x 5.25 = 10.50 TL, min_expected_edge_pct
# %1 → 50 TL güvenlik payı → gereken toplam 60.50 TL.


def _edge_cfg(min_expected_edge_pct: float) -> TradingConfig:
    return replace(CFG, min_expected_edge_pct=min_expected_edge_pct)


def test_check_expected_edge_is_noop_when_disabled_by_default():
    """min_expected_edge_pct=0.0 (varsayılan) → kontrol KAPALI: beklenen kar
    sıfır/negatif olsa bile hiçbir aday reddedilmez (geriye uyumluluk)."""
    assert CFG.min_expected_edge_pct == 0.0
    for expected_return_pct in (0.0, -0.05):
        check = check_expected_edge(expected_return_pct, 5000.0, 10.50, CFG)
        assert check.allowed is True
    assert "kapalı" in check.reason


def test_check_expected_edge_is_noop_for_negative_threshold():
    """Ayar verilmiş ama negatifse de kapalı sayılır (<= 0 koruması)."""
    check = check_expected_edge(-0.5, 5000.0, 10.50, _edge_cfg(-0.5))
    assert check.allowed is True


def test_check_expected_edge_rejects_when_expected_profit_below_costs():
    """%0 beklenen getiri: kâr 0 TL, gereken 60.50 TL → red, sebep rakamları
    içerir (log'da insanın okuyup doğrulayabileceği biçimde)."""
    check = check_expected_edge(0.0, 5000.0, 10.50, _edge_cfg(0.01))

    assert check.allowed is False
    assert "beklenen kâr 0.00 TL" in check.reason
    assert "maliyet+pay 60.50 TL" in check.reason
    assert "round-trip ücret 10.50 TL" in check.reason


def test_check_expected_edge_rejects_when_profit_covers_fee_but_not_safety_margin():
    """Bacak başına 25 TL (toplam 50 TL) kâr, round-trip ücreti 10.50 TL'yi
    aşıyor ama %1'lik güvenlik payını (50 TL) karşılamıyor → hâlâ red."""
    check = check_expected_edge(0.01, 5000.0, 10.50, _edge_cfg(0.01))

    assert check.allowed is False
    assert "50.00 TL < maliyet+pay 60.50 TL" in check.reason


def test_check_expected_edge_allows_when_expected_profit_clears_margin():
    """%2 beklenen getiri → 100 TL kâr > 60.50 TL gereken → al."""
    check = check_expected_edge(0.02, 5000.0, 10.50, _edge_cfg(0.01))

    assert check.allowed is True
    assert "edge yeterli" in check.reason
