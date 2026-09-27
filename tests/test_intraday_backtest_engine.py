"""`backtest/intraday_engine.py` integration tests against the real local
Postgres (synthetic bars, no network) — mirrors tests/test_backtest_engine.py.

DİKKAT — `tests/conftest.py::committed_conn` bu dosyada KULLANILMIYOR: onun
teardown'ı tüm tabloları TRUNCATE eder, yani veritabanındaki gerçek (ve
pahalı) 547K satırlık intraday verisini siler. Aşağıdaki `intraday_conn`
deseni `tests/test_intraday_dataset.py`'dekiyle aynıdır ama teardown'da
YALNIZCA bu dosyanın kullandığı test ticker'larını temizler.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
import pytest

from backtest.costs import BacktestCostConfig
from backtest.intraday_engine import run_intraday_backtest
from pipeline.db import (
    ConnWrapper,
    get_database_url,
    init_schema,
    upsert_intraday_prices,
    upsert_symbol,
)
from pipeline.intraday_dataset import INTRADAY_FEATURE_COLUMNS
from trading.config import TradingConfig

INTERVAL = "1h"

# Yalnız bu dosyada üretilen ticker'lar — temizlik bunlarla sınırlıdır
# (test_intraday_dataset.py'nin TEST1.IS/KEEP.IS, test_intraday_train_model.py'nin
# TRAIN1.IS/TRAIN2.IS ile çakışmaz).
_SEED_TICKERS = ("BT1.IS", "BT2.IS")

# `config/intraday_model.yaml::features.min_history_bars` eşiğini geçmek için
# gereken bar sayısı (200) — sembol bu eşiğin altında kalırsa dataset'e hiç girmez.
_SEED_DAYS = 45


class _FixedProbModel:
    """joblib-picklable stand-in for a trained classifier: always reports the
    same prob_up, so the engine's OWN decision loop can be tested without
    training a real model."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


class _FixedReturnModel:
    """`pipeline/intraday_train_model.py`'deki `magnitude_model` yerine geçen
    sabit getiri (oran) tahmin eden stand-in."""

    def __init__(self, expected_return: float):
        self.expected_return = expected_return

    def predict(self, X):
        return np.full(len(X), self.expected_return)


def _seed_symbol_with_intraday_bars(conn, ticker: str, *, days: int, bars_per_day: int = 5) -> int:
    """`tests/test_intraday_dataset.py` ile aynı üretim deseni. Zig-zag fiyat
    şart: `compute_rsi` düşüş olmayan seride `avg_loss=0` -> NaN üretir ve
    sembol dataset'e hiç giremez."""
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    rows = []
    price = 100.0
    for d in range(days):
        day = pd.Timestamp("2026-01-05", tz="Europe/Istanbul") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            price += 0.1 if b % 2 == 0 else -0.05
            ts = day + pd.Timedelta(hours=10 + b)
            rows.append((ts.isoformat(), price, price + 0.5, price - 0.5, price, 1000.0 + b))
    upsert_intraday_prices(conn, symbol_id, INTERVAL, iter(rows))
    return symbol_id


@pytest.fixture(scope="module")
def intraday_conn():
    """Modül kapsamlı: tüm testler AYNI sembolleri kullanıyor, dolayısıyla veri
    bir kez yazılıp bir kez okunuyor (dataset'in gerçek fiyat tablosunu
    taraması ~15s sürüyor)."""
    raw = psycopg.connect(get_database_url())
    wrapper = ConnWrapper(raw)
    init_schema(wrapper)
    raw.commit()  # kilitleri bırak, gövde başlasın
    try:
        yield wrapper
        raw.commit()
    finally:
        raw.rollback()
        cur = raw.cursor()
        cur.execute("SELECT id FROM symbols WHERE ticker = ANY(%s)", (list(_SEED_TICKERS),))
        ids = [r[0] for r in cur.fetchall()]
        if ids:
            cur.execute("DELETE FROM prices_intraday WHERE symbol_id = ANY(%s)", (ids,))
            cur.execute("DELETE FROM symbols WHERE id = ANY(%s)", (ids,))
        raw.commit()
        raw.close()


@pytest.fixture(scope="module")
def intraday_rows(intraday_conn) -> pd.DataFrame:
    """Gerçek `build_intraday_dataset()` çıktısı, YALNIZCA test sembollerine
    kırpılmış hali.

    Kırpma bir hız/ determinizm önlemi: dataset gerçek 101 sembolün 545K barını
    tarıyor ve sahte model (prob_up=0.9) bunların hepsinde alım üretirdi —
    testler kendi sembollerine odaklanamazdı. Motorun üzerinde çalıştığı
    feature/hedef/gün-sınırı kodu ve okuduğu veri tamamen GERÇEK; sadece
    hangi satırların oynandığı daraltılıyor."""
    for ticker in _SEED_TICKERS:
        _seed_symbol_with_intraday_bars(intraday_conn, ticker, days=_SEED_DAYS)
    intraday_conn.commit()

    from pipeline.intraday_dataset import build_intraday_dataset

    df = build_intraday_dataset(interval=INTERVAL, require_target=False)
    rows = df[df["ticker"].isin(_SEED_TICKERS)].copy()
    rows["_day"] = rows["ts"].dt.tz_convert(None).dt.normalize()
    return rows


@pytest.fixture
def scoped_dataset(monkeypatch, intraday_rows):
    """`run_intraday_backtest`'in kendi bağlantısıyla dataset'i kurması yerine
    (daha yavaş olurdu) test sembollerinin GERÇEK feature satırlarını verir."""
    import backtest.intraday_engine as engine

    monkeypatch.setattr(engine, "build_intraday_dataset", lambda **kwargs: intraday_rows)
    return intraday_rows


def _observable_days(rows: pd.DataFrame, min_bars: int = 5) -> list[pd.Timestamp]:
    """Feature ısınması (20 bar) sonrası tamamen GÖZLEMLENEBİLİR günler."""
    counts = rows.groupby("_day").size()
    return sorted(counts[counts >= min_bars].index)


def _dump_model(tmp_path: Path, prob: float, expected_return: float | None) -> Path:
    """Faz 6/7 sonrası bundle: `magnitude_model` anahtarı da içerir.
    `expected_return=None` ise ESKİ (magnitude_model'siz) bundle üretilir."""
    bundle: dict = {
        "model": _FixedProbModel(prob),
        "features": INTRADAY_FEATURE_COLUMNS,
        "version": "test",
    }
    if expected_return is not None:
        bundle["magnitude_model"] = _FixedReturnModel(expected_return)
    model_path = tmp_path / "intraday_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return model_path


CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.55,
    sell_threshold=0.50,
    max_hold_days=5,  # intraday'de YOK sayılır — gün sonunda zorla kapanır
    max_open_positions=3,
    max_position_pct=0.5,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=1.0,
    min_position_value_try=100.0,
)


def test_positions_never_survive_overnight(scoped_dataset, tmp_path):
    """EN ÖNEMLİ test — model gece sıçraması hakkında hiçbir fikir üretmiyor, bu
    yüzden strateji GECE POZİSYON TUTMAMALI.

    Sahte model her barda prob_up=0.9 döndürür (alım eşiğinin üstünde), bu yüzden
    günün İLK bar'ında alım olmalı ve o pozisyon O GÜN içinde kapanmalı."""
    rows = scoped_dataset
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=0.02)

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    buys = [d for d in result.decisions if d["action"] == "al"]
    assert buys, "prob_up eşik üstündeyken hiç alım olmadı"

    # 1) Her alım, AYNI günde "gun_sonu" gerekçeli bir satışla kapanmalı.
    day_end_closes = [
        d for d in result.decisions
        if d["action"] == "sat" and "gun_sonu" in d["reason"]
    ]
    assert day_end_closes, "hiçbir pozisyon gün sonunda kapatılmadı"
    buy_days = {d["date"][:10] for d in buys}
    close_days = {d["date"][:10] for d in day_end_closes}
    for day in buy_days:
        assert day in close_days, f"{day} günü alım var ama gün sonu satışı yok"

    # 2) Kapanan HER işlem aynı takvim gününde açılıp kapanmalı — dolaylı olarak
    #    2. güne devralınan pozisyon YOK.
    assert result.trades, "hiçbir işlem kapanmadı"
    for trade in result.trades:
        opened = pd.Timestamp(trade.opened_at).date()
        closed = pd.Timestamp(trade.closed_at).date()
        assert opened == closed, (
            f"{trade.symbol} GECE POZİSYONU TUTMUŞ: {opened} -> {closed}"
        )

    # 3) prob sabit ve eşiklerin üstünde olduğu için başka bir çıkış sebebi
    #    (stop_loss/take_profit/prob_düştü) ÜRETİLMEMELİ.
    assert {t.exit_reason for t in result.trades} == {"gun_sonu_kapanis"}


def test_no_position_is_carried_into_the_next_day(scoped_dataset, tmp_path):
    """Bir sonraki günün İLK bar'ında portföy boş olmalı: o günün ilk kararı
    asla "sat" olamaz (satış, aynı güne ait bir alım gerektirir)."""
    rows = scoped_dataset
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=0.02)

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    days = _observable_days(rows)
    assert len(days) >= 2, "test için en az iki gözlemlenebilir gün gerekiyor"
    first_bar_of_day = {
        day: min(ts for ts in rows.loc[rows["_day"] == day, "ts"])
        for day in days
    }

    checked = 0
    for decision in result.decisions:
        day = decision["date"][:10]
        if day not in {str(d.date()) for d in days}:
            continue
        if decision["date"] != first_bar_of_day[pd.Timestamp(day)].isoformat():
            continue
        assert decision["action"] != "sat", (
            f"{decision['date']} ilk barında satış var → pozisyon GECEDEN devralınmış"
        )
        checked += 1
    assert checked, "hiçbir günün ilk barına karar düşmedi"


def test_legacy_bundle_without_magnitude_model_does_not_crash(scoped_dataset, tmp_path):
    """`magnitude_model` anahtarı olmayan eski bundle: `.get()` ile None'a
    düşer, `expected_return` 0.0 kabul edilir (backtest parity deseni)."""
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=None)
    assert "magnitude_model" not in joblib.load(model_path)

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    buys = [d for d in result.decisions if d["action"] == "al"]
    assert buys and result.trades
    assert all(
        pd.Timestamp(t.opened_at).date() == pd.Timestamp(t.closed_at).date()
        for t in result.trades
    )


def test_min_expected_edge_blocks_candidate_via_existing_buy_path(scoped_dataset, tmp_path):
    """Edge-after-cost filtresinin intraday'e YENİ kod yazmadan ulaştığının
    kanıtı: aynı `BacktestPortfolio.buy()` çağrısı, `min_expected_edge_pct > 0`
    iken düşük beklenen getirili adayı reddeder (canlı `trading/portfolio.py`
    ile aynı davranış)."""
    edge_cfg = replace(CFG, min_expected_edge_pct=0.01)
    # %0.5 beklenen getiri, %1 güvenlik payını + round-trip ücretini karşılamaz.
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=0.005)

    result = run_intraday_backtest(
        trading_cfg=edge_cfg, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    rejected = [d for d in result.decisions if d["action"] == "red"]
    assert rejected, "düşük beklenen getiri reddedilmeliydi"
    assert all("beklenen kâr" in d["reason"] for d in rejected)
    assert not any(d["action"] == "al" for d in result.decisions)
    assert not result.trades
    assert all(value == edge_cfg.starting_balance for _, value in result.equity_curve)


def test_low_prob_never_buys(scoped_dataset, tmp_path):
    model_path = _dump_model(tmp_path, prob=0.1, expected_return=0.02)

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    assert not result.trades
    assert all(d["action"] != "al" for d in result.decisions)
    assert all(value == CFG.starting_balance for _, value in result.equity_curve)
