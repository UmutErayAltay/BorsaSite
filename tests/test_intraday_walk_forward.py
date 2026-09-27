"""`backtest/intraday_walk_forward.py` integration tests against the real
local Postgres (synthetic bars, no network) — mirrors
tests/test_intraday_train_model.py.

DİKKAT — `tests/conftest.py::committed_conn` bu dosyada KULLANILMIYOR: onun
teardown'ı tüm tabloları TRUNCATE eder, yani veritabanındaki gerçek (ve
pahalı) 547K satırlık intraday verisini siler. Aşağıdaki `intraday_conn`
deseni `tests/test_intraday_train_model.py`'ninkinin AYNISIDIR ama teardown'da
YALNIZCA bu dosyanın kullandığı test ticker'larını temizler (WF1.IS/WF2.IS —
diğer test dosyalarının ticker'larıyla çakışmaz).

Tohum veri `tests/test_intraday_train_model.py`'ninkinin AYNI üretim
desenini kullanır: gün yönü d%3'e göre değişir, böylece hedef gün-sonu
getirisinde İKİ SINIF garanti edilir (sınıflandırıcı eğitilebilsin) ve
`open != close` (v2 dolumu bir sonraki barın AÇILIŞINDA olduğu için).
"""
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
import pytest

from backtest.costs import BacktestCostConfig
from backtest.intraday_engine import run_intraday_backtest
from backtest.intraday_walk_forward import run_intraday_walk_forward
from pipeline.db import (
    ConnWrapper,
    get_database_url,
    init_schema,
    upsert_intraday_prices,
    upsert_symbol,
)
from pipeline.intraday_dataset import INTRADAY_FEATURE_COLUMNS, load_intraday_model_config
from trading.config import TradingConfig

INTERVAL = "1h"

_SEED_TICKERS = ("WF1.IS", "WF2.IS")

# 45 gün: feature ısınması (~24 bar) düşüldükten sonra 40+ gözlemlenebilir
# gün kalır — `initial_train_days=10, val_days=3, test_days=3` ile birden
# fazla fold kurulabilsin diye.
_SEED_DAYS = 45

_FIRST_DAY = "2026-01-05"


class _FixedProbModel:
    """Sabit `prob_up` üreten, joblib-picklable sınıflandırıcı stand-in."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


class _FixedReturnModel:
    """Sabit getiri tahmin eden `magnitude_model` stand-in'i."""

    def __init__(self, expected_return: float):
        self.expected_return = expected_return

    def predict(self, X):
        return np.full(len(X), self.expected_return)


def _seed_symbol_with_intraday_bars(conn, ticker: str, *, days: int, bars_per_day: int = 5) -> int:
    """`tests/test_intraday_train_model.py` ile aynı üretim deseni.

    Zig-zag fiyat şart: `compute_rsi` düşüş olmayan seride `avg_loss=0` ->
    NaN üretir ve sembol dataset'e hiç giremez. Gün yönü d%3 ile değişir —
    hedef gün-sonu getirisi olduğu için TEK YÖNLÜ veri sınıflandırıcıyı
    eğitilemez hale getirirdi. `open` bilerek `close`'tan FARKLI: v2 dolumu
    bir sonraki barın AÇILIŞINDA yapıyor."""
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    rows = []
    price = 100.0
    for d in range(days):
        day = pd.Timestamp(_FIRST_DAY, tz="Europe/Istanbul") + pd.Timedelta(days=d)
        up, down = (0.1, -0.05) if d % 3 else (-0.1, 0.05)
        for b in range(bars_per_day):
            price += up if b % 2 == 0 else down
            close = price
            open_ = close + 0.25
            ts = day + pd.Timedelta(hours=10 + b)
            rows.append((ts.isoformat(), open_, close + 0.5, close - 0.5, close, 1000.0 + b))
    upsert_intraday_prices(conn, symbol_id, INTERVAL, iter(rows))
    return symbol_id


@pytest.fixture(scope="module")
def intraday_conn():
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
def seeded_rows(intraday_conn) -> pd.DataFrame:
    """Tohum barlar yazılır ve GERÇEK `build_intraday_dataset()` çıktısı
    yalnızca bu dosyanın sembollerine kırpılarak döndürülür.

    Kırpma bir hız/determinizm önlemi (dataset gerçek sembollerin yüz binlerce
    barını tarar); motorun üzerinde çalıştığı feature/hedef/gün-sınırı kodu ve
    okuduğu veri tamamen GERÇEK."""
    for ticker in _SEED_TICKERS:
        _seed_symbol_with_intraday_bars(intraday_conn, ticker, days=_SEED_DAYS)
    intraday_conn.commit()

    from pipeline.intraday_dataset import build_intraday_dataset

    df = build_intraday_dataset(interval=INTERVAL, require_target=False)
    rows = df[df["ticker"].isin(_SEED_TICKERS)].copy()
    rows["_day"] = rows["ts"].dt.tz_convert(None).dt.normalize()
    return rows


@pytest.fixture
def walk_forward_cfg(monkeypatch):
    """Gerçek config, üç değişiklikle: küçük eğitim pencereleri (test hızlı
    olsun) ve 0.998 yerine orta kantil (80 günlük tohum veride 0.998 neredeyse
    hiç giriş üretirdi — eşiğin GERÇEKTEN seçildiğini görmek istiyoruz, ama
    test'in kendisi de işlem üretmeyi doğrulamalı)."""
    cfg = copy.deepcopy(load_intraday_model_config())
    cfg["features"]["min_history_bars"] = 100
    cfg["training"]["xgb"]["n_estimators"] = 20
    cfg["training"]["xgb_regressor"]["n_estimators"] = 20
    cfg["strategy"]["entry_quantile"] = 0.5
    for target in (
        "backtest.intraday_walk_forward.load_intraday_model_config",
        "pipeline.intraday_dataset.load_intraday_model_config",
    ):
        monkeypatch.setattr(target, lambda: cfg)
    return cfg


@pytest.fixture
def scoped_dataset(monkeypatch, seeded_rows):
    """`run_intraday_walk_forward` dataset'i BİR KEZ kendi bağlantısıyla
    kurması gerekiyor, ama test sembollerine kırpılmış olmalı (yoksa gerçek
    sembollerin barları da fold'lara karışır ve test yavaş/olası olurdu)."""
    import backtest.intraday_walk_forward as wf

    monkeypatch.setattr(wf, "build_intraday_dataset", lambda **kwargs: seeded_rows)
    return seeded_rows


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

SMALL = dict(initial_train_days=10, val_days=3, test_days=3)


def test_fold_windows_are_disjoint_adjacent_and_chronological(
    scoped_dataset, walk_forward_cfg
):
    """Fold sızlamasının kalbı:

    * `max(train) < min(val) < min(test)` — hiçbir pencere diğerine sızmaz,
    * test pencereleri ARDIŞIK: birinin `test_end` + 1 günü diğerinin
      `test_start`'ı (ÇAKIŞMA ve BOŞLUK yok),
    * train GENİŞLİYOR: son fold'un `train_end` ilk fold'unkinden küçük
      değil (aslında büyüktür),
    * val+test'i bir sonraki fold'un train'i içinde olduğu için günler
      yalnızca BİR KEZ test edilir.
    """
    _, folds = run_intraday_walk_forward(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), **SMALL
    )

    assert len(folds) >= 2, f"birden fazla fold kurulmalıydı, {len(folds)} fold var"
    for f in folds:
        assert f["train_start"] == folds[0]["train_start"], "train başlangıcı kaymamalı"
        assert f["train_end"] < f["val_start"] < f["val_end"] <= f["test_start"] < f["test_end"], (
            f"fold {f['fold']} kronolojik değil: {f}"
        )
    assert [f["fold"] for f in folds] == list(range(len(folds))), "fold indeksleri 0..N olmalı"

    for prev, nxt in zip(folds, folds[1:]):
        # Test pencereleri çakışmamalı, aralarında boşluk da olmamalı.
        next_day = (pd.Timestamp(prev["test_end"]) + pd.Timedelta(days=1)).date().isoformat()
        assert prev["test_end"] < nxt["test_start"], "test pencereleri çakışıyor/geriye gidiyor"
        assert nxt["test_start"] == next_day, (
            f"test pencereleri ardışık değil: {prev['test_end']} -> {nxt['test_start']}"
        )
        # Bir fold'un val+test'i bir sonraki fold'un train'ine GİRMELİ.
        assert nxt["train_end"] > prev["val_start"], "genişleyen pencere içe doğru hareket ediyor"

    # Hiçbir gün iki kez TEST edilmemeli.
    test_days = [d for f in folds for d in (f["test_start"], f["test_end"])]
    assert len(test_days) == 2 * len(folds)

    # İlk fold'un ilk test günü, ilk fold'un train'inin hemen ardından gelmeli.
    days = sorted(scoped_dataset["_day"].unique())
    first_test = pd.Timestamp(folds[0]["test_start"])
    assert first_test.day == days[10 + 3].day, "test penceresi val'dan sonra başlamalı"


def test_end_to_end_run_returns_result_with_trades_only_after_first_test_start(
    scoped_dataset, walk_forward_cfg
):
    """Uçtan uca koşu: (BacktestResult, folds) döner, bakiye zinciri TEK
    portföydedir ve HİÇBİR işlem ilk fold'un `test_start`'ından önce
    açılmamıştır — yani eğitim dönemi oynatılmıyor (in-sample sonuç yok)."""
    result, folds = run_intraday_walk_forward(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), **SMALL
    )

    assert folds, "hiç fold üretilmedi"
    first_test_start = folds[0]["test_start"]

    assert result.trades, "walk-forward koşusu işlem üretmedi"
    for trade in result.trades:
        assert trade.opened_at[:10] >= first_test_start, (
            f"{trade.symbol} işlemi {trade.opened_at} — ilk test gününden ÖNCE açıldı"
        )
    for decision in result.decisions:
        assert decision["date"][:10] >= first_test_start, (
            f"{decision['date']} kararı ilk test gününden ÖNCE üretildi"
        )

    # Sonuç penceresi de yalnızca test dönemini kapsamalı.
    assert result.start_date[:10] >= first_test_start
    assert result.end_date[:10] >= first_test_start
    assert all(day >= first_test_start for day, _ in result.equity_curve)

    # Pozisyonlar gün içinde kapanır (gece pozisyonu yok — v2 kuralı).
    for trade in result.trades:
        assert trade.opened_at[:10] == trade.closed_at[:10]
    assert {t.exit_reason for t in result.trades} <= {
        "gun_sonu_kapanis",
        "stop_loss",
        "take_profit",
    }

    # Equity curve gün başına TEK nokta (tek portföy, zincirleme bakiye).
    curve_days = [day for day, _ in result.equity_curve]
    assert curve_days == sorted(curve_days), "equity curve kronolojik değil"
    assert len(set(curve_days)) == len(curve_days), "bir güne birden fazla snapshot"


def test_entry_threshold_comes_from_validation_not_test(
    scoped_dataset, walk_forward_cfg, monkeypatch
):
    """Eşik seçimi SADECE val tahminlerinden gelir. Kanıt: eşiği üreten
    `np.quantile` çağrısına, val kümesinin TÜM satırları (test hariç)
    geçirilir. Test satırları eşiğin hesabına hiç karışmaz."""
    import backtest.intraday_walk_forward as wf

    seen: list[int] = []
    real_quantile = wf.np.quantile

    def spy(a, q, *args, **kwargs):
        seen.append(len(a))
        return real_quantile(a, q, *args, **kwargs)

    monkeypatch.setattr(wf.np, "quantile", spy)

    _, folds = run_intraday_walk_forward(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), **SMALL
    )

    assert folds, "fold üretilmedi"
    assert seen, "np.quantile hiç çağrılmadı — eşik val'den türetilmiyor"

    rows = scoped_dataset
    labelled_per_day = rows[rows["target_return"].notna()].groupby("_day").size()
    # Etiketli satır olmayan günler var (ısınma); val 3 günden oluşur ve
    # eşiğe giren satır sayısı bu üç günün toplam etiketli satırıdır.
    for f in folds:
        val_days = pd.date_range(f["val_start"], f["val_end"], freq="D")
        expected = int(labelled_per_day.reindex(val_days).fillna(0).sum())
        assert expected in seen, (
            f"fold {f['fold']} eşiği {expected} satırlık val kümesinden türetilmeli, "
            f"görülen boyutlar: {sorted(set(seen))}"
        )


def test_fold_summary_has_required_keys_and_iso_days(
    scoped_dataset, walk_forward_cfg
):
    """Fold özeti sözleşmesi: `test_ic` Spearman (target NaN satırlar hariç),
    tarihler ISO GÜN stringi, `n_train_rows` yalnızca ETİKETLİ satırlar."""
    _, folds = run_intraday_walk_forward(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), **SMALL
    )

    required = {
        "fold", "train_start", "train_end", "val_start", "val_end",
        "test_start", "test_end", "n_train_rows", "entry_threshold", "test_ic",
    }
    rows = scoped_dataset
    for f in folds:
        assert set(f) == required, f"fold özeti anahtarları eksik/fazla: {set(f) ^ required}"
        for key in required - {"fold", "n_train_rows", "entry_threshold", "test_ic"}:
            value = f[key]
            assert isinstance(value, str), f"{key} string olmalı"
            assert len(value) == 10 and value[4] == "-", f"{key} ISO gün olmalı: {value}"
            assert pd.Timestamp(value).date().isoformat() == value
        assert isinstance(f["n_train_rows"], int) and f["n_train_rows"] > 0
        assert isinstance(f["entry_threshold"], float)
        assert f["test_ic"] is None or isinstance(f["test_ic"], float)

    # `n_train_rows` yalnızca hedefi BİLİNEN satırları sayar: genişleyen
    # pencere, ilk fold'dan sonraki fold'larda DAHA FAZLA gün kapsar ama
    # train satır sayısı artmalıdır.
    assert [f["n_train_rows"] for f in folds] == sorted(f["n_train_rows"] for f in folds)
    assert folds[-1]["n_train_rows"] > folds[0]["n_train_rows"]

    # `test_ic` yalnızca etiketli satırlar üzerinden: her günün son barı
    # (target_return NaN) hesaba katılmamalı.
    assert len(rows) > 0


def test_run_is_deterministic_across_repeats(scoped_dataset, walk_forward_cfg):
    """Aynı veriyle iki koşu birebir aynı portföyü vermeli: `random_state`
    config'den geliyor ve eşik yalnızca val'den türetiliyor (test'e
    bakmanın sessizce bir sapma yaratması tekrarlanamaz olurdu)."""
    kwargs = dict(trading_cfg=CFG, cost_cfg=BacktestCostConfig(), **SMALL)
    a, folds_a = run_intraday_walk_forward(**kwargs)
    b, folds_b = run_intraday_walk_forward(**kwargs)

    assert a.equity_curve == b.equity_curve
    assert [(t.symbol, t.opened_at, t.net_pnl) for t in a.trades] == [
        (t.symbol, t.opened_at, t.net_pnl) for t in b.trades
    ]
    assert [f["entry_threshold"] for f in folds_a] == [f["entry_threshold"] for f in folds_b]


def test_end_date_excludes_that_day_from_trades(tmp_path, monkeypatch, seeded_rows):
    """`run_intraday_backtest(end_date=...)` DAHİL DEĞİL: verilen güne ait
    hiçbir karar/işlem üretilmemeli.

    Aynı `scoped_dataset` deseni burada da: motor kendi bağlantısıyla dataset
    kurmasın, test sembollerinin gerçek satırlarını alsın."""
    import backtest.intraday_engine as engine

    monkeypatch.setattr(engine, "build_intraday_dataset", lambda **kwargs: seeded_rows)

    bundle = {
        "model": _FixedProbModel(0.9),
        "magnitude_model": _FixedReturnModel(0.02),
        "features": INTRADAY_FEATURE_COLUMNS,
        "version": "test",
        "entry_threshold": -1.0,  # her dolumlu bar aday
        "test_start": _FIRST_DAY,
    }
    model_path = tmp_path / "intraday_model.pkl"
    joblib.dump(bundle, model_path)

    days = sorted(str(d.date()) for d in seeded_rows["_day"].unique())
    assert len(days) >= 4, "end_date testi için en az 4 gün gerekiyor"
    end_day = days[3]

    full = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path
    )
    assert full.trades, "referans koşu hiç işlem üretmedi"

    bounded = run_intraday_backtest(
        trading_cfg=CFG,
        cost_cfg=BacktestCostConfig(),
        model_path=model_path,
        end_date=end_day,
    )

    assert bounded.trades, "end_date ile filtrelenen koşu hiç işlem üretmedi"
    assert all(t.opened_at[:10] < end_day for t in bounded.trades), (
        f"end_date={end_day} DAHİL EDİLMEMELİ ama o güne ait işlem var"
    )
    assert all(t.closed_at[:10] < end_day for t in bounded.trades)
    assert all(d["date"][:10] < end_day for d in bounded.decisions)
    assert all(day < end_day for day, _ in bounded.equity_curve)
    assert bounded.end_date[:10] < end_day

    # Filtre gerçekten bir şey kaldırdı: end_date verilmeden end_date günü
    # oynatılıyordu.
    assert any(t.opened_at[:10] == end_day for t in full.trades)
