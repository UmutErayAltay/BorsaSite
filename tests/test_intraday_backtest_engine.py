"""`backtest/intraday_engine.py` integration tests against the real local
Postgres (synthetic bars, no network) — mirrors tests/test_backtest_engine.py.

DİKKAT — `tests/conftest.py::committed_conn` bu dosyada KULLANILMIYOR: onun
teardown'ı tüm tabloları TRUNCATE eder, yani veritabanındaki gerçek (ve
pahalı) 547K satırlık intraday verisini siler. Aşağıdaki `intraday_conn`
deseni `tests/test_intraday_dataset.py`'dekiyle aynıdır ama teardown'da
YALNIZCA bu dosyanın kullandığı test ticker'larını temizler.

Testler STRATEJİ v2'yi doğrular: günde sembol başına tek giriş, dolum bir
sonraki barın AÇILIŞINDA, prob-flip çıkışı yok, gün sonu barında giriş yok,
ve varsayılan olarak SADECE HİÇ GÖRÜLMEMİŞ test dönemi oynatılır.
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

# Bundle'da `test_start` olarak verilip `start_date` verilmediğinde motorun
# oynatmayacağı gün aralığının BAŞLANGICI. Tohum veri 2026-01-05'te başlıyor;
# bu günün çok ilerisini seçiyoruz ki test_start öncesi barlar kesinlikle
# mevcut olsun (aksi halde "test_start öncesi karar yok" testi, o barlar
# zaten yok diye anlamsızca geçerdi).
_TEST_START = "2026-01-25"


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
    sembol dataset'e hiç giremez.

    `open` bilerek `close`'tan FARKLI: v2 dolumu bir sonraki barın AÇILIŞINDA
    yaptığı için testler bu ikisinin ayırt edilebilir olmasına güveniyor."""
    symbol_id = upsert_symbol(conn, ticker, "BIST", "TRY")
    rows = []
    price = 100.0
    for d in range(days):
        day = pd.Timestamp("2026-01-05", tz="Europe/Istanbul") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            price += 0.1 if b % 2 == 0 else -0.05
            close = price
            open_ = close + 0.25  # her barda açılış != kapanış
            ts = day + pd.Timedelta(hours=10 + b)
            rows.append((ts.isoformat(), open_, close + 0.5, close - 0.5, close, 1000.0 + b))
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
    tarıyor ve sahte model bunların hepsinde alım üretirdi — testler kendi
    sembollerine odaklanamazdı. Motorun üzerinde çalıştığı feature/hedef/
    gün-sınırı kodu ve okuduğu veri tamamen GERÇEK; sadece hangi satırların
    oynandığı daraltılıyor."""
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


def _dump_model(
    tmp_path: Path,
    prob: float,
    expected_return: float | None,
    *,
    entry_threshold: float | None = -1.0,
    test_start: str | None = _TEST_START,
) -> Path:
    """v2 bundle: `magnitude_model` + `entry_threshold` + `test_start`.

    `expected_return=None` VE/VEYA `entry_threshold=None` ise v2 ÖNCESİ (legacy)
    bundle üretilir — `load_intraday_model` bunu açık `ValueError` ile
    reddetmelidir. `test_start=None` ise bundle'da anahtar hiç yoktur (tüm
    geçmiş oynatılır)."""
    bundle: dict = {
        "model": _FixedProbModel(prob),
        "features": INTRADAY_FEATURE_COLUMNS,
        "version": "test",
    }
    if expected_return is not None:
        bundle["magnitude_model"] = _FixedReturnModel(expected_return)
    if entry_threshold is not None:
        bundle["entry_threshold"] = entry_threshold
    if test_start is not None:
        bundle["test_start"] = test_start
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

    Sahte model her barda pozitif getiri tahmini döndürür (giriş eşiğinin
    üstünde), bu yüzden günün İLK dolumlu bar'ında alım olmalı ve o pozisyon
    O GÜN içinde kapanmalı."""
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=0.02)

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    buys = [d for d in result.decisions if d["action"] == "al"]
    assert buys, "giriş eşiğinin üstündeyken hiç alım olmadı"

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
    #    2. güne devralınan pozisyon YOK. `closed_at` artık gün sonu barının
    #    GERÇEK zaman damgası (gece yarısı DEĞİL), yani asla `opened_at`'tan
    #    önce görünmemeli.
    assert result.trades, "hiçbir işlem kapanmadı"
    for trade in result.trades:
        opened = pd.Timestamp(trade.opened_at)
        closed = pd.Timestamp(trade.closed_at)
        assert opened.date() == closed.date(), (
            f"{trade.symbol} GECE POZİSYONU TUTMUŞ: {opened} -> {closed}"
        )
        assert closed >= opened, (
            f"{trade.symbol} KAPANIŞ AÇILIŞTAN ÖNCE: {opened} -> {closed}"
        )

    # 3) Getiri tahmini sabit ve pozitif olduğu için başka bir çıkış sebebi
    #    (stop_loss/take_profit) ÜRETİLMEMELİ.
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


def test_never_enters_on_the_last_bar_of_a_day(scoped_dataset, tmp_path):
    """v2 kuralı: günün SON barında giriş YOK. Orada girilirse bir sonraki
    bar diye bir şey kalmadığı için pozisyon ANINDA gün sonu kapanışına
    düşer — getiri saf ücret kaybı olur (v1'in en pahalı hatası).

    Tohum veride `open == close` OLMAYAN bir yapı kurulduğu için bu kontrol
    dolum fiyatını da eşleştiriyor: son barda girilseydi `next_open` NaN
    olduğu için hiçbir dolum gerçekleşmezdi."""
    rows = scoped_dataset
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=0.02)

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    last_bar_of_day = (
        rows.loc[rows["_day"] == day, "ts"].max()
        for day in _observable_days(rows)
    )
    last_bars = {ts.isoformat() for ts in last_bar_of_day}

    buys = [d for d in result.decisions if d["action"] == "al"]
    assert buys, "giriş eşiğinin üstündeyken hiç alım olmadı"
    for decision in buys:
        assert decision["date"] not in last_bars, (
            f"{decision['date']} günün son barı — anında kapanış = saf ücret kaybı"
        )
    # İşlem kaydı da aynı şeyi söylemeli: opened_at asla son bar olamaz.
    for trade in result.trades:
        assert pd.Timestamp(trade.opened_at).isoformat() not in last_bars


def test_at_most_one_entry_per_symbol_per_day(scoped_dataset, tmp_path):
    """v2 kuralı: bir sembole günde EN FAZLA BİR giriş. v1 her barda pozisyon
    kapatıp yeniden açabildiği için gün içi churn üretiyordu.

    Tohum fiyatlar gün içinde ağırlıklı olarak YÜKSELTİR (take_profit yok,
    entry zaten yakın), ama testin sağlamlığı fiyat yoluna değil sayıma
    dayanır: aynı sembol + aynı gün birden fazla "al" kararı OLMAMALI —
    ancak `entered_today` sayesinde pozisyon açıldıktan sonra ikinci giriş
    denemesi bile yapılmaz."""
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=0.02)

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    per_day: dict[tuple[str, str], int] = {}
    for decision in result.decisions:
        if decision["action"] == "al":
            key = (decision["symbol"], decision["date"][:10])
            per_day[key] = per_day.get(key, 0) + 1

    assert per_day, "hiç alım olmadı — test anlamsız olurdu"
    for (symbol, day), count in per_day.items():
        assert count == 1, f"{symbol} {day} gününde {count} giriş yapıldı (en fazla 1 olmalı)"


def test_fill_happens_at_next_bar_open_not_signal_bar_close(scoped_dataset, tmp_path):
    """v2 kuralı: karar bar t'nin KAPANIŞINDA verilir, dolum bar t+1'in
    AÇILIŞINDA olur. Slippage 0 iken `entry_price`, bir sonraki barın AÇILIŞ
    fiyatına EŞİT olmalıdır — sinyal barının kapanışına DEĞİL.

    Tohum veride her barın `open`'ı `close`'undan farklıdır; eşleştirme yapmak
    için dolumun yapıldığı barı bulup fiyatını doğruluyoruz."""
    rows = scoped_dataset
    # 0.0 eşiği: her dolumlu bar adaydır (0 getiri tahmini bile eşiği geçer
    # değil — çünkü eşik `>` ile karşılaştırılıyor, eşiği 0.01 yapsak
    # sıfır getiri elenir; bu yüzden tahmin 0.02 ile eşik -1.0).
    model_path = _dump_model(
        tmp_path, prob=0.9, expected_return=0.02, entry_threshold=-1.0
    )

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    buys = [d for d in result.decisions if d["action"] == "al"]
    assert buys, "hiç alım olmadı"

    # (ticker, ts) -> açılış/kapalış fiyatı haritası
    ohlc = {
        (row["ticker"], row["ts"].isoformat()): (float(row["open"]), float(row["close"]))
        for _, row in rows.iterrows()
    }
    checked = 0
    for trade in result.trades:
        bar = (trade.symbol, trade.opened_at)
        assert bar in ohlc, f"dolgum zaman damgası veriyle eşleşmiyor: {bar}"
        open_price, close_price = ohlc[bar]
        assert trade.entry_price == pytest.approx(open_price), (
            f"{trade.symbol} dolumu {open_price} açılışında değil, "
            f"{trade.entry_price} fiyatında yapıldı"
        )
        # Sinyal barının kapanışı dolum fiyatı OLAMAZ (open != close).
        assert open_price != close_price, "tohum veri open==close — test anlamsız olurdu"
        checked += 1
    assert checked, "hiçbir işlem fiyat eşleştirmesi yapamadı"


def test_exit_reasons_have_no_prob_flip(scoped_dataset, tmp_path):
    """v2 kuralı: açık pozisyon yalnızca stop_loss / take_profit ile veya gün
    sonunda kapanır. v1'in `prob_up < sell_threshold` çıkışı — saatlik prob
    salınımının her turunda round-trip ücreti üretmesi — KALDIRILDI.

    Tohum model prob'u sabit; test ayrıca SELL eşiğinin ALTINDA bir prob ile
    de koşar, ki motor yanlışlıkla prob'a bakmaya devam etse yakalansın."""
    model_path = _dump_model(tmp_path, prob=0.9, expected_return=0.02)
    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    # prob_up = 0.1, sell_threshold = 0.50 → v1'de HER pozisyon anında kapanırdı.
    low_prob_path = _dump_model(
        tmp_path / "lowprob", prob=0.1, expected_return=0.02, entry_threshold=-1.0
    )
    low_prob_result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=low_prob_path,
    )

    allowed = {"gun_sonu_kapanis", "stop_loss", "take_profit"}
    for res in (result, low_prob_result):
        assert res.trades, "hiçbir işlem kapanmadı"
        for trade in res.trades:
            assert trade.exit_reason in allowed, (
                f"beklenmeyen çıkış sebebi: {trade.exit_reason} (izin verilenler {allowed})"
            )
    # Düşük prob'lu koşuda da v1'in "prob_düştü" sürüsü üretilmemeli.
    assert all(
        "prob_düştü" not in str(d.get("reason", "")) for res in (result, low_prob_result)
        for d in res.decisions
    )


def test_defaults_to_out_of_sample_test_start(scoped_dataset, tmp_path):
    """`start_date` verilmezse motor bundle'daki `test_start`'ten başlar:
    v1'in backtest'i EĞİTİM dönemini de içeriyordu, yani in-sample sonuç
    üretiyordu. Test dönemi öncesi HİÇBİR karar/işlem olmamalı."""
    model_path = _dump_model(
        tmp_path, prob=0.9, expected_return=0.02, entry_threshold=-1.0,
        test_start=_TEST_START,
    )

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    # Test_start öncesi günler gerçekten veride VAR (test anlamsız olmasın).
    all_days = {str(d.date()) for d in scoped_dataset["_day"].unique()}
    assert any(day < _TEST_START for day in all_days), (
        "tohum veride test_start öncesi gün yok — test anlamsız olurdu"
    )

    assert result.trades, "hiçbir işlem kapanmadı"
    for decision in result.decisions:
        assert decision["date"][:10] >= _TEST_START, (
            f"{decision['date']} kararı test dönemi ÖNCESİNDE üretildi (in-sample)"
        )
    for trade in result.trades:
        assert trade.opened_at[:10] >= _TEST_START
        assert trade.closed_at[:10] >= _TEST_START
    # Equity curve de yalnızca test dönemini kapsamalı.
    assert all(day >= _TEST_START for day, _ in result.equity_curve)
    # Sonuç penceresi de filtrelenmiş veriden gelmeli.
    assert result.start_date >= f"{_TEST_START}T00:00:00"


def test_explicit_start_date_overrides_bundle(scoped_dataset, tmp_path):
    """`start_date` verilirse bundle `test_start`'i GEÇERSIZ kılır — hız/ara
    sıralama testleri ve özel senaryolar için gerekli."""
    model_path = _dump_model(
        tmp_path, prob=0.9, expected_return=0.02, entry_threshold=-1.0,
        test_start=_TEST_START,
    )

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
        start_date="2026-01-06",
    )

    assert result.trades, "hiçbir işlem kapanmadı"
    assert all(d["date"][:10] >= "2026-01-06" for d in result.decisions)


@pytest.mark.parametrize(
    "expected_return, entry_threshold",
    [(None, -1.0), (0.02, None), (None, None)],
    ids=["magnitude_model_yok", "entry_threshold_yok", "ikisi_de_yok"],
)
def test_legacy_v1_bundle_is_rejected(scoped_dataset, tmp_path, expected_return, entry_threshold):
    """v2 stratejisi `magnitude_model` + `entry_threshold` olmadan ANLAMSAZ
    olur. v1 bundle'ı sessizce "0 getiri / 0 eşik" kabul edilip anlamsız bir
    backtest üretmemeli — açık `ValueError` vermeli."""
    model_path = _dump_model(
        tmp_path, prob=0.9, expected_return=expected_return,
        entry_threshold=entry_threshold, test_start=None,
    )

    with pytest.raises(ValueError, match="v2 öncesi"):
        run_intraday_backtest(
            trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
        )


def test_pred_rod_at_or_below_entry_threshold_never_buys(scoped_dataset, tmp_path):
    """Giriş kararı yalnızca `pred_rod > entry_threshold`'ten gelir; eşiğin
    altında (ve tam eşit) tahmin HİÇBİR alım üretmemeli."""
    model_path = _dump_model(
        tmp_path, prob=0.9, expected_return=0.0, entry_threshold=0.0,
    )

    result = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    assert not result.trades
    assert all(d["action"] != "al" for d in result.decisions)
    assert all(value == CFG.starting_balance for _, value in result.equity_curve)

    # Eşiğin ÜSTÜne çıkınca alım başlamalı — aksi halde test "hiçbir şey olmuyor"
    # anlamına gelirdi.
    above_path = _dump_model(
        tmp_path / "above", prob=0.9, expected_return=0.01, entry_threshold=0.0,
    )
    above = run_intraday_backtest(
        trading_cfg=CFG, cost_cfg=BacktestCostConfig(), model_path=above_path,
    )
    assert any(d["action"] == "al" for d in above.decisions)


def test_min_expected_edge_blocks_candidate_via_existing_buy_path(scoped_dataset, tmp_path):
    """Edge-after-cost filtresinin intraday'e YENİ kod yazmadan ulaştığının
    kanıtı: aynı `BacktestPortfolio.buy()` çağrısı, `min_expected_edge_pct > 0`
    iken düşük beklenen getirili adayı reddeder (canlı `trading/portfolio.py`
    ile aynı davranış).

    v2'de `expected_return=pred_rod` ANLAMCA DOĞRU: regresor gün sonuna kadar
    beklenen getiriyi tahmin ediyor, yani filtre modelin kendi görüşüne
    soruyor."""
    edge_cfg = replace(CFG, min_expected_edge_pct=0.01)
    # %0.5 beklenen getiri, %1 güvenlik payını + round-trip ücretini karşılamaz.
    model_path = _dump_model(
        tmp_path, prob=0.9, expected_return=0.005, entry_threshold=-1.0
    )

    result = run_intraday_backtest(
        trading_cfg=edge_cfg, cost_cfg=BacktestCostConfig(), model_path=model_path,
    )

    rejected = [d for d in result.decisions if d["action"] == "red"]
    assert rejected, "düşük beklenen getiri reddedilmeliydi"
    assert all("beklenen kâr" in d["reason"] for d in rejected)
    assert not any(d["action"] == "al" for d in result.decisions)
    assert not result.trades
    assert all(value == edge_cfg.starting_balance for _, value in result.equity_curve)
