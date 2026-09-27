"""Gerçek historical intraday (1h) backtest motoru — backtest/engine.py'nin
(günlük) intraday karşılığı. STRATEJİ v2.

KRİTİK: strateji GECE POZİSYON TUTMAZ. Model hedefi (pipeline/
intraday_features.py) yalnızca gün-içi, GÜN SONU'NA KADAR getiriyi hedefler,
gece sıçraması hakkında hiçbir fikir üretmez — bu yüzden her günün SONUNDA
açık pozisyonlar o günün son barının fiyatıyla ZORLA kapatılır. Bu, günlük
backtest/engine.py'den TEK büyük mimari fark: `max_hold_days`/`held_days`
kavramına gerek yok (pozisyon zaten aynı gün içinde açılıp kapanıyor).

v2'nin dört kuralı (hepsi v1'in GERÇEK backtest'teki kaybının köküydü):

1. **Günde sembol başına TEK giriş.** v1 her barda, eşik üstü olduğu sürece
   pozisyon kapatıp yeniden açabiliyordu; gün içi "prob düştü" çıkışı
   (churn) ücreti ikiye katlıyordu.
2. **Giriş kararı bar t KAPANIŞINDA, dolum bar t+1 AÇILIŞINDA.** v1 sinyal
   barının kapanışında hem karar veriyor hem aynı fiyattan alıyordu — aynı
   bar yürütme iyimserliği. Artık dolum fiyatı `next_open`'dır (slippage 0
   iken `entry_price` bir sonraki barın açılışına EŞITTİR).
3. **prob-flip çıkışı YOK.** Açık pozisyon yalnızca `stop_loss` /
   `take_profit` ile veya gün sonunda kapanır; aksi halde "tut".
4. **Günün son barında giriş YOK.** Orada girilirse anında zorla kapanış
   elde edilir, yani getiri saf ücret kaybıdır (bkz.
   `pipeline/intraday_features.py` docstring'i — aynı gerekçeyle o barın
   hedefi de NaN yapılmıştı).

GİRİŞ EŞİĞİ: `pred_rod > entry_threshold`, `entry_threshold` regresorun
VALIDATION tahminlerinin `entry_quantile` kantilidir (`pipeline/
intraday_train_model.py`, config `strategy.entry_quantile`). Eşik modelin
TAHMİN ETTİĞİ gün sonu getirisidir; `expected_return=pred_rod` olarak
`min_expected_edge_pct` filtresine de aynı anlamda girer.

VARSAYILAN OLARAK SADECE HİÇ GÖRÜLMEMİŞ TEST DÖNEMİ oynatılır: `start_date`
verilmezse bundle'daki `test_start` kullanılır. v1'in backtest'i EĞİTİM
dönemini de içeriyordu, yani in-sample sonuç üretiyordu. Feature'lar tüm
geçmiş üzerinden hesaplanır (ısınma/rolling), SADECE karar satırları
`start_date`'e göre filtrelenir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backtest.costs import BacktestCostConfig
from backtest.engine import BacktestResult
from backtest.portfolio import BacktestPortfolio
from pipeline.intraday_dataset import build_intraday_dataset, load_intraday_model_config
from trading.config import TradingConfig, load_trading_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Günlük motorun `DEFAULT_MODEL_PATH` gibi hardcode DEĞİL: intraday model
# dosyası config/intraday_model.yaml::model.path'te tanımlı, oradan okunur.
DEFAULT_MODEL_PATH = PROJECT_ROOT / load_intraday_model_config()["model"]["path"]


def load_intraday_model(
    model_path: Path | None = None,
) -> tuple[Any, Any, list[str], dict[str, Any]]:
    """Bundle'ın TAMAMINI döner: `(model, magnitude_model, features, bundle)`.

    v2 stratejisi (`pred_rod > entry_threshold` + t+1 açılış dolumu) v1'den
    farklı iki anahtara zorunlu ihtiyaç duyar: `magnitude_model` (beklenen
    gün sonu getirisi) ve `entry_threshold` (validation'dan türetilmiş giriş
   eşiği). v1 bundle'ında ikisi de yoktur — sessizce "0 getiri / 0 eşik"
    varsaymak, backtest'i sessizce anlamsızlaştırırdı, o yüzden AÇIK hata
    veriyoruz."""
    model_path = model_path or DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}\nÖnce: python -m pipeline.intraday_train_model"
        )
    bundle = joblib.load(model_path)  # trusted, locally-trained artifact (same as backtest/engine.py)
    if bundle.get("magnitude_model") is None:
        raise ValueError(
            "Bu model v2 öncesi (magnitude_model yok); yeniden eğitin: "
            "python -m pipeline.intraday_train_model"
        )
    if "entry_threshold" not in bundle:
        raise ValueError(
            "Bu model v2 öncesi (entry_threshold yok); yeniden eğitin: "
            "python -m pipeline.intraday_train_model"
        )
    return bundle["model"], bundle["magnitude_model"], bundle["features"], bundle


def _localized_day_bound(bound: str) -> pd.Timestamp:
    """`"2026-05-11"` gibi gün bazlı ISO GÜN eşiğini `ts`'in TIMESTAMPTZ
    dünyasına bağlar. `ts` tz-aware olduğu için naive eşikle karşılaştırmak
    TypeError verir; bağlama yereli İstanbul'dur çünkü gün sınırını o
    yerel tanımlar."""
    ts = pd.Timestamp(bound)
    return ts if ts.tz is not None else ts.tz_localize("Europe/Istanbul")


def simulate_intraday(
    df: pd.DataFrame,
    trading_cfg: TradingConfig,
    cost_cfg: BacktestCostConfig,
) -> BacktestResult:
    """KARAR SATIRLARINI oynatır ve portföyü kurar.

    `df` = `build_intraday_dataset` satırları + model çıktıları. Üç kolon
    ZORUNLUDUR ve SADECE bu fonksiyonun girdisidir:

    * `pred_rod` — regresörün beklenen gün sonu getirisi,
    * `prob_up` — sınıflandırıcının karar logu (yalnızca raporlama),
    * `entry_threshold` — SATIR BAZLI giriş eşiği.

    Sonuncusu neden kolondur: `run_intraday_backtest` tek bir EŞİK ile çalışır
    ama `backtest/intraday_walk_forward.py` her fold için kendi eşiğini
    ÜRETİR (val tahminlerinin kantili). Eşiği satırdan okumak, fold
    sınırlarını oynatma kodundan ayırır.

    TÜM strateji kuralı (günde tek giriş, t+1 açılış dolumu, gün sonu zorla
    kapanış) burada yaşar ve `run_intraday_backtest` ile paylaşılır — iki
    giriş noktası aynı kodu çalıştırır, sapma olamaz.

    Girdi boşsa boş `BacktestResult` döner (çağıran taraf `None` pencereleri
    tolere edebilsin diye exception DEĞİL)."""
    if df.empty:
        return BacktestResult([], [], [], None, None)

    df = df.copy()
    # Gün tespiti: `ts` TIMESTAMPTZ olduğu için önce naive'e çevrilip
    # normalize edilir — `pipeline/intraday_features.py`'deki AYNI dönüşüm
    # (aksi halde `normalize()` UTC gün sınırında keser ve barlar yanlış güne yazılır).
    df["_day"] = df["ts"].dt.tz_convert(None).dt.normalize()

    # Karar barının KAPANIŞINDA al, bir sonraki barın AÇILIŞINDA doldur.
    # `next_open` NaN olan barlar günün son barıdır (bir sonraki bar başka
    # bir güne ait) — onlara girilmez (gün sonunda anında kapanış = saf ücret).
    grouped = df.groupby(["symbol_id", "_day"], sort=False)
    df["next_open"] = grouped["open"].shift(-1)
    df["next_ts"] = grouped["ts"].shift(-1)

    result_start = df["ts"].min().isoformat()
    result_end = df["ts"].max().isoformat()

    portfolio = BacktestPortfolio(starting_balance=trading_cfg.starting_balance)
    decisions: list[dict[str, Any]] = []

    for day, day_rows in df.groupby("_day", sort=True):
        last_price_of_day: dict[str, float] = {}
        last_ts_of_day: dict[str, pd.Timestamp] = {}
        # v2: bir sembol BİR GÜN en fazla bir kez girilebilir.
        entered_today: set[str] = set()

        for ts, bar_rows in day_rows.sort_values("ts").groupby("ts", sort=True):
            for ticker, close, row_ts in zip(
                bar_rows["ticker"], bar_rows["close"], bar_rows["ts"]
            ):
                last_price_of_day[ticker] = float(close)
                last_ts_of_day[ticker] = row_ts
            by_symbol = {row["ticker"]: row for _, row in bar_rows.iterrows()}

            # 1) Açık pozisyonları değerlendir: SADECE stop_loss / take_profit.
            # v1'deki `prob_up < sell_threshold` çıkışı KALDIRILDI: saatlik
            # prob salınımı (churn) her seferinde round-trip ücreti demekti
            # ve backtest'in en büyük zarar kaynağıydı. Eşiklerin içinde
            # kalan pozisyon GÜN SONU'NA KADAR tutulur.
            for symbol in list(portfolio.open_positions.keys()):
                row = by_symbol.get(symbol)
                if row is None:
                    continue  # o barda bu sembol için veri yok; pozisyona dokunma
                position = portfolio.open_positions[symbol]
                price = float(row["close"])
                prob_up = float(row["prob_up"])

                if (
                    trading_cfg.stop_loss_pct > 0
                    and price <= position.entry_price * (1 - trading_cfg.stop_loss_pct)
                ):
                    portfolio.sell(symbol, price, "stop_loss", ts.isoformat(), trading_cfg, cost_cfg)
                    decisions.append(dict(date=ts.isoformat(), symbol=symbol, action="sat", reason=f"stop_loss: fiyat entry karşısında %{trading_cfg.stop_loss_pct * 100:.1f} düştü", prob_up=prob_up))
                elif (
                    trading_cfg.take_profit_pct > 0
                    and price >= position.entry_price * (1 + trading_cfg.take_profit_pct)
                ):
                    portfolio.sell(symbol, price, "take_profit", ts.isoformat(), trading_cfg, cost_cfg)
                    decisions.append(dict(date=ts.isoformat(), symbol=symbol, action="sat", reason=f"take_profit: fiyat entry karşısında %{trading_cfg.take_profit_pct * 100:.1f} yükseldi", prob_up=prob_up))
                else:
                    decisions.append(dict(date=ts.isoformat(), symbol=symbol, action="tut", reason="eşiklerin içinde", prob_up=prob_up))

            # 2) Giriş adayları — yalnızca dolumu mümkün olan barlar. Eşik
            #    SATIR BAZLIdır (`entry_threshold` kolonu): tek eşikli
            #    backtest'te sabit, walk-forward'da fold'a göre değişir.
            candidates = bar_rows[
                (bar_rows["pred_rod"] > bar_rows["entry_threshold"])
                & bar_rows["next_open"].notna()
            ].sort_values("pred_rod", ascending=False)
            for _, row in candidates.iterrows():
                symbol = row["ticker"]
                if symbol in portfolio.open_positions or symbol in entered_today:
                    continue
                pred_rod = float(row["pred_rod"])
                prob_up = float(row["prob_up"])
                # Karar bar t kapanışında verilir, dolum bar t+1 AÇILIŞINDA
                # olur: `opened_at` dolumun zaman damgasıdır.
                ok, reason = portfolio.buy(
                    symbol, float(row["next_open"]), prob_up, row["next_ts"].isoformat(),
                    trading_cfg, cost_cfg, expected_return=pred_rod,
                )
                if ok:
                    entered_today.add(symbol)
                decisions.append(dict(date=ts.isoformat(), symbol=symbol, action="al" if ok else "red", reason=reason, prob_up=prob_up, pred_rod=pred_rod))

        # GÜN SONU: model gece riski hakkında hiçbir fikir üretmiyor, bu yüzden
        # açık pozisyonlar son bilinen fiyatla ZORLA kapatılır. `closed_at`
        # gece yarısı DEĞİL, o sembolün günün SON barının gerçek zaman damgası
        # olmalıdır — aksi halde closed_at opened_at'tan ÖNCE görünür ve
        # `average_holding_days` negatif çıkar.
        for symbol in list(portfolio.open_positions.keys()):
            price = last_price_of_day[symbol]
            closed_at = last_ts_of_day[symbol].isoformat()
            portfolio.sell(symbol, price, "gun_sonu_kapanis", closed_at, trading_cfg, cost_cfg)
            decisions.append(dict(date=closed_at, symbol=symbol, action="sat", reason="gun_sonu_kapanis (gece pozisyon tutulmuyor)", prob_up=None))

        portfolio.record_snapshot(day.isoformat(), last_price_of_day)

    return BacktestResult(portfolio.equity_curve, portfolio.closed_trades, decisions, result_start, result_end)


def run_intraday_backtest(
    trading_cfg: TradingConfig | None = None,
    cost_cfg: BacktestCostConfig | None = None,
    model_path: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BacktestResult:
    """`bist_only` parametresi KALDIRILDI (2026-09-27): `build_intraday_dataset`
    artık zaten sadece BIST döndürüyor (canlı `trading/engine.py`'nin alım
    sorgusuyla aynı evren — US sembolleri hiç ticaret edilmiyor), ayrıca
    filtrelemeye gerek yok.

    `start_date` verilmezse bundle'daki `test_start` kullanılır: backtest
    varsayılan olarak SADECE HİÇ GÖRÜLMEMİŞ test dönemini oynatır.

    `end_date` DAHİL DEĞİL: verilen günün barları oynatılmaz (o günün 00:00
    damgası `ts < end_ts` filtresinin dışında kalır). `start_date` ile aynı
    Europe/Istanbul gün sınırı mantığı kullanılır."""
    trading_cfg = trading_cfg or load_trading_config()
    cost_cfg = cost_cfg or BacktestCostConfig()
    model, magnitude_model, features, bundle = load_intraday_model(model_path)
    entry_threshold = float(bundle["entry_threshold"])

    # Feature'lar TÜM geçmiş üzerinden hesaplanır (rolling/ısınma pencereleri
    # eksik olsaydı ilk günlerin feature'ları NaN olurdu) — filtreleme yalnızca
    # KARAR satırlarına uygulanır.
    df = build_intraday_dataset(require_target=False)
    if df.empty:
        return BacktestResult([], [], [], None, None)
    df = df.copy()

    start = start_date or bundle.get("test_start")
    if start:
        df = df[df["ts"] >= _localized_day_bound(start)].copy()
    if end_date:
        df = df[df["ts"] < _localized_day_bound(end_date)].copy()
    if df.empty:
        return BacktestResult([], [], [], None, None)

    df["pred_rod"] = magnitude_model.predict(df[features])
    df["prob_up"] = model.predict_proba(df[features])[:, 1]  # yalnızca karar logu
    # Tek eşik, motorun SATIR BAZLI girdi sözleşmesine yazılır; oynatma
    # mantığı `simulate_intraday`'da yaşar.
    df["entry_threshold"] = entry_threshold

    return simulate_intraday(df, trading_cfg, cost_cfg)
