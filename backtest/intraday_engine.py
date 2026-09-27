"""Gerçek historical intraday (1h) backtest motoru — backtest/engine.py'nin
(günlük) intraday karşılığı.

KRİTİK: strateji GECE POZİSYON TUTMAZ. Model hedefi (pipeline/
intraday_features.py) yalnızca gün-içi bir sonraki bara bakar, gece
sıçraması hakkında hiçbir fikir üretmez — bu yüzden her günün SONUNDA
açık pozisyonlar son bilinen fiyatla ZORLA kapatılır. Bu, günlük
backtest/engine.py'den TEK büyük mimari fark: `max_hold_days`/`held_days`
kavramına gerek yok (pozisyon zaten aynı gün içinde açılıp kapanıyor)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from backtest.costs import BacktestCostConfig
from backtest.engine import BacktestResult
from backtest.portfolio import BTClosedTrade, BacktestPortfolio
from pipeline.intraday_dataset import build_intraday_dataset, load_intraday_model_config
from trading.config import TradingConfig, load_trading_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Günlük motorun `DEFAULT_MODEL_PATH` gibi hardcode DEĞİL: intraday model
# dosyası config/intraday_model.yaml::model.path'te tanımlı, oradan okunur.
DEFAULT_MODEL_PATH = PROJECT_ROOT / load_intraday_model_config()["model"]["path"]


def load_intraday_model(model_path: Path | None = None) -> tuple[Any, Any | None, list[str]]:
    model_path = model_path or DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}\nÖnce: python -m pipeline.intraday_train_model"
        )
    bundle = joblib.load(model_path)  # trusted, locally-trained artifact (same as backtest/engine.py)
    # `magnitude_model` Faz 6/7 ile eklendi; egitilmeden onceki bundle'lar yok —
    # `.get()` ile None'a dusulur ve beklenen getiri 0 kabul edilir.
    return bundle["model"], bundle.get("magnitude_model"), bundle["features"]


def run_intraday_backtest(
    trading_cfg: TradingConfig | None = None,
    cost_cfg: BacktestCostConfig | None = None,
    model_path: Path | None = None,
) -> BacktestResult:
    """`bist_only` parametresi KALDIRILDI (2026-09-27): `build_intraday_dataset`
    artık zaten sadece BIST döndürüyor (canlı `trading/engine.py`'nin alım
    sorgusuyla aynı evren — US sembolleri hiç ticaret edilmiyor), ayrıca
    filtrelemeye gerek yok."""
    trading_cfg = trading_cfg or load_trading_config()
    cost_cfg = cost_cfg or BacktestCostConfig()
    model, magnitude_model, features = load_intraday_model(model_path)

    df = build_intraday_dataset(require_target=False)
    if df.empty:
        return BacktestResult([], [], [], None, None)
    df = df.copy()

    df["prob_up"] = model.predict_proba(df[features])[:, 1]
    df["expected_return"] = magnitude_model.predict(df[features]) if magnitude_model is not None else 0.0
    # Gün tespiti: `ts` TIMESTAMPTZ olduğu için önce naive'e çevrilip
    # normalize edilir — `pipeline/intraday_features.py`'deki AYNI dönüşüm
    # (aksi halde `normalize()` UTC gün sınırında keser ve barlar yanlış güne yazılır).
    df["_day"] = df["ts"].dt.tz_convert(None).dt.normalize()

    start_date = df["ts"].min().isoformat()
    end_date = df["ts"].max().isoformat()

    portfolio = BacktestPortfolio(starting_balance=trading_cfg.starting_balance)
    decisions: list[dict[str, Any]] = []

    for day, day_rows in df.groupby("_day", sort=True):
        last_price_of_day: dict[str, float] = {}

        for ts, bar_rows in day_rows.sort_values("ts").groupby("ts", sort=True):
            for ticker, close in zip(bar_rows["ticker"], bar_rows["close"]):
                last_price_of_day[ticker] = float(close)
            by_symbol = {row["ticker"]: row for _, row in bar_rows.iterrows()}

            # Açık pozisyonları değerlendir: sat ya da tut.
            # Rula sırası canlı `trading/engine.py`'nin AYNISI — stop_loss →
            # take_profit → prob düştü (`max_hold_days` YOK: pozisyon zaten aynı
            # gün içinde kapanır). İlk uyan kural satışı tetikler.
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
                elif prob_up < trading_cfg.sell_threshold:
                    portfolio.sell(symbol, price, "prob_düştü", ts.isoformat(), trading_cfg, cost_cfg)
                    decisions.append(dict(date=ts.isoformat(), symbol=symbol, action="sat", reason=f"prob_up {prob_up:.3f} < eşik {trading_cfg.sell_threshold}", prob_up=prob_up))
                else:
                    decisions.append(dict(date=ts.isoformat(), symbol=symbol, action="tut", reason="eşiklerin içinde", prob_up=prob_up))

            candidates = bar_rows[bar_rows["prob_up"] > trading_cfg.buy_threshold].sort_values("prob_up", ascending=False)
            for _, row in candidates.iterrows():
                symbol = row["ticker"]
                if symbol in portfolio.open_positions:
                    continue
                ok, reason = portfolio.buy(
                    symbol, float(row["close"]), float(row["prob_up"]), ts.isoformat(),
                    trading_cfg, cost_cfg, expected_return=float(row["expected_return"]),
                )
                decisions.append(dict(date=ts.isoformat(), symbol=symbol, action="al" if ok else "red", reason=reason, prob_up=float(row["prob_up"])))

        # GÜN SONU: model gece riski hakkında hiçbir fikir üretmiyor, bu yüzden
        # açık pozisyonlar son bilinen fiyatla ZORLA kapatılır.
        for symbol in list(portfolio.open_positions.keys()):
            price = last_price_of_day[symbol]
            portfolio.sell(symbol, price, "gun_sonu_kapanis", day.isoformat(), trading_cfg, cost_cfg)
            decisions.append(dict(date=day.isoformat(), symbol=symbol, action="sat", reason="gun_sonu_kapanis (gece pozisyon tutulmuyor)", prob_up=None))

        portfolio.record_snapshot(day.isoformat(), last_price_of_day)

    return BacktestResult(portfolio.equity_curve, portfolio.closed_trades, decisions, start_date, end_date)
