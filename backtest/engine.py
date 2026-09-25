"""Gerçek historical backtest motoru.

`trading/engine.py` (canlı) sadece "en güncel satır"a bakar — geçmiş bir tarih
aralığını yeniden oynatamaz. Bu motor verilen [start_date, end_date] aralığındaki
HER günü kronolojik sırayla oynatır; her gün için sadece o güne kadar (<=D)
hesaplanmış feature'ları ve o günün kapanış fiyatını kullanır. Gelecekteki hiçbir
fiyat/sentiment/tahmin feature olarak kullanılmaz — bkz. docs/BACKTEST_AUDIT.md §3.

Bilinen ve BİLİNÇLİ sınır (audit §5'te de belirtildi): işlem, tahminin üretildiği
GÜNÜN kendi kapanışında yürütülüyor (`trading/engine.py`'nin canlı davranışıyla
birebir aynı) — sadece bu davranışın üzerine slippage/spread ekleniyor. Bu
"aynı-bar execution" varsayımının kendisini (D+1 açılışına kaydırmak gibi)
değiştirmek bu fazın kapsamı dışında, çünkü bu canlı motorun da mevcut, bilinen
davranışı; sessizce değiştirmek üretim sistemiyle karşılaştırmayı bozar.

Faz 1 sınırı: TEK bir (mevcut, tüm geçmişle eğitilmiş) model kullanılıyor — bu
yüzden bu fazın ürettiği sonuçlar in-sample'dır, gerçek out-of-sample performans
walk-forward (Faz 4) olmadan iddia edilemez."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backtest.costs import BacktestCostConfig
from backtest.portfolio import BacktestPortfolio, BTClosedTrade
from pipeline.dataset import build_dataset
from trading.config import TradingConfig, load_trading_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "xgb_up.pkl"


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[tuple[str, float]]
    trades: list[BTClosedTrade]
    decisions: list[dict[str, Any]]
    start_date: str | None
    end_date: str | None


def load_model(model_path: Path | None = None) -> tuple[Any, list[str]]:
    model_path = model_path or DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}\nÖnce: python scripts/run_train.py")
    bundle = joblib.load(model_path)  # trusted, locally-trained artifact (same as pipeline/predict_model.py)
    return bundle["model"], bundle["features"]


def run_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    trading_cfg: TradingConfig | None = None,
    cost_cfg: BacktestCostConfig | None = None,
    model_path: Path | None = None,
    bist_only: bool = True,
) -> BacktestResult:
    trading_cfg = trading_cfg or load_trading_config()
    cost_cfg = cost_cfg or BacktestCostConfig()
    model, features = load_model(model_path)

    df = build_dataset(require_target=False)
    if df.empty:
        return BacktestResult([], [], [], start_date, end_date)

    if bist_only:
        df = df[df["is_bist"] == 1.0]
    if start_date:
        df = df[df["feature_date"] >= start_date]
    if end_date:
        df = df[df["feature_date"] <= end_date]
    if df.empty:
        return BacktestResult([], [], [], start_date, end_date)

    df = df.copy()
    df["prob_up"] = model.predict_proba(df[features])[:, 1]

    portfolio = BacktestPortfolio(starting_balance=trading_cfg.starting_balance)
    decisions: list[dict[str, Any]] = []

    for decision_date, day_rows in df.groupby("feature_date", sort=True):
        latest_prices = dict(zip(day_rows["ticker"], day_rows["close"]))
        by_symbol = {row["ticker"]: row for _, row in day_rows.iterrows()}

        for symbol in list(portfolio.open_positions.keys()):
            row = by_symbol.get(symbol)
            if row is None:
                continue  # o gün bu sembol için veri yok; pozisyona dokunma
            position = portfolio.open_positions[symbol]
            prob_up = float(row["prob_up"])
            held_days = (
                date.fromisoformat(decision_date) - date.fromisoformat(position.opened_at)
            ).days

            if held_days >= trading_cfg.max_hold_days:
                portfolio.sell(symbol, float(row["close"]), "max_hold_süresi", decision_date, trading_cfg, cost_cfg)
                decisions.append(dict(date=decision_date, symbol=symbol, action="sat", reason="max_hold_süresi doldu", prob_up=prob_up))
            elif prob_up < trading_cfg.sell_threshold:
                portfolio.sell(symbol, float(row["close"]), "prob_düştü", decision_date, trading_cfg, cost_cfg)
                decisions.append(dict(date=decision_date, symbol=symbol, action="sat", reason=f"prob_up {prob_up:.3f} < eşik {trading_cfg.sell_threshold}", prob_up=prob_up))
            else:
                decisions.append(dict(date=decision_date, symbol=symbol, action="tut", reason="eşiklerin içinde", prob_up=prob_up))

        candidates = day_rows[day_rows["prob_up"] > trading_cfg.buy_threshold].sort_values("prob_up", ascending=False)
        for _, row in candidates.iterrows():
            symbol = row["ticker"]
            if symbol in portfolio.open_positions:
                continue
            ok, reason = portfolio.buy(symbol, float(row["close"]), float(row["prob_up"]), decision_date, trading_cfg, cost_cfg)
            decisions.append(dict(date=decision_date, symbol=symbol, action="al" if ok else "red", reason=reason, prob_up=float(row["prob_up"])))

        portfolio.record_snapshot(decision_date, latest_prices)

    return BacktestResult(portfolio.equity_curve, portfolio.closed_trades, decisions, start_date, end_date)
