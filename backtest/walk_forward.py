"""Walk-forward, leakage-safe backtest motoru.

Her pencere sıkı sıkıya TRAIN -> VALIDATION -> OOS sırasında ilerler.
Calibration ve threshold seçimi SADECE validation'da fit edilir; OOS
etiketleri asla parametre seçiminde kullanılmaz. Sinyal D kapanışında
üretilir, işlem D+1 açılışında yürütülür (bkz. `_schedule_orders`/
`_execute_pending`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from backtest.calibration import ProbabilityCalibrator
from backtest.costs import BacktestCostConfig
from backtest.portfolio import BacktestPortfolio, BTClosedTrade
from backtest.thresholds import select_buy_threshold
from pipeline.dataset import FEATURE_COLUMNS, build_dataset, load_model_config
from pipeline.model_factory import build_xgb_model
from trading.config import TradingConfig, load_trading_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_CONFIG_PATH = PROJECT_ROOT / "config" / "backtest.yaml"


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 504
    validation_days: int = 63
    oos_days: int = 63
    step_days: int = 63
    min_train_rows: int = 100
    calibration_method: str = "sigmoid"
    min_threshold_trades: int = 10
    execution_delay_days: int = 1


def load_walk_forward_config() -> WalkForwardConfig:
    with open(BACKTEST_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return WalkForwardConfig(**cfg.get("walk_forward", {}))


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    oos_start: str
    oos_end: str


@dataclass
class WalkForwardResult:
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    trades: list[BTClosedTrade] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)
    # Her pencerenin OOS tahminleri, havuzlanmış (tüm pencerelerden birleşik)
    # AUC/IC hesabı için — bkz. scripts/run_walk_forward.py.
    oos_predictions: list[dict[str, Any]] = field(default_factory=list)


def _daily_rank_ic(oos: pd.DataFrame) -> float | None:
    """Her `feature_date` içinde prob_up ile target_up'ın Spearman
    korelasyonu, günler arası ortalaması. Piyasa-geneli hareketi (o günün
    TÜM hisseleri aynı yöne gitmesi) değil, modelin o gün İÇİNDE hisseleri
    doğru SIRALAYIP sıralamadığını ölçer — AUC'nin tamamlayıcısı, tek bir
    havuzlanmış sayının gizleyebileceği gün-içi sıralama gücünü gösterir."""
    daily = oos.groupby("feature_date").apply(
        lambda g: g["prob_up"].corr(g["target_up"], method="spearman")
        if g["target_up"].nunique() > 1 and len(g) > 2
        else float("nan"),
        include_groups=False,
    )
    daily = daily.dropna()
    return float(daily.mean()) if len(daily) else None


def _pooled_auc(oos: pd.DataFrame) -> float | None:
    if oos["target_up"].nunique() < 2:
        return None
    return float(roc_auc_score(oos["target_up"].astype(int), oos["prob_up"]))


def make_windows(dates: list[str], cfg: WalkForwardConfig) -> list[WalkForwardWindow]:
    dates = sorted(set(dates))
    required = cfg.train_days + cfg.validation_days + cfg.oos_days
    if len(dates) < required:
        return []

    windows: list[WalkForwardWindow] = []
    start = 0
    while start + required <= len(dates):
        train_end_idx = start + cfg.train_days
        validation_end_idx = train_end_idx + cfg.validation_days
        oos_end_idx = validation_end_idx + cfg.oos_days
        windows.append(
            WalkForwardWindow(
                train_start=dates[start],
                train_end=dates[train_end_idx - 1],
                validation_start=dates[train_end_idx],
                validation_end=dates[validation_end_idx - 1],
                oos_start=dates[validation_end_idx],
                oos_end=dates[oos_end_idx - 1],
            )
        )
        start += cfg.step_days
    return windows


def _schedule_orders(
    df: pd.DataFrame,
    decision_date: str,
    portfolio: BacktestPortfolio,
    threshold: float,
    trading_cfg: TradingConfig,
    pending: dict[str, tuple[str, str]],
    decisions: list[dict[str, Any]],
) -> None:
    day = df[df["feature_date"] == decision_date]
    by_symbol = {str(r.ticker): r for r in day.itertuples()}

    # Açık pozisyonlar: sadece çıkış planlanır, yürütme bir sonraki işlem günü olur.
    for symbol, position in list(portfolio.open_positions.items()):
        row = by_symbol.get(symbol)
        if row is None:
            continue
        prob = float(row.prob_up)
        held_days = (pd.Timestamp(decision_date) - pd.Timestamp(position.opened_at)).days
        if held_days >= trading_cfg.max_hold_days or prob < trading_cfg.sell_threshold:
            reason = "max_hold_süresi" if held_days >= trading_cfg.max_hold_days else "prob_düştü"
            pending[symbol] = ("sell", reason)
            decisions.append({
                "date": decision_date, "symbol": symbol, "action": "sat_sinyali",
                "reason": reason, "prob_up": prob,
            })

    # Yeni girişler D kapanışında seçilir, D+1 açılışında yürütülür.
    candidates = day[day["prob_up"] > threshold].sort_values("prob_up", ascending=False)
    for row in candidates.itertuples():
        symbol = str(row.ticker)
        if symbol in portfolio.open_positions or symbol in pending:
            continue
        pending[symbol] = ("buy", "threshold")
        decisions.append({
            "date": decision_date, "symbol": symbol, "action": "al_sinyali",
            "reason": f"prob_up {float(row.prob_up):.3f} > {threshold:.3f}",
            "prob_up": float(row.prob_up),
        })


def _execute_pending(
    execution_date: str,
    df: pd.DataFrame,
    portfolio: BacktestPortfolio,
    pending: dict[str, tuple[str, str]],
    trading_cfg: TradingConfig,
    cost_cfg: BacktestCostConfig,
    decisions: list[dict[str, Any]],
) -> None:
    day = df[df["feature_date"] == execution_date]
    by_symbol = {str(r.ticker): r for r in day.itertuples()}
    # Önce çıkışlar, böylece serbest kalan nakit/pozisyon kapasitesi aynı gün
    # yeni girişlerde kullanılabilir.
    orders = sorted(pending.items(), key=lambda item: 0 if item[1][0] == "sell" else 1)
    for symbol, (side, reason) in orders:
        row = by_symbol.get(symbol)
        if row is None:
            continue
        open_price = float(row.open)
        prob = float(row.prob_up)
        if side == "sell":
            trade = portfolio.sell(symbol, open_price, reason, execution_date, trading_cfg, cost_cfg)
            if trade:
                decisions.append({
                    "date": execution_date, "symbol": symbol, "action": "sat",
                    "reason": reason, "prob_up": prob, "execution": "next_open",
                })
        else:
            ok, msg = portfolio.buy(symbol, open_price, prob, execution_date, trading_cfg, cost_cfg)
            decisions.append({
                "date": execution_date, "symbol": symbol, "action": "al" if ok else "red",
                "reason": msg, "prob_up": prob, "execution": "next_open",
            })
    pending.clear()


def run_walk_forward(
    cfg: WalkForwardConfig | None = None,
    trading_cfg: TradingConfig | None = None,
    cost_cfg: BacktestCostConfig | None = None,
    dataset: pd.DataFrame | None = None,
    bist_only: bool = True,
) -> WalkForwardResult:
    cfg = cfg or load_walk_forward_config()
    trading_cfg = trading_cfg or load_trading_config()
    cost_cfg = cost_cfg or BacktestCostConfig()
    model_cfg = load_model_config()
    training_cfg = model_cfg["training"]
    seed = int(training_cfg.get("random_state", 42))

    df = dataset.copy() if dataset is not None else build_dataset(require_target=False)
    if df.empty:
        return WalkForwardResult()
    if bist_only:
        df = df[df["is_bist"] == 1.0].copy()
    df = df.sort_values(["feature_date", "ticker"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_up", "open"])
    dates = sorted(df["feature_date"].unique().tolist())
    windows = make_windows(dates, cfg)
    if not windows:
        return WalkForwardResult()

    portfolio = BacktestPortfolio(starting_balance=trading_cfg.starting_balance)
    result = WalkForwardResult()
    pending: dict[str, tuple[str, str]] = {}

    for window in windows:
        train = df[(df.feature_date >= window.train_start) & (df.feature_date <= window.train_end)]
        valid = df[(df.feature_date >= window.validation_start) & (df.feature_date <= window.validation_end)]
        oos = df[(df.feature_date >= window.oos_start) & (df.feature_date <= window.oos_end)]
        if len(train) < cfg.min_train_rows or valid.empty or oos.empty:
            continue

        model = build_xgb_model(training_cfg, seed)
        model.fit(
            train[FEATURE_COLUMNS],
            train["target_up"].astype(int),
            eval_set=[(valid[FEATURE_COLUMNS], valid["target_up"].astype(int))],
            verbose=False,
        )

        valid_raw = model.predict_proba(valid[FEATURE_COLUMNS])[:, 1]
        try:
            calibrator = ProbabilityCalibrator(cfg.calibration_method).fit(
                valid_raw, valid["target_up"].astype(int).to_numpy()
            )
            valid_prob = calibrator.transform(valid_raw)
            threshold_info = select_buy_threshold(
                valid_prob,
                valid["target_up"].astype(int).to_numpy(),
                min_trades=cfg.min_threshold_trades,
            )
        except ValueError:
            # Bu pencerenin VALIDATION'ında ne kalibrasyon fit edilebildi (tek
            # sınıf) ne de min_threshold_trades'i sağlayan bir eşik bulundu —
            # yetersiz TRAIN/VALIDATION/OOS verisiyle aynı kategoride: işlem
            # yapılabilir bir sinyal yok, pencere atlanır, koşu çökmez.
            continue
        threshold = float(threshold_info["threshold"])

        oos_raw = model.predict_proba(oos[FEATURE_COLUMNS])[:, 1]
        oos = oos.copy()
        oos["prob_up"] = calibrator.transform(oos_raw)

        window_auc = _pooled_auc(oos)
        window_ic = _daily_rank_ic(oos)
        result.oos_predictions.extend(
            oos[["feature_date", "ticker", "prob_up", "target_up"]].to_dict("records")
        )

        oos_dates = sorted(oos["feature_date"].unique().tolist())
        for day in oos_dates:
            # Dünün sinyalini bugünün AÇILIŞINDA yürüt, sonra bugünün kararına geç.
            if pending:
                _execute_pending(day, oos, portfolio, pending, trading_cfg, cost_cfg, result.decisions)
            day_rows = oos[oos.feature_date == day]
            latest_prices = dict(zip(day_rows.ticker.astype(str), day_rows.close.astype(float)))
            portfolio.record_snapshot(day, latest_prices)
            _schedule_orders(oos, day, portfolio, threshold, trading_cfg, pending, result.decisions)

        result.windows.append({
            "train_start": window.train_start, "train_end": window.train_end,
            "validation_start": window.validation_start, "validation_end": window.validation_end,
            "oos_start": window.oos_start, "oos_end": window.oos_end,
            "train_rows": len(train), "validation_rows": len(valid), "oos_rows": len(oos),
            "threshold": threshold, "calibration": cfg.calibration_method,
            "oos_auc": window_auc, "oos_ic": window_ic,
        })

    result.equity_curve = portfolio.equity_curve
    result.trades = portfolio.closed_trades
    return result
