"""Ön kayıtlı test: docs/experiments/2026-09-27-gunluk-model-durust-test.md
Çalıştırma (repo kökünden): PYTHONPATH=. .venv/bin/python docs/experiments/2026-09-27-gunluk-model-durust-test.py
"""
import dataclasses, json, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import backtest.walk_forward as wfm
from backtest.costs import BacktestCostConfig, execution_price
from backtest.metrics import summarize
from pipeline.dataset import build_dataset
from trading.config import load_trading_config

TC = load_trading_config()  # Midas: 0 komisyon (config/trading.yaml)
CC = BacktestCostConfig(slippage_bps=5.0, spread_bps=10.0)
DATASET = build_dataset(require_target=False)
REAL_BUILD = wfm.build_xgb_model


class RandomModel:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def fit(self, *a, **k):
        return self

    def predict_proba(self, X):
        p = self.rng.random(len(X))
        return np.column_stack([1 - p, p])


def run(seed=None):
    if seed is None:
        wfm.build_xgb_model = REAL_BUILD
    else:
        r = RandomModel(seed)
        wfm.build_xgb_model = lambda *a, **k: r
    res = wfm.run_walk_forward(trading_cfg=TC, cost_cfg=CC, dataset=DATASET)
    return res


def curve(res):
    s = pd.Series({pd.Timestamp(d): v for d, v in res.equity_curve}).sort_index()
    return s


def rand_total(seed):
    c = curve(run(seed))
    return float(c.iloc[-1] / TC.starting_balance - 1) * 100 if len(c) else float("nan")


if __name__ == "__main__":
    res = run()
    eq = curve(res)
    s = summarize(res.equity_curve, res.trades, TC.starting_balance); s.pop("daily_returns", None)
    start, end = eq.index[0], eq.index[-1]
    print(f"OOS {start.date()} -> {end.date()}, {len(res.windows)} pencere, {len(res.trades)} islem")

    # B1: aynı 50 hissenin eşit ağırlıklı al-tut'u
    bist = DATASET[DATASET["is_bist"] == 1.0].copy()
    bist["feature_date"] = pd.to_datetime(bist["feature_date"])
    closes = bist.pivot_table(index="feature_date", columns="ticker", values="close").sort_index()
    opens = bist.pivot_table(index="feature_date", columns="ticker", values="open").sort_index()
    closes = closes.loc[start:end].ffill()
    first_open = opens.loc[start].dropna()
    per = TC.starting_balance / len(first_open)
    qty = {t: per / execution_price(float(p), "buy", CC) for t, p in first_open.items()}
    b1 = closes[list(qty)].mul(pd.Series(qty)).sum(axis=1)
    b1_total = (b1.iloc[-1] / TC.starting_balance - 1) * 100

    # B2: XU100
    try:
        import yfinance as yf
        xu = yf.download("XU100.IS", start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                         progress=False, auto_adjust=False)["Close"].squeeze()
        b2_total = float(xu.iloc[-1] / xu.iloc[0] - 1) * 100
    except Exception as e:  # bilgi amaçlı; kriter değil
        b2_total = float("nan"); print("XU100 alinamadi:", e)

    strat_total = s["total_return_pct"]
    yrs = (end - start).days / 365.25
    ann = lambda t: ((1 + t / 100) ** (1 / yrs) - 1) * 100
    print(f"Strateji: toplam %{strat_total:.1f} (yillik %{ann(strat_total):.1f}), sharpe {s['sharpe_ratio']}, "
          f"maxDD %{s['max_drawdown_pct']['max_drawdown_pct']:.1f}, ucret {s['total_fees']:.0f}")
    print(f"B1 esit agirlikli al-tut: toplam %{b1_total:.1f} (yillik %{ann(b1_total):.1f})")
    print(f"B2 XU100 al-tut: toplam %{b2_total:.1f} (yillik %{ann(b2_total):.1f})")
    print("Yillara gore (strateji / B1):")
    for y in sorted(set(eq.index.year)):
        e, b = eq[eq.index.year == y], b1[b1.index.year == y]
        pe = eq[eq.index.year < y].iloc[-1] if (eq.index.year < y).any() else TC.starting_balance
        pb = b1[b1.index.year < y].iloc[-1] if (b1.index.year < y).any() else TC.starting_balance
        print(f"  {y}: %{(e.iloc[-1]/pe-1)*100:7.1f}  /  %{(b.iloc[-1]/pb-1)*100:7.1f}")

    # Kriter 2: günlük aktif getiri, 20 günlük blok bootstrap
    j = pd.concat([eq.pct_change(), b1.pct_change()], axis=1, keys=["s", "b"]).dropna()
    act = (j["s"] - j["b"]).to_numpy()
    rng = np.random.default_rng(7); B = 20; n = len(act)
    boots = []
    for _ in range(5000):
        idx = np.concatenate([np.arange(k, k + B) for k in rng.integers(0, n - B, n // B + 1)])[:n]
        boots.append(act[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 252 * 100
    print(f"aktif getiri (yilliklandirilmis) ort %{act.mean()*252*100:.1f}, %95 GA [{lo:.1f}, {hi:.1f}]")

    with ProcessPoolExecutor(max_workers=4) as ex:
        rnd = list(ex.map(rand_total, range(30)))
    beaten = sum(strat_total > x for x in rnd)
    print(f"rastgele model toplam getiri medyan %{np.nanmedian(rnd):.1f} (min {np.nanmin(rnd):.1f}, max {np.nanmax(rnd):.1f}); "
          f"strateji {beaten}/30 gecti")

    c1, c2, c3 = strat_total > b1_total, lo > 0, beaten >= 27
    print(f"\nKRITERLER: >B1 {c1} | aktif GA alt>0 {c2} | >=27/30 {c3} => {'GECTI' if c1 and c2 and c3 else 'KALDI'}")
    json.dump(dict(summary=s, b1=b1_total, b2=b2_total, ci=[lo, hi], random=rnd, windows=res.windows,
                   criteria=[bool(c1), bool(c2), bool(c3)]), open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"),
              default=str, indent=1)
