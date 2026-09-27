"""Ön kayıtlı son deneme: docs/experiments/2026-09-27-intraday-son-deneme.md"""
import dataclasses, json, logging, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from backtest.costs import BacktestCostConfig
from backtest.intraday_engine import simulate_intraday
from backtest.intraday_walk_forward import _normalize_days, _spearman_ic
from backtest.metrics import summarize
from pipeline.intraday_dataset import INTRADAY_FEATURE_COLUMNS as F, build_intraday_dataset, load_intraday_model_config
from pipeline.intraday_train_model import fit_intraday_models
from trading.config import load_trading_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
OUT = sys.argv[1]
TC = dataclasses.replace(load_trading_config(), commission_pct=0.0, bsmv_pct_of_commission=0.0, min_commission_try=0.0)
CC = BacktestCostConfig(slippage_bps=5.0, spread_bps=10.0)
CC0 = BacktestCostConfig(0.0, 0.0)
GRID = [0.95, 0.98, 0.99, 0.995, 0.998]
INIT, VAL, TEST = 250, 60, 60
cfg = load_intraday_model_config()

df = build_intraday_dataset(require_target=False).copy()
df["_day"] = _normalize_days(df)
# kesitsel fazla getiri (yalnız hedef bilinen satırlar üzerinden ortalama)
df["excess"] = df["target_return"] - df.groupby("ts")["target_return"].transform("mean")
days = sorted(df["_day"].unique())
pos = pd.Series(np.arange(len(days)), index=pd.DatetimeIndex(days)).reindex(pd.DatetimeIndex(df["_day"])).to_numpy()


def as_excess(d):
    d = d[d["excess"].notna()].copy()
    d["target_return"] = d["excess"]
    d["target_up"] = (d["excess"] > 0).astype(float)
    return d


def net_on(frame, q_thr):
    f = frame.copy(); f["entry_threshold"] = q_thr
    r = simulate_intraday(f, TC, CC)
    return sum(t.net_pnl for t in r.trades), len(r.trades)


parts, folds = [], []
t_start = time.time()
for k, t0 in enumerate(range(INIT, len(days), TEST)):
    ve = t0 + VAL; te = min(ve + TEST, len(days))
    if te <= ve:
        break
    tr = as_excess(df[pos < t0]); va_all = df[(pos >= t0) & (pos < ve)].copy(); te_all = df[(pos >= ve) & (pos < te)].copy()
    va = as_excess(va_all)
    clf, reg = fit_intraday_models(tr, va, cfg)
    vpred = reg.predict(va[F])
    va_all["pred_rod"] = reg.predict(va_all[F]); va_all["prob_up"] = clf.predict_proba(va_all[F])[:, 1]
    choice = []
    for q in GRID:
        thr = float(np.quantile(vpred, q))
        net, n = net_on(va_all, thr)
        choice.append((q, thr, net, n))
    ok = [c for c in choice if c[3] >= 10]
    q, thr, vnet, vn = max(ok, key=lambda c: c[2]) if ok else next(c for c in choice if c[0] == 0.95)
    te_all["pred_rod"] = reg.predict(te_all[F]); te_all["prob_up"] = clf.predict_proba(te_all[F])[:, 1]
    te_all["entry_threshold"] = thr
    sc = te_all[te_all["excess"].notna()]
    folds.append(dict(fold=k, test_start=str(days[ve].date()), test_end=str(days[te - 1].date()), q=q, thr=thr,
                      val_net=vnet, val_n=vn, grid=choice,
                      ic_excess=_spearman_ic(sc["pred_rod"], sc["excess"]), ic_raw=_spearman_ic(sc["pred_rod"], sc["target_return"])))
    print(f"fold {k} {folds[-1]['test_start']}..{folds[-1]['test_end']} secilen q={q} (val net {vnet:.0f} TL, {vn} islem) "
          f"ic_excess={folds[-1]['ic_excess']:.4f} ic_raw={folds[-1]['ic_raw']:.4f}  [{time.time()-t_start:.0f}s]", flush=True)
    parts.append(te_all)

frame = pd.concat(parts).sort_values("ts", kind="stable")
res = simulate_intraday(frame, TC, CC)
s = summarize(res.equity_curve, res.trades, TC.starting_balance); s.pop("daily_returns", None)
res0 = simulate_intraday(frame, TC, CC0)
tr = pd.DataFrame([dict(day=t.opened_at[:10], ret=t.gross_pnl / (t.entry_price * t.quantity)) for t in res0.trades])
print(f"\nBIRINCIL (Midas + slip5/spread10): {len(res.trades)} islem, net %{s['total_return_pct']:.2f}, "
      f"sharpe {s['sharpe_ratio']}, maxDD {s['max_drawdown_pct']['max_drawdown_pct']:.1f}%")
print(f"brut islem-basi: ort {tr.ret.mean()*1e4:.1f} bps, kazanma %{(tr.ret>0).mean()*100:.1f}; yil: "
      + ", ".join(f"{y}: {g.ret.mean()*1e4:.1f} bps (n={len(g)})" for y, g in tr.groupby(tr.day.str[:4])))
rng = np.random.default_rng(7)
daily = tr.groupby("day")["ret"].mean().values
bb = np.array([rng.choice(daily, len(daily)).mean() * 1e4 for _ in range(10000)])
lo, hi = np.percentile(bb, [2.5, 97.5])
print(f"gun-blok bootstrap %95 GA [{lo:.1f}, {hi:.1f}] bps")

# rastgele taban: fold başına modelin giriş oranı
fold_id = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
frame_u = pd.concat(parts)  # parts sırasıyla hizalı
rate = {i: float((p["pred_rod"] > p["entry_threshold"]).mean()) for i, p in enumerate(parts)}


def rand_run(seed):
    r = np.random.default_rng(seed)
    d = frame_u.copy()
    d["pred_rod"] = r.random(len(d))
    d["entry_threshold"] = [1 - rate[i] for i in fold_id]
    d = d.sort_values("ts", kind="stable")
    rr = simulate_intraday(d, TC, CC0)
    v = [t.gross_pnl / (t.entry_price * t.quantity) for t in rr.trades]
    return float(np.mean(v) * 1e4) if v else float("nan")


with ProcessPoolExecutor(max_workers=4) as ex:
    rnd = list(ex.map(rand_run, range(30)))
model_bps = tr.ret.mean() * 1e4
beaten = sum(model_bps > x for x in rnd)
print(f"rastgele taban medyan {np.nanmedian(rnd):.1f} bps; model {model_bps:.1f} -> {beaten}/30 rastgeleyi gecti")

c1, c2, c3 = s["total_return_pct"] > 0, lo > 0, beaten >= 27
print(f"\nKRITERLER: net>0 {c1} | GA alt>0 {c2} | >=27/30 {c3} => {'GECTI' if c1 and c2 and c3 else 'KALDI'}")
json.dump(dict(folds=folds, summary=s, n=len(res.trades), model_bps=model_bps, ci=[lo, hi], random=rnd,
               criteria=[bool(c1), bool(c2), bool(c3)]), open(OUT, "w"), default=str, indent=1)
