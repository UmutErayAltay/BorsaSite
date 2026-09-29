# Borsa AI — Data pipeline, prediction and paper trading for BIST + US stocks

![Borsa AI dashboard](docs/screenshots/dashboard.png)

## Description

A data pipeline that collects daily price and news data for BIST and US stocks,
runs sentiment analysis and machine learning to produce a 5-day return
probability. Data comes from yfinance (daily + hourly bars), 9 RSS news sources
and KAP's official JSON API; news is converted into a `-1..+1` sentiment score
by FinBERT (EN) and a Turkish BERT model. The XGBoost model combines technical
indicators (RSI, MACD, SMA) with sentiment to answer "will this stock beat the
BIST median?". Predictions feed a threshold-based **paper** trading engine that
accounts for commission/BSMV and risk limits — there is no order-placing code in
this repository. The same strategy is stress-tested against history (same-bar
and walk-forward engines) and in the pre-registered honesty test of 2026-09-27
it **lost to passive buy-and-hold**; that result is stated in this README
rather than hidden.

> Türkçe sürüm: [README.md](README.md)

## Setup

```bash
cd "borsa projesi"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put `DATABASE_URL` and (for sentiment analysis) `HF_TOKEN=hf_...` into `.env`
(`https://huggingface.co/settings/tokens`).

**A database is required** — SQLite was dropped entirely. Locally:

```bash
docker compose up -d db
```

`DATABASE_URL=postgresql://borsa:borsa@localhost:5432/borsa`. No migration step:
every script and the API startup run `pipeline/db.py::SCHEMA_SQL` automatically.

## Price data (F0)

First run (default: last 2 years):

```bash
python scripts/run_fetch_prices.py
```

Daily update (last 7 days):

```bash
python scripts/run_fetch_prices.py --incremental 7
```

Other flags: `--period` (`1y`, `2y`, `max` — cannot be combined with
`--incremental`), `-v`.

One transaction per symbol: a failure on a single symbol does not break the
whole loop or the counters.

## News data (F1)

```bash
python scripts/run_fetch_news.py
```

Test (at most 5 items per feed):

```bash
python scripts/run_fetch_news.py --max-per-feed 5
```

Summary:

```bash
python scripts/inspect_news.py
```

Sources live in `config/news_feeds.yaml`: AA, Bloomberg HT, Dünya, Google News
KAP, Google News BIST, CNBC, MarketWatch, Yahoo Finance, Google News (S&P 500 /
Wall Street).

**KAP:** `python scripts/run_kap_sync.py --days 7` — pulls disclosures from the
official web API (`byCriteria`) and matches symbols directly via the
`stockCodes` field. The Google News KAP stream stays as an extra source.

**News–symbol linking:** `pipeline/entity_linker.py` uses regex + company names
+ `config/symbol_aliases.yaml`, filtered by `config/entity_blocklist.yaml` so
short words (`AI`, `US`, `CEO`, `IPO`, `SPK`, `KAP`, …) are not mistaken for
tickers.

```bash
python scripts/run_relink.py        # re-link existing RSS news
```

## Sentiment analysis (F2)

First run (models download, ~500MB–1GB):

```bash
python scripts/clear_hf_locks.py
python scripts/run_sentiment.py --limit 20
```

All news:

```bash
python scripts/run_sentiment.py
```

Summary:

```bash
python scripts/inspect_sentiment.py
```

Flags: `--limit N` (first-run smoke test), `--force` (reprocess),
`--skip-aggregate` (skip the daily per-symbol rollup), `-v`.

| Language | Model |
|----------|-------|
| EN | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — financial news |
| TR | [savasy/bert-base-turkish-sentiment-cased](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased) — general Turkish sentiment |

Output: a **-1 … +1** score per article (`news_sentiment`), plus the
per-symbol daily average (`sentiment_daily`). Model settings live in
`config/sentiment.yaml`.

If downloading hangs: close every `python` process, run `clear_hf_locks.py`,
retry (one terminal at a time).

## Prediction model (F3)

Technical indicators (RSI14, MACD, SMA20/50, return and volume ratios) +
sentiment → an **XGBoost pair**:

- **Classifier** (`prob_up`): whether the 5-day `forward_return` (D+1 open →
  D+5 close) is above that day's **BIST median**. Not absolute direction — a
  **cross-sectional** label stripped of market-wide movement.
- **Regressor** (`expected_return`): the expected magnitude of the return for
  the same trade. Direction probability alone does not answer "how much do we
  make?".

```bash
python scripts/run_train.py
python scripts/run_predict.py
python scripts/inspect_predictions.py
```

Model: `data/models/xgb_up.pkl` | Predictions: the `predictions` table.
Parameters live in `config/model.yaml` (`training::xgb` and
`training::xgb_regressor` are separate blocks; walk-forward uses early
stopping).

**Warning:** Probabilities are trained on historical data; not a guarantee.

## Dashboard (F4)

```bash
pip install -r requirements-web.txt
python scripts/run_server.py
```

Open **http://127.0.0.1:8000** — single page, two tabs: **Tahminler**
(predictions table; per-symbol detail with 1h/1d/1w/1m/1y chart + period
change chart + latest news) and **Bot** (portfolio, BIST 100 comparison, open
positions, closed trades).

REST API:

| Endpoint | Description |
|---|---|
| `/api/health`, `/api/stats` | Health and table row counts |
| `/api/symbols`, `/api/symbols/{ticker}` | Symbol list / detail (price, news, prediction) |
| `/api/predictions` | Latest predictions (`market`, `sort`, `order`, `limit`) |
| `/api/chart/{ticker}` | Chart data (`interval=1h\|1d\|1w\|1m\|1y`) |
| `/api/prices/{ticker}` | Daily price series (`days`, 7–730) |
| `/api/news` | News feed (`ticker`, `limit`) |
| `/api/portfolio` | Balance, open positions, total assets |
| `/api/portfolio/live` | Same, priced live from yfinance ("check now" button) |
| `/api/portfolio/history` | Daily equity snapshots (`days`) |
| `/api/portfolio/benchmark` | Portfolio curve normalized against BIST 100 (`XU100.IS`) |
| `/api/trades` | Closed trades + gross/net P&L, fees, win rate |

The hourly chart is fetched live from yfinance (60 days); other intervals are
resampled from the database.

## Paper trading (F5)

Uses existing predictions (`prob_up`) to trade **BIST symbols only** with a fake
10,000 TL balance, threshold-based — it never places a real order. Commission
and BSMV are charged on every trade. Every decision (buy, sell, hold, **reject**)
is written to the `trade_decisions` table with a human-readable reason.

```bash
python scripts/run_trading.py
```

The dashboard's "Bot" tab shows the current balance, open positions, closed
trades (gross/net P&L and paid commission separately) and the portfolio curve
next to BIST 100 on the same chart.

**Warning:** Purely a simulation; not investment advice.

## Risk management (Phase 6–7)

All in `config/trading.yaml`; defaults are backward compatible (0.0/0 = off).

| Setting | Default | Effect |
|---|---|---|
| `max_open_positions` | 8 | Max 8 open positions at a time |
| `max_position_pct` | 15% | Per-position balance cap |
| `max_portfolio_exposure_pct` | 90% | Total cash + positions cap |
| `max_hold_days` | 10 | Max holding period |
| `stop_loss_pct` | 7% | Price-based stop-loss |
| `take_profit_pct` | 15% | Price-based take-profit |
| `cooldown_days_after_exit` | 3 days | No re-entry into a sold symbol |
| `min_expected_edge_pct` | 0.0 | Expected return ≥ round-trip cost + safety margin |
| `min_position_value_try` | 500 TL | Minimum trade size |

Exit rules are evaluated in order: **stop-loss → take-profit → max_hold_days →
`prob_up` dropped**.

Commission side: `commission_pct: 0.0`, `bsmv_pct_of_commission: 5.0` — modelled
on BIST commission-free trading at Midas/Enpara (switching to a bank broker:
~0.2% + BSMV + minimum fee).

## Historical backtest (Phase 1–3)

A separate engine that genuinely replays a past date range, unlike the live
motor: `backtest/engine.py`. It accounts for commission + BSMV + slippage +
spread, and produces Sharpe/Sortino/max drawdown/profit factor metrics plus a
buy-and-hold benchmark. **Deliberate limitation:** it uses a SINGLE (current,
trained on all history) model — results are therefore in-sample.

```bash
python scripts/run_backtest.py --scenario all
python scripts/run_backtest.py --start-date 2019-01-01 --end-date 2026-09-01
python scripts/run_backtest.py --no-db        # do not write to the DB
```

Cost scenarios in `config/backtest.yaml::scenarios`: `base` (0/0 bps),
`conservative` (10/20), `stress` (30/50). Commission/BSMV come from the same
place as the live motor (`config/trading.yaml`). Results are written to
`reports/` as JSON/CSV/HTML (`backtest/reports.py`).

For detail, architecture and the other deliberate limitations (same-bar
execution): `docs/BACKTEST.md`. System audit: `docs/BACKTEST_AUDIT.md`.

## Walk-forward backtest (Phase 4)

Removes the in-sample limitation of Phases 1–3: it measures how a model that
never saw the future would have performed historically. Data is split into
TRAIN → VALIDATION → OOS windows, the model is trained on each window's TRAIN
part only, and calibration and the entry threshold are selected on VALIDATION
only — OOS labels are never used for any parameter choice. The signal is
produced at the close of D and executed at the open of D+1 (see the
`backtest/walk_forward.py` docstring).

```bash
python scripts/run_walk_forward.py                    # config/backtest.yaml::default_scenario
python scripts/run_walk_forward.py --scenario stress   # extra slippage/spread
```

Window sizes live in `config/backtest.yaml::walk_forward` (train 250 / val 60 /
oos 50 days, 50-day step, expanding window).

**Warning:** Past performance does not guarantee future results; purely a
simulation, not investment advice.

## Honesty test: does the strategy actually add value? — **IT FAILED**

`docs/experiments/2026-09-27-gunluk-model-durust-test.md` had its criteria
committed **before** the run (pre-registration) and was executed afterwards.

OOS 2019-02-05 → 2026-08-06, 25 windows, 918 trades, same 50 BIST stocks:

| | Total | Annual |
|---|---|---|
| Strategy | %120.2 | %11.1 (Sharpe 0.62, max drawdown %24.9) |
| Equal-weight buy-and-hold (same 50 stocks) | %2032.2 | %50.4 |
| XU100 buy-and-hold (price index) | %1246.9 | %41.4 |
| Random model (30 runs) median | %226.0 | — |

Average annual active return **-22.8%**; the strategy beat only 4 of 30 random
models (i.e. worse than random). **Verdict: the daily strategy does not add
value by stock selection; passive index investing is clearly superior.** Known
biases (survivorship, non-interest-bearing idle cash) are documented.

This README does not claim the model works — it was tested, it lost, and the
result was left in place.

## Intraday track (Phase 9–10) — **CLOSED**

Not just running the daily model more often: a genuinely separate hourly model
was built:

```bash
python scripts/run_fetch_intraday_prices.py --interval 1h
python -m pipeline.intraday_train_model          # no script, module is run directly
python scripts/run_intraday_walk_forward.py
```

547,058 real 1h bars (2023-10-27 → 2026-09-25) were fetched for 101 symbols;
the target is intraday only (overnight jumps are not targeted) and the strategy
forces an end-of-day close — the model has no opinion about overnight risk.

Pre-registered test (`docs/experiments/2026-09-27-intraday-son-deneme.md`):
343 trades, net **-11.18%** at primary costs. The model ranks about 8 bps better
than random selection (real but small information), but random entry itself is
-13.6 bps — in this universe, buying at the t+1 open and selling at the close
loses money on average. **Verdict (per the pre-registration): the intraday
track was closed and will not go live.**

The code stays (`pipeline/intraday_*.py`, `backtest/intraday_*.py`) — closed,
not deleted.

## Intraday news watch

The daily cycle (F5) runs only after the close — a bad KAP disclosure during
the day stays invisible until the next day. This layer adds **no new buys** to
the daily prediction/trading cycle; it scans open positions against KAP a few
times per day and exits early when a strongly negative item appears (Turkish
BERT score below a threshold, default -0.5) — a risk-reduction layer, not a new
trading strategy.

```bash
python scripts/run_intraday_watch.py                        # default threshold -0.5
python scripts/run_intraday_watch.py --negative-threshold -0.3
```

`.github/workflows/intraday-watch.yml` runs it hourly on weekdays while BIST is
open (10:30–18:30 Istanbul). There is NO intraday price feed — the early exit is
marked as executed at the last known close (deliberate limitation, see the
`pipeline/intraday_watch.py` docstring).

## Daily pipeline

**News–symbol linking + the full automatic flow**
(price → news → KAP → link → sentiment → train → predict → trade):

```bash
python scripts/run_daily.py
```

Flags: `--days N` (price window, default 7), `--skip-sentiment`, `--skip-train`,
`--skip-predict`, `--skip-trading`, `--no-relink`.

`run_daily.py` retrains the model on every run (XGBoost training takes seconds)
because GitHub Actions checks out fresh each time, so the trained model file
never persists; otherwise the predict step fails with `FileNotFoundError`. Use
`--skip-train` to skip it.

Windows scheduler (daily at 19:00):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_task_scheduler.ps1
```

Logs: `logs/daily_*.log`. Each step's duration and error come back in the
summary JSON.

## Symbol list

`config/symbols.yaml` — **50 BIST** (`.IS` suffix, TRY) + **51 US** stocks
(101 total), selected as liquid stocks by market cap.

## Tests

```bash
pytest
```

208 tests. `tests/conftest.py` runs every test against a **real local
Postgres** and always rolls back at the end (no commit) — tests do not pollute
each other and no extra test-DB dependency is needed. The tests refuse to start
if `DATABASE_URL` is not localhost.

## Live deployment

The dashboard and the daily pipeline run in two different places — Render Cron
Jobs require payment details (minimum $1/month), so the daily job was moved to
GitHub Actions, which is free.

1. Create a Supabase project and apply `pipeline/db.py::SCHEMA_SQL`.
2. **Dashboard (Render):** bind this repo as a Blueprint (`render.yaml`) in
   Render and enter `DATABASE_URL` (the Supabase connection string) manually on
   the `borsa-ai-dashboard` service. Build installs `requirements-web.txt`, start
   command is `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
3. **Daily pipeline (GitHub Actions):** Repo → Settings → Secrets and variables
   → Actions → add the `DATABASE_URL` and `HF_TOKEN` secrets.
   `.github/workflows/daily-trading.yml` runs automatically every weekday after
   the BIST close (19:00 Istanbul = 16:00 UTC). It can also be triggered
   manually with "Run workflow" in the Actions tab.
   `.github/workflows/intraday-watch.yml` uses the same two secrets and needs no
   extra setup.

## Phase status

| Phase | Content | Status |
|---|---|---|
| F0 | Daily price data (yfinance) | ✓ |
| F1 | RSS news + KAP pipeline, news–symbol linking | ✓ |
| F2 | FinBERT / TR BERT sentiment analysis | ✓ |
| F3 | XGBoost + technical indicators | ✓ |
| F4 | FastAPI + web dashboard | ✓ |
| F5 | Paper trading engine | ✓ |
| Phase 1–3 | Historical backtest (same-bar, single model) | ✓ |
| Phase 4 | Walk-forward backtest (TRAIN/VALIDATION/OOS) | ✓ |
| Phase 5 | Calibration + threshold selection (val only) | ✓ |
| Phase 6 | Risk management (stop-loss / take-profit / cooldown) | ✓ |
| Phase 7 | Expected return magnitude model + edge filter | ✓ |
| Phase 8 | Dashboard + reporting | ✓ |
| Phase 9–10 | Intraday architecture + backtest | ✗ CLOSED — failed its acceptance criteria |

Remaining deliberate gaps: the new feature-group experiments of Phase 7
(momentum / volatility / market context) and the other items in
`BACKTEST_AUDIT.md §11`.

## Disclaimer

**Predictions are for information only; not investment advice.** Entirely a
simulation, there is no real order placement. Past performance does not
guarantee future results — this project's own test shows the strategy lost to
passive buy-and-hold.
