"""Borsa AI — REST API ve dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.chart_data import INTERVALS, get_chart_data
from pipeline.db import get_connection, init_schema
from trading.config import load_trading_config
from trading.portfolio import current_price, ensure_portfolio, get_state as get_trading_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(
    title="Borsa AI",
    description="BIST + ABD hisse analizi, haber sentiment ve tahmin API",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _init_schema_on_startup() -> None:
    with get_connection() as conn:
        init_schema(conn)


def _rows(cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cursor.fetchall()]


@app.get("/")
def dashboard():
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Dashboard bulunamadı")
    return FileResponse(index)


@app.get("/api/health")
def health():
    with get_connection() as conn:
        sym = conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"]
    return {"status": "ok", "symbols": sym}


@app.get("/api/stats")
def stats():
    with get_connection() as conn:
        q = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM symbols) AS symbols,
                (SELECT COUNT(*) FROM prices_daily) AS prices,
                (SELECT COUNT(*) FROM news_raw) AS news,
                (SELECT COUNT(*) FROM news_sentiment) AS sentiment,
                (SELECT COUNT(*) FROM predictions) AS predictions
            """
        ).fetchone()
    return dict(q)


@app.get("/api/symbols")
def list_symbols(market: str | None = Query(None, description="BIST veya US")):
    sql = "SELECT id, ticker, market, currency, name, sector FROM symbols WHERE 1=1"
    params: list[Any] = []
    if market:
        sql += " AND market = ?"
        params.append(market.upper() if market.upper() == "BIST" else "US")
    sql += " ORDER BY ticker"
    with get_connection() as conn:
        rows = _rows(conn.execute(sql, params))
    return {"items": rows, "count": len(rows)}


@app.get("/api/predictions")
def list_predictions(
    market: str | None = None,
    sort: str = Query("prob_up", pattern="^(prob_up|ticker)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    limit: int = Query(100, ge=1, le=200),
):
    sql = """
        SELECT
            s.ticker,
            s.market,
            s.name,
            p.feature_date,
            p.target_date,
            p.prob_up,
            p.predicted_up,
            p.model_version,
            (
                SELECT pd.close FROM prices_daily pd
                WHERE pd.symbol_id = s.id
                ORDER BY pd.date DESC LIMIT 1
            ) AS last_close,
            (
                SELECT pd.date FROM prices_daily pd
                WHERE pd.symbol_id = s.id
                ORDER BY pd.date DESC LIMIT 1
            ) AS last_price_date,
            (
                SELECT sd.avg_score FROM sentiment_daily sd
                WHERE sd.symbol_id = s.id
                ORDER BY sd.date DESC LIMIT 1
            ) AS sentiment
        FROM predictions p
        JOIN symbols s ON s.id = p.symbol_id
        WHERE p.feature_date = (
            SELECT MAX(p2.feature_date) FROM predictions p2
            WHERE p2.symbol_id = p.symbol_id
        )
    """
    params: list[Any] = []
    if market:
        sql += " AND s.market = ?"
        params.append(market.upper() if market.upper() == "BIST" else "US")

    sort_col = "p.prob_up" if sort == "prob_up" else "s.ticker"
    sql += f" ORDER BY {sort_col} {order.upper()} LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = _rows(conn.execute(sql, params))
    return {"items": rows, "count": len(rows)}


@app.get("/api/symbols/{ticker}")
def symbol_detail(ticker: str):
    ticker = ticker.upper()
    with get_connection() as conn:
        sym = conn.execute(
            "SELECT * FROM symbols WHERE ticker = ?", (ticker,)
        ).fetchone()
        if not sym:
            raise HTTPException(404, f"Sembol bulunamadı: {ticker}")
        symbol_id = sym["id"]

        pred = conn.execute(
            """
            SELECT * FROM predictions
            WHERE symbol_id = ?
            ORDER BY feature_date DESC LIMIT 1
            """,
            (symbol_id,),
        ).fetchone()

        price = conn.execute(
            """
            SELECT date, close, volume FROM prices_daily
            WHERE symbol_id = ?
            ORDER BY date DESC LIMIT 1
            """,
            (symbol_id,),
        ).fetchone()

        news = _rows(
            conn.execute(
                """
                SELECT n.title, n.source, n.published_at, n.url, s.score, s.label
                FROM news_symbol_links l
                JOIN news_raw n ON n.id = l.news_id
                LEFT JOIN news_sentiment s ON s.news_id = n.id
                WHERE l.symbol_id = ?
                ORDER BY n.published_at DESC
                LIMIT 10
                """,
                (symbol_id,),
            )
        )

    return {
        "symbol": dict(sym),
        "prediction": dict(pred) if pred else None,
        "latest_price": dict(price) if price else None,
        "news": news,
    }


@app.get("/api/chart/{ticker}")
def symbol_chart(
    ticker: str,
    interval: str = Query("1d", pattern="^(1h|1d|1w|1m|1y)$"),
):
    ticker = ticker.upper()
    if interval not in INTERVALS:
        raise HTTPException(400, f"interval: 1h, 1d, 1w, 1m, 1y")
    with get_connection() as conn:
        sym = conn.execute(
            "SELECT id, ticker, name, market FROM symbols WHERE ticker = ?", (ticker,)
        ).fetchone()
        if not sym:
            raise HTTPException(404, f"Sembol bulunamadı: {ticker}")
        try:
            data = get_chart_data(ticker, int(sym["id"]), interval)
        except Exception as e:
            raise HTTPException(502, f"Grafik verisi alınamadı: {e}") from e
    return {"ticker": ticker, "name": sym["name"], "market": sym["market"], **data}


@app.get("/api/portfolio")
def portfolio_summary():
    with get_connection() as conn:
        cfg = load_trading_config()
        ensure_portfolio(conn, cfg.starting_balance)
        state = get_trading_state(conn)
        positions = []
        positions_value = 0.0
        for p in state.open_positions:
            sym = conn.execute("SELECT ticker, name FROM symbols WHERE id = ?", (p.symbol_id,)).fetchone()
            price = current_price(conn, p.symbol_id, p.entry_price)
            market_value = price * p.quantity
            positions_value += market_value
            positions.append({
                "ticker": sym["ticker"] if sym else "?",
                "name": sym["name"] if sym else None,
                "entry_price": p.entry_price,
                "quantity": round(p.quantity, 6),
                "current_price": price,
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(market_value - p.entry_price * p.quantity, 2),
                "opened_at": p.opened_at,
            })
    return {
        "balance": round(state.balance, 2),
        "starting_balance": state.starting_balance,
        "positions_value": round(positions_value, 2),
        "total_value": round(state.balance + positions_value, 2),
        "open_positions": positions,
    }


@app.get("/api/portfolio/history")
def portfolio_history(days: int = Query(365, ge=1, le=3650)):
    with get_connection() as conn:
        rows = _rows(
            conn.execute(
                """
                SELECT snapshot_date, balance, positions_value, total_value
                FROM portfolio_snapshots
                ORDER BY snapshot_date DESC
                LIMIT ?
                """,
                (days,),
            )
        )
    rows.reverse()
    return {"items": rows, "count": len(rows)}


@app.get("/api/trades")
def trade_history(limit: int = Query(50, ge=1, le=200)):
    with get_connection() as conn:
        rows = _rows(
            conn.execute(
                """
                SELECT t.*, s.ticker, s.name
                FROM trades t
                JOIN symbols s ON s.id = t.symbol_id
                ORDER BY t.closed_at DESC, t.id DESC
                LIMIT ?
                """,
                (limit,),
            )
        )
        totals = conn.execute(
            """
            SELECT
                COUNT(*) AS trade_count,
                COALESCE(SUM(net_pnl), 0) AS total_net_pnl,
                COALESCE(SUM(fees_paid), 0) AS total_fees,
                COALESCE(SUM(gross_pnl), 0) AS total_gross_pnl,
                COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
            FROM trades
            """
        ).fetchone()
    trade_count = int(totals["trade_count"])
    return {
        "items": rows,
        "count": len(rows),
        "totals": {
            "trade_count": trade_count,
            "total_gross_pnl": round(float(totals["total_gross_pnl"]), 2),
            "total_fees": round(float(totals["total_fees"]), 2),
            "total_net_pnl": round(float(totals["total_net_pnl"]), 2),
            "win_rate": round(int(totals["wins"]) / trade_count * 100, 1) if trade_count else None,
        },
    }


@app.get("/api/prices/{ticker}")
def symbol_prices(ticker: str, days: int = Query(90, ge=7, le=730)):
    ticker = ticker.upper()
    with get_connection() as conn:
        sym = conn.execute(
            "SELECT id FROM symbols WHERE ticker = ?", (ticker,)
        ).fetchone()
        if not sym:
            raise HTTPException(404, f"Sembol bulunamadı: {ticker}")
        rows = _rows(
            conn.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM prices_daily
                WHERE symbol_id = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (sym["id"], days),
            )
        )
    rows.reverse()
    return {
        "ticker": ticker,
        "items": rows,
        "dates": [r["date"] for r in rows],
        "close": [r["close"] for r in rows],
    }


@app.get("/api/news")
def list_news(
    ticker: str | None = None,
    limit: int = Query(30, ge=1, le=100),
):
    with get_connection() as conn:
        if ticker:
            ticker = ticker.upper()
            sym = conn.execute(
                "SELECT id FROM symbols WHERE ticker = ?", (ticker,)
            ).fetchone()
            if not sym:
                raise HTTPException(404, f"Sembol bulunamadı: {ticker}")
            rows = _rows(
                conn.execute(
                    """
                    SELECT n.id, n.title, n.source, n.published_at, n.url,
                           n.language, s.score AS sentiment_score, s.label AS sentiment_label,
                           GROUP_CONCAT(sym.ticker) AS symbols
                    FROM news_raw n
                    JOIN news_symbol_links l ON l.news_id = n.id
                    JOIN symbols sym ON sym.id = l.symbol_id
                    LEFT JOIN news_sentiment s ON s.news_id = n.id
                    WHERE l.symbol_id = ?
                    GROUP BY n.id
                    ORDER BY n.published_at DESC
                    LIMIT ?
                    """,
                    (sym["id"], limit),
                )
            )
        else:
            rows = _rows(
                conn.execute(
                    """
                    SELECT n.id, n.title, n.source, n.published_at, n.url,
                           n.language, s.score AS sentiment_score, s.label AS sentiment_label
                    FROM news_raw n
                    LEFT JOIN news_sentiment s ON s.news_id = n.id
                    ORDER BY n.published_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            )
    return {"items": rows, "count": len(rows)}
