"""Veritabanı bağlantısı ve şema (PostgreSQL — Supabase'te canlı, yerelde
docker-compose::db). SQLite tamamen bırakıldı: Render Cron Job'lar kalıcı
disk kullanamıyor, bu proje artık bir ağ veritabanına ihtiyaç duyuyor."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Iterator

import psycopg
from psycopg.rows import dict_row

DEFAULT_DATABASE_URL = "postgresql://borsa:borsa@localhost:5432/borsa"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS symbols (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE,
    market TEXT NOT NULL,
    currency TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prices_daily (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    adj_close REAL,
    volume REAL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(symbol_id, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices_daily(symbol_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_symbols_market ON symbols(market);

CREATE TABLE IF NOT EXISTS news_raw (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    feed_id TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT NOT NULL UNIQUE,
    published_at TEXT,
    language TEXT NOT NULL DEFAULT 'tr',
    content TEXT,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS news_symbol_links (
    news_id INTEGER NOT NULL REFERENCES news_raw(id) ON DELETE CASCADE,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    match_reason TEXT,
    PRIMARY KEY (news_id, symbol_id)
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_raw(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_source ON news_raw(source);
CREATE INDEX IF NOT EXISTS idx_news_links_symbol ON news_symbol_links(symbol_id);

CREATE TABLE IF NOT EXISTS news_sentiment (
    news_id INTEGER PRIMARY KEY REFERENCES news_raw(id) ON DELETE CASCADE,
    score REAL NOT NULL,
    label TEXT NOT NULL,
    positive_prob REAL,
    negative_prob REAL,
    neutral_prob REAL,
    model TEXT NOT NULL,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sentiment_daily (
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    avg_score REAL NOT NULL,
    news_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, date)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_daily_date ON sentiment_daily(date DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    feature_date TEXT NOT NULL,
    target_date TEXT,
    prob_up REAL NOT NULL,
    predicted_up INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(symbol_id, feature_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol_id, feature_date DESC);

CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    starting_balance NUMERIC(14,2) NOT NULL,
    balance NUMERIC(14,2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    entry_price NUMERIC(14,4) NOT NULL,
    quantity NUMERIC(14,4) NOT NULL,
    entry_prob_up REAL NOT NULL,
    opened_at DATE NOT NULL,
    entry_fee NUMERIC(14,2) NOT NULL,
    UNIQUE(symbol_id)
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    entry_price NUMERIC(14,4) NOT NULL,
    exit_price NUMERIC(14,4) NOT NULL,
    quantity NUMERIC(14,4) NOT NULL,
    gross_pnl NUMERIC(14,2) NOT NULL,
    fees_paid NUMERIC(14,2) NOT NULL,
    net_pnl NUMERIC(14,2) NOT NULL,
    exit_reason TEXT NOT NULL,
    opened_at DATE NOT NULL,
    closed_at DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_decisions (
    id SERIAL PRIMARY KEY,
    decision_date DATE NOT NULL,
    symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    prob_up REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_date ON trade_decisions(decision_date DESC);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,
    balance NUMERIC(14,2) NOT NULL,
    positions_value NUMERIC(14,2) NOT NULL,
    total_value NUMERIC(14,2) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(snapshot_date DESC);

-- Faz 1: her backtest çalıştırmasının "bu sonucu hangi model/config üretti"
-- sorusuna cevap verebilmesi için (bkz. docs/BACKTEST_AUDIT.md §19).
CREATE TABLE IF NOT EXISTS backtest_runs (
    id SERIAL PRIMARY KEY,
    scenario TEXT NOT NULL,
    model_version TEXT,
    config_hash TEXT NOT NULL,
    start_date DATE,
    end_date DATE,
    starting_balance NUMERIC(14,2) NOT NULL,
    slippage_bps REAL NOT NULL DEFAULT 0,
    spread_bps REAL NOT NULL DEFAULT 0,
    metrics JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    entry_price NUMERIC(14,4) NOT NULL,
    exit_price NUMERIC(14,4) NOT NULL,
    quantity NUMERIC(14,4) NOT NULL,
    gross_pnl NUMERIC(14,2) NOT NULL,
    fees_paid NUMERIC(14,2) NOT NULL,
    net_pnl NUMERIC(14,2) NOT NULL,
    exit_reason TEXT NOT NULL,
    opened_at DATE NOT NULL,
    closed_at DATE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(run_id);

CREATE TABLE IF NOT EXISTS backtest_equity (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    total_value NUMERIC(14,2) NOT NULL,
    UNIQUE(run_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_backtest_equity_run ON backtest_equity(run_id, snapshot_date);
"""


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


class ConnWrapper:
    """sqlite3.Connection'ın `execute()`/`executemany()`/`executescript()`
    kısayollarını taklit eden ince bir psycopg sarmalayıcı — mevcut ~15
    çağrı noktası (api/main.py, pipeline/*.py, scripts/inspect_*.py) `?`
    placeholder + `row["col"]` dict-erişimi kullanıyor, hiçbiri tuple-index
    erişimi (`row[0]`) yapmıyor (kod taramasıyla doğrulandı) — bu yüzden bu
    sarmalayıcı dışında TEK BİR dosyaya bile dokunmadan taşıma tamamlanır."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple | None = None) -> psycopg.Cursor:
        cur = self._conn.cursor(row_factory=dict_row)
        cur.execute(sql.replace("?", "%s"), params)
        return cur

    def executemany(self, sql: str, seq_of_params: Iterator[tuple]) -> psycopg.Cursor:
        cur = self._conn.cursor()
        cur.executemany(sql.replace("?", "%s"), list(seq_of_params))
        return cur

    def executescript(self, sql: str) -> None:
        cur = self._conn.cursor()
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


@contextmanager
def get_connection() -> Generator[ConnWrapper, None, None]:
    # prepare_threshold=None: Supabase'in transaction-mode pooler'ı (port 6543,
    # Supavisor/PgBouncer) her sorguyu farklı bir arka uç bağlantısına
    # yönlendirebilir — psycopg'nin varsayılan server-side prepared statement
    # davranışı bu modda "prepared statement already exists" hatasına yol
    # açar (canlıda gerçekleşti, GitHub Actions log'unda doğrulandı).
    raw = psycopg.connect(get_database_url(), prepare_threshold=None)
    wrapper = ConnWrapper(raw)
    try:
        yield wrapper
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def init_schema(conn: ConnWrapper) -> None:
    conn.executescript(SCHEMA_SQL)


def upsert_symbol(
    conn: ConnWrapper,
    ticker: str,
    market: str,
    currency: str,
    name: str | None = None,
    sector: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO symbols (ticker, market, currency, name, sector, updated_at)
        VALUES (?, ?, ?, ?, ?, NOW())
        ON CONFLICT(ticker) DO UPDATE SET
            market = excluded.market,
            currency = excluded.currency,
            name = COALESCE(excluded.name, symbols.name),
            sector = COALESCE(excluded.sector, symbols.sector),
            updated_at = NOW()
        """,
        (ticker, market, currency, name, sector),
    )
    row = conn.execute("SELECT id FROM symbols WHERE ticker = ?", (ticker,)).fetchone()
    return int(row["id"])


def upsert_prices(conn: ConnWrapper, symbol_id: int, rows: Iterator[tuple]) -> int:
    """rows: (date, open, high, low, close, adj_close, volume)"""
    cur = conn.executemany(
        """
        INSERT INTO prices_daily (symbol_id, date, open, high, low, close, adj_close, volume, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW())
        ON CONFLICT(symbol_id, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            adj_close = excluded.adj_close,
            volume = excluded.volume,
            fetched_at = NOW()
        """,
        ((symbol_id, *r) for r in rows),
    )
    return cur.rowcount


def count_symbols(conn: ConnWrapper) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"])


def count_prices(conn: ConnWrapper) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM prices_daily").fetchone()["n"])


def count_news(conn: ConnWrapper) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM news_raw").fetchone()["n"])


def count_news_links(conn: ConnWrapper) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM news_symbol_links").fetchone()["n"])


def insert_news(
    conn: ConnWrapper,
    source: str,
    feed_id: str | None,
    title: str,
    summary: str | None,
    url: str,
    published_at: str | None,
    language: str,
    content: str | None = None,
) -> int | None:
    """Yeni haber ekler; URL zaten varsa None döner."""
    cur = conn.execute(
        """
        INSERT INTO news_raw
            (source, feed_id, title, summary, url, published_at, language, content, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW())
        ON CONFLICT (url) DO NOTHING
        RETURNING id
        """,
        (source, feed_id, title, summary, url, published_at, language, content),
    )
    row = cur.fetchone()
    return int(row["id"]) if row else None


def get_news_id_by_url(conn: ConnWrapper, url: str) -> int | None:
    row = conn.execute("SELECT id FROM news_raw WHERE url = ?", (url,)).fetchone()
    return int(row["id"]) if row else None


def link_news_symbols(conn: ConnWrapper, news_id: int, links: list[tuple[int, str]]) -> None:
    if not links:
        return
    conn.executemany(
        """
        INSERT INTO news_symbol_links (news_id, symbol_id, match_reason)
        VALUES (?, ?, ?)
        ON CONFLICT (news_id, symbol_id) DO NOTHING
        """,
        [(news_id, symbol_id, reason) for symbol_id, reason in links],
    )


def count_sentiment(conn: ConnWrapper) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM news_sentiment").fetchone()["n"])


def upsert_news_sentiment(
    conn: ConnWrapper,
    news_id: int,
    score: float,
    label: str,
    positive_prob: float | None,
    negative_prob: float | None,
    neutral_prob: float | None,
    model: str,
) -> None:
    conn.execute(
        """
        INSERT INTO news_sentiment
            (news_id, score, label, positive_prob, negative_prob, neutral_prob, model, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, NOW())
        ON CONFLICT(news_id) DO UPDATE SET
            score = excluded.score,
            label = excluded.label,
            positive_prob = excluded.positive_prob,
            negative_prob = excluded.negative_prob,
            neutral_prob = excluded.neutral_prob,
            model = excluded.model,
            analyzed_at = NOW()
        """,
        (news_id, score, label, positive_prob, negative_prob, neutral_prob, model),
    )


def rebuild_sentiment_daily(conn: ConnWrapper) -> int:
    conn.execute("DELETE FROM sentiment_daily")
    cur = conn.execute(
        """
        INSERT INTO sentiment_daily (symbol_id, date, avg_score, news_count, updated_at)
        SELECT
            l.symbol_id,
            date(COALESCE(n.published_at, n.fetched_at::text)) AS d,
            AVG(s.score) AS avg_score,
            COUNT(*) AS news_count,
            NOW()
        FROM news_symbol_links l
        JOIN news_raw n ON n.id = l.news_id
        JOIN news_sentiment s ON s.news_id = n.id
        GROUP BY l.symbol_id, date(COALESCE(n.published_at, n.fetched_at::text))
        """
    )
    return cur.rowcount


def upsert_prediction(
    conn: ConnWrapper,
    symbol_id: int,
    feature_date: str,
    target_date: str | None,
    prob_up: float,
    predicted_up: int,
    model_version: str,
) -> None:
    conn.execute(
        """
        INSERT INTO predictions
            (symbol_id, feature_date, target_date, prob_up, predicted_up, model_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, NOW())
        ON CONFLICT(symbol_id, feature_date, model_version) DO UPDATE SET
            target_date = excluded.target_date,
            prob_up = excluded.prob_up,
            predicted_up = excluded.predicted_up,
            created_at = NOW()
        """,
        (symbol_id, feature_date, target_date, prob_up, predicted_up, model_version),
    )


def count_predictions(conn: ConnWrapper) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"])


def insert_backtest_run(
    conn: ConnWrapper,
    scenario: str,
    model_version: str | None,
    config_hash: str,
    start_date: str | None,
    end_date: str | None,
    starting_balance: float,
    slippage_bps: float,
    spread_bps: float,
    metrics: dict,
) -> int:
    import json as _json

    row = conn.execute(
        """
        INSERT INTO backtest_runs
            (scenario, model_version, config_hash, start_date, end_date,
             starting_balance, slippage_bps, spread_bps, metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (scenario, model_version, config_hash, start_date, end_date,
         starting_balance, slippage_bps, spread_bps, _json.dumps(metrics)),
    )
    return int(row.fetchone()["id"])


def insert_backtest_trades(conn: ConnWrapper, run_id: int, trades: list[dict]) -> None:
    if not trades:
        return
    conn.executemany(
        """
        INSERT INTO backtest_trades
            (run_id, symbol, entry_price, exit_price, quantity, gross_pnl,
             fees_paid, net_pnl, exit_reason, opened_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (run_id, t["symbol"], t["entry_price"], t["exit_price"], t["quantity"],
             t["gross_pnl"], t["fees_paid"], t["net_pnl"], t["exit_reason"],
             t["opened_at"], t["closed_at"])
            for t in trades
        ),
    )


def insert_backtest_equity(conn: ConnWrapper, run_id: int, equity_curve: list[tuple[str, float]]) -> None:
    if not equity_curve:
        return
    conn.executemany(
        "INSERT INTO backtest_equity (run_id, snapshot_date, total_value) VALUES (?, ?, ?) "
        "ON CONFLICT (run_id, snapshot_date) DO UPDATE SET total_value = excluded.total_value",
        ((run_id, d, v) for d, v in equity_curve),
    )
