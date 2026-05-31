"""Veritabanı bağlantısı ve şema (SQLite varsayılan, PostgreSQL opsiyonel)."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "borsa.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    market TEXT NOT NULL,
    currency TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prices_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    adj_close REAL,
    volume REAL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
    UNIQUE(symbol_id, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices_daily(symbol_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_symbols_market ON symbols(market);

CREATE TABLE IF NOT EXISTS news_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    feed_id TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT NOT NULL UNIQUE,
    published_at TEXT,
    language TEXT NOT NULL DEFAULT 'tr',
    content TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS news_symbol_links (
    news_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    match_reason TEXT,
    PRIMARY KEY (news_id, symbol_id),
    FOREIGN KEY (news_id) REFERENCES news_raw(id) ON DELETE CASCADE,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_raw(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_source ON news_raw(source);
CREATE INDEX IF NOT EXISTS idx_news_links_symbol ON news_symbol_links(symbol_id);

CREATE TABLE IF NOT EXISTS news_sentiment (
    news_id INTEGER PRIMARY KEY,
    score REAL NOT NULL,
    label TEXT NOT NULL,
    positive_prob REAL,
    negative_prob REAL,
    neutral_prob REAL,
    model TEXT NOT NULL,
    analyzed_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (news_id) REFERENCES news_raw(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sentiment_daily (
    symbol_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    avg_score REAL NOT NULL,
    news_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (symbol_id, date),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sentiment_daily_date ON sentiment_daily(date DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    feature_date TEXT NOT NULL,
    target_date TEXT,
    prob_up REAL NOT NULL,
    predicted_up INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
    UNIQUE(symbol_id, feature_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol_id, feature_date DESC);
"""


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")


def is_sqlite() -> bool:
    return get_database_url().startswith("sqlite")


def get_sqlite_path() -> Path:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1)
        return Path(path)
    return DEFAULT_SQLITE_PATH


def ensure_data_dir() -> None:
    get_sqlite_path().parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """SQLite bağlantısı (F0). PostgreSQL F1+ için genişletilebilir."""
    if not is_sqlite():
        raise NotImplementedError(
            "PostgreSQL henüz bağlı değil. DATABASE_URL boş bırakın veya sqlite kullanın."
        )
    ensure_data_dir()
    conn = sqlite3.connect(get_sqlite_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def upsert_symbol(
    conn: sqlite3.Connection,
    ticker: str,
    market: str,
    currency: str,
    name: str | None = None,
    sector: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO symbols (ticker, market, currency, name, sector, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(ticker) DO UPDATE SET
            market = excluded.market,
            currency = excluded.currency,
            name = COALESCE(excluded.name, symbols.name),
            sector = COALESCE(excluded.sector, symbols.sector),
            updated_at = datetime('now')
        """,
        (ticker, market, currency, name, sector),
    )
    row = conn.execute("SELECT id FROM symbols WHERE ticker = ?", (ticker,)).fetchone()
    return int(row["id"])


def upsert_prices(
    conn: sqlite3.Connection,
    symbol_id: int,
    rows: Iterator[tuple],
) -> int:
    """rows: (date, open, high, low, close, adj_close, volume)"""
    conn.executemany(
        """
        INSERT INTO prices_daily (symbol_id, date, open, high, low, close, adj_close, volume, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(symbol_id, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            adj_close = excluded.adj_close,
            volume = excluded.volume,
            fetched_at = datetime('now')
        """,
        ((symbol_id, *r) for r in rows),
    )
    return conn.total_changes


def count_symbols(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])


def count_prices(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM prices_daily").fetchone()[0])


def count_news(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0])


def count_news_links(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM news_symbol_links").fetchone()[0])


def insert_news(
    conn: sqlite3.Connection,
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
        INSERT OR IGNORE INTO news_raw
            (source, feed_id, title, summary, url, published_at, language, content, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (source, feed_id, title, summary, url, published_at, language, content),
    )
    if cur.rowcount == 0:
        return None
    return int(cur.lastrowid)


def get_news_id_by_url(conn: sqlite3.Connection, url: str) -> int | None:
    row = conn.execute("SELECT id FROM news_raw WHERE url = ?", (url,)).fetchone()
    return int(row["id"]) if row else None


def link_news_symbols(
    conn: sqlite3.Connection,
    news_id: int,
    links: list[tuple[int, str]],
) -> None:
    if not links:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO news_symbol_links (news_id, symbol_id, match_reason)
        VALUES (?, ?, ?)
        """,
        [(news_id, symbol_id, reason) for symbol_id, reason in links],
    )


def count_sentiment(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0])


def upsert_news_sentiment(
    conn: sqlite3.Connection,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(news_id) DO UPDATE SET
            score = excluded.score,
            label = excluded.label,
            positive_prob = excluded.positive_prob,
            negative_prob = excluded.negative_prob,
            neutral_prob = excluded.neutral_prob,
            model = excluded.model,
            analyzed_at = datetime('now')
        """,
        (news_id, score, label, positive_prob, negative_prob, neutral_prob, model),
    )


def rebuild_sentiment_daily(conn: sqlite3.Connection) -> int:
    conn.execute("DELETE FROM sentiment_daily")
    cur = conn.execute(
        """
        INSERT INTO sentiment_daily (symbol_id, date, avg_score, news_count, updated_at)
        SELECT
            l.symbol_id,
            date(COALESCE(n.published_at, n.fetched_at)) AS d,
            AVG(s.score) AS avg_score,
            COUNT(*) AS news_count,
            datetime('now')
        FROM news_symbol_links l
        JOIN news_raw n ON n.id = l.news_id
        JOIN news_sentiment s ON s.news_id = n.id
        GROUP BY l.symbol_id, date(COALESCE(n.published_at, n.fetched_at))
        """
    )
    return cur.rowcount


def upsert_prediction(
    conn: sqlite3.Connection,
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
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(symbol_id, feature_date, model_version) DO UPDATE SET
            target_date = excluded.target_date,
            prob_up = excluded.prob_up,
            predicted_up = excluded.predicted_up,
            created_at = datetime('now')
        """,
        (symbol_id, feature_date, target_date, prob_up, predicted_up, model_version),
    )


def count_predictions(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
