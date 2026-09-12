# Sanal Alım-Satım Motoru + Canlı Dağıtım Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Borsa AI'ın mevcut (salt okunur) BIST tahmin sistemine sahte parayla otomatik alım-satım yapan bir motor eklemek, komisyon+BSMV'yi gerçekçi şekilde hesaba katmak, ve tüm sistemi (mevcut pipeline + yeni trading motoru) gerçekten canlı bir sunucuda günlük olarak çalışır hale getirmek.

**Architecture:** SQLite tamamen bırakılıp Supabase Postgres'e geçilir (Render Cron Job'ların kalıcı disk kullanamaması nedeniyle zorunlu). Yeni `trading/` paketi (saf `costs.py`, DB'ye yazan `portfolio.py`, günlük karar mantığı `engine.py`) mevcut `predictions` tablosunu okuyup yeni `positions`/`trades`/`trade_decisions` tablolarına yazar. Mevcut `daily_pipeline.py`'a yeni bir adım olarak eklenir. Render'da bir Cron Job (günlük pipeline+trading) ve bir Web Service (dashboard, genişletilmiş).

**Tech Stack:** Python, FastAPI, psycopg3, PostgreSQL (Supabase), pytest, vanilla JS (mevcut dashboard deseniyle), Render (Cron Job + Web Service).

**Spec:** `docs/superpowers/specs/2026-09-12-paper-trading-bot-design.md`

## Global Constraints

- Sadece BIST sembolleri işlem görür (`symbols.market = 'BIST'`) — ABD hisseleri trading kapsamı dışında.
- Sahte para, gerçek emir gönderimi YOK.
- Komisyon+BSMV her alım/satımda hesaba katılır; asgari işlem ücreti (`min_commission_try`) uygulanır.
- Para/miktar alanları `NUMERIC`, olasılık/fiyat piyasa verisi `REAL` kalır (mevcut şemayla tutarlı).
- Mevcut haber/sentiment/ML pipeline'ına (network/model bağımlı) yeni test YAZILMAZ — sadece yeni trading mantığı test edilir.
- Yeni bir test bağımlılığı (`pytest-postgresql` vb.) eklenmez — yerel `docker-compose.yml::db` Postgres'ine karşı, transaction-rollback deseniyle test edilir.
- Dashboard'a yeni bir JS kütüphanesi/framework eklenmez — mevcut vanilla JS + `fetchJSON`/template-string deseni ve mevcut CSS sınıfları (`.panel`, `.stat-card`, `.table-wrap`, `.sentiment-pos/-neg`) yeniden kullanılır.

---

## Task 1: `pipeline/db.py`'ı Postgres'e taşı (uyumluluk katmanıyla)

**Files:**
- Modify: `pipeline/db.py` (tam yeniden yazım)
- Modify: `requirements.txt`
- Modify: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_db_migration.py`

**Interfaces:**
- Produces: `pipeline.db.get_connection() -> ContextManager[ConnWrapper]`, `pipeline.db.ConnWrapper` (public sınıf — `.execute(sql, params) -> Cursor`, `.executemany(sql, seq) -> Cursor`, `.executescript(sql) -> None`, `.commit()`, `.rollback()`, `.close()`), `pipeline.db.init_schema(conn) -> None`, `pipeline.db.get_database_url() -> str`. Mevcut `upsert_symbol`, `upsert_prices`, `count_symbols`, `count_prices`, `count_news`, `count_news_links`, `insert_news`, `get_news_id_by_url`, `link_news_symbols`, `count_sentiment`, `upsert_news_sentiment`, `rebuild_sentiment_daily`, `upsert_prediction`, `count_predictions` imzaları AYNEN korunur (sadece iç gövdeleri Postgres uyumlu hale gelir).

- [ ] **Step 1: Bağımlılıkları ekle**

`requirements.txt`'e ekle:
```
psycopg[binary]>=3.1.0
pytest>=8.0.0
```

- [ ] **Step 2: Yerel Postgres'i ayağa kaldır**

Run: `docker compose up -d db`
Expected: `borsa-postgres` konteyneri `healthy` durumuna geçer (`docker compose ps` ile kontrol et).

- [ ] **Step 3: `.env.example`'ı güncelle**

```
DATABASE_URL=postgresql://borsa:borsa@localhost:5432/borsa

# Hugging Face model indirme (F2) — https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here
HF_HUB_DISABLE_SYMLINKS_WARNING=1
HF_HUB_DOWNLOAD_TIMEOUT=600
```

- [ ] **Step 4: `pipeline/db.py`'ı tamamen yeniden yaz**

```python
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

    def execute(self, sql: str, params: tuple = ()) -> psycopg.Cursor:
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
    raw = psycopg.connect(get_database_url())
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
```

> **Not:** `count_*` fonksiyonları eskiden `fetchone()[0]` (tuple-index) kullanıyordu — bu SQLite'a özgü satır davranışıydı. `dict_row` ile artık `fetchone()["n"]` gerekiyor, bu yüzden `SELECT COUNT(*)` sorgularına `AS n` eklendi. Bu, "diğer 15 dosyaya dokunulmuyor" kuralının TEK istisnası — çünkü bu fonksiyonların KENDİSİ zaten db.py içinde, dışarıdan sadece çağrılıyorlar, dönüş imzaları (int) değişmedi.

- [ ] **Step 5: `tests/__init__.py` oluştur (boş)**

- [ ] **Step 6: `tests/conftest.py` yaz**

```python
"""Her test gerçek yerel Postgres'e karşı çalışır, sonunda HER ZAMAN
rollback edilir (commit yok) — testler birbirini kirletmez, ekstra bir
test-DB bağımlılığı (pytest-postgresql vb.) gerekmez."""

from __future__ import annotations

import psycopg
import pytest

from pipeline.db import ConnWrapper, get_database_url, init_schema


@pytest.fixture
def conn():
    raw = psycopg.connect(get_database_url())
    wrapper = ConnWrapper(raw)
    init_schema(wrapper)
    try:
        yield wrapper
    finally:
        raw.rollback()
        raw.close()
```

- [ ] **Step 7: `tests/test_db_migration.py` yaz (başarısız olacak — henüz `psycopg` kurulu değil/DB ayakta değilse)**

```python
from pipeline.db import (
    count_symbols,
    get_news_id_by_url,
    insert_news,
    upsert_prices,
    upsert_symbol,
)


def test_init_schema_creates_all_tables(conn):
    tables = conn.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        """
    ).fetchall()
    names = {t["table_name"] for t in tables}
    expected = {
        "symbols", "prices_daily", "news_raw", "news_symbol_links",
        "news_sentiment", "sentiment_daily", "predictions",
        "portfolio", "positions", "trades", "trade_decisions",
    }
    assert expected.issubset(names)


def test_upsert_symbol_roundtrip(conn):
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY", "Türk Hava Yolları")
    assert count_symbols(conn) == 1
    same_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY", "Türk Hava Yolları (güncel)")
    assert same_id == symbol_id
    assert count_symbols(conn) == 1


def test_upsert_prices_reports_rowcount(conn):
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    changed = upsert_prices(
        conn, symbol_id,
        [("2026-09-10", 100.0, 105.0, 99.0, 104.0, 104.0, 1000.0)],
    )
    assert changed == 1


def test_insert_news_ignores_duplicate_url(conn):
    news_id = insert_news(
        conn, "test-source", None, "Başlık", "Özet",
        "https://example.com/haber-1", "2026-09-10T10:00:00", "tr",
    )
    assert news_id is not None
    duplicate_id = insert_news(
        conn, "test-source", None, "Farklı başlık", "Özet",
        "https://example.com/haber-1", "2026-09-10T10:00:00", "tr",
    )
    assert duplicate_id is None
    assert get_news_id_by_url(conn, "https://example.com/haber-1") == news_id
```

- [ ] **Step 8: Testleri çalıştır**

Run: `pytest tests/test_db_migration.py -v`
Expected: 4/4 PASS (Postgres ayakta değilse önce `docker compose up -d db` çalıştır).

- [ ] **Step 9: Commit**

```bash
git add pipeline/db.py requirements.txt .env.example tests/__init__.py tests/conftest.py tests/test_db_migration.py
git commit -m "Migrate data layer from SQLite to Postgres with a thin compat shim"
```

---

## Task 2: `trading/config.py` + `config/trading.yaml`

**Files:**
- Create: `config/trading.yaml`
- Create: `trading/__init__.py`
- Create: `trading/config.py`
- Create: `tests/test_trading_config.py`

**Interfaces:**
- Produces: `trading.config.TradingConfig` (dataclass: `starting_balance: float`, `buy_threshold: float`, `sell_threshold: float`, `max_hold_days: int`, `max_open_positions: int`, `max_position_pct: float`, `max_portfolio_exposure_pct: float`, `commission_pct: float`, `bsmv_pct_of_commission: float`, `min_commission_try: float`, `min_position_value_try: float`), `trading.config.load_trading_config(path: Path | None = None) -> TradingConfig`.

- [ ] **Step 1: `config/trading.yaml` oluştur**

```yaml
starting_balance: 100000.0

buy_threshold: 0.62
sell_threshold: 0.50
max_hold_days: 10

max_open_positions: 8
max_position_pct: 0.15
max_portfolio_exposure_pct: 0.90

commission_pct: 0.05
bsmv_pct_of_commission: 5.0
min_commission_try: 5.0
min_position_value_try: 500.0
```

- [ ] **Step 2: `trading/__init__.py` oluştur (boş)**

- [ ] **Step 3: Başarısız testi yaz — `tests/test_trading_config.py`**

```python
from pathlib import Path

from trading.config import load_trading_config

FIXTURE = Path(__file__).parent / "fixtures" / "trading_test.yaml"


def test_load_trading_config_from_project_default():
    cfg = load_trading_config()
    assert cfg.starting_balance == 100000.0
    assert cfg.buy_threshold == 0.62
    assert cfg.max_open_positions == 8


def test_load_trading_config_from_explicit_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
starting_balance: 5000.0
buy_threshold: 0.7
sell_threshold: 0.4
max_hold_days: 5
max_open_positions: 3
max_position_pct: 0.2
max_portfolio_exposure_pct: 0.8
commission_pct: 0.1
bsmv_pct_of_commission: 5.0
min_commission_try: 2.0
min_position_value_try: 100.0
""",
        encoding="utf-8",
    )
    cfg = load_trading_config(custom)
    assert cfg.starting_balance == 5000.0
    assert cfg.max_open_positions == 3
```

- [ ] **Step 4: Testin `ModuleNotFoundError` ile başarısız olduğunu doğrula**

Run: `pytest tests/test_trading_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.config'`

- [ ] **Step 5: `trading/config.py`'ı yaz**

```python
"""Trading motorunun yapılandırması — config/trading.yaml'dan yüklenir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "trading.yaml"


@dataclass(frozen=True)
class TradingConfig:
    starting_balance: float
    buy_threshold: float
    sell_threshold: float
    max_hold_days: int
    max_open_positions: int
    max_position_pct: float
    max_portfolio_exposure_pct: float
    commission_pct: float
    bsmv_pct_of_commission: float
    min_commission_try: float
    min_position_value_try: float


def load_trading_config(path: Path | None = None) -> TradingConfig:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TradingConfig(**data)
```

- [ ] **Step 6: Testleri çalıştır**

Run: `pytest tests/test_trading_config.py -v`
Expected: 2/2 PASS

- [ ] **Step 7: Commit**

```bash
git add config/trading.yaml trading/__init__.py trading/config.py tests/test_trading_config.py
git commit -m "Add trading config loader"
```

---

## Task 3: `trading/costs.py` — saf komisyon/BSMV hesaplama

**Files:**
- Create: `trading/costs.py`
- Create: `tests/test_costs.py`

**Interfaces:**
- Consumes: `trading.config.TradingConfig` (Task 2)
- Produces: `trading.costs.CommissionResult` (dataclass: `commission: float`, `bsmv: float`, `total_fee: float`), `trading.costs.calculate_fee(trade_value: float, cfg: TradingConfig) -> CommissionResult`, `trading.costs.PositionSizeCheck` (dataclass: `allowed: bool`, `reason: str`), `trading.costs.check_position_size(position_value: float, cfg: TradingConfig) -> PositionSizeCheck`.

- [ ] **Step 1: Başarısız testleri yaz — `tests/test_costs.py`**

```python
from trading.config import TradingConfig
from trading.costs import calculate_fee, check_position_size

CFG = TradingConfig(
    starting_balance=100000.0,
    buy_threshold=0.62,
    sell_threshold=0.50,
    max_hold_days=10,
    max_open_positions=8,
    max_position_pct=0.15,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=5.0,
    min_position_value_try=500.0,
)


def test_calculate_fee_normal_trade():
    result = calculate_fee(10000.0, CFG)
    # komisyon: 10000 * 0.05% = 5.0 TL; BSMV: 5.0 * %5 = 0.25 TL
    assert result.commission == 5.0
    assert result.bsmv == 0.25
    assert result.total_fee == 5.25


def test_calculate_fee_applies_minimum_commission():
    # 100 TL'lik işlemde komisyon 0.05 TL + BSMV ~0.0025 TL — asgari 5 TL'nin altında
    result = calculate_fee(100.0, CFG)
    assert result.total_fee == 5.0


def test_check_position_size_rejects_below_minimum():
    check = check_position_size(499.0, CFG)
    assert check.allowed is False
    assert "499" in check.reason


def test_check_position_size_allows_at_minimum():
    check = check_position_size(500.0, CFG)
    assert check.allowed is True
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_costs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.costs'`

- [ ] **Step 3: `trading/costs.py`'ı yaz**

```python
"""Komisyon/BSMV hesabı ve pozisyon-büyüklüğü kontrolü — SAF fonksiyonlar,
hiçbir I/O yok. `risk/cost_check.py` (kripto-trading-bot) disiplininin
aynısı: finansal mantık tek, izole, kolay test edilir bir yerde durur."""

from __future__ import annotations

from dataclasses import dataclass

from trading.config import TradingConfig


@dataclass(frozen=True)
class CommissionResult:
    commission: float
    bsmv: float
    total_fee: float


def calculate_fee(trade_value: float, cfg: TradingConfig) -> CommissionResult:
    commission = trade_value * (cfg.commission_pct / 100.0)
    bsmv = commission * (cfg.bsmv_pct_of_commission / 100.0)
    total_fee = max(commission + bsmv, cfg.min_commission_try)
    return CommissionResult(
        commission=round(commission, 4),
        bsmv=round(bsmv, 4),
        total_fee=round(total_fee, 4),
    )


@dataclass(frozen=True)
class PositionSizeCheck:
    allowed: bool
    reason: str


def check_position_size(position_value: float, cfg: TradingConfig) -> PositionSizeCheck:
    """`kripto-trading-bot`'taki 'beklenen brüt kâr > toplam maliyet'
    kontrolünün DEĞİL, bu stratejinin şekline (eşik-bazlı, stop-loss/
    take-profit yok) uyarlanmış basitleştirilmiş biçimi: pozisyon o kadar
    küçükse asgari işlem ücretinin payı orantısız büyür, böyle bir işlem
    reddedilir."""
    if position_value < cfg.min_position_value_try:
        return PositionSizeCheck(
            allowed=False,
            reason=(
                f"Pozisyon çok küçük: {position_value:.2f} TL < "
                f"asgari {cfg.min_position_value_try:.2f} TL"
            ),
        )
    return PositionSizeCheck(allowed=True, reason="Pozisyon büyüklüğü yeterli")
```

- [ ] **Step 4: Testleri çalıştır**

Run: `pytest tests/test_costs.py -v`
Expected: 4/4 PASS

- [ ] **Step 5: Commit**

```bash
git add trading/costs.py tests/test_costs.py
git commit -m "Add pure commission/BSMV and position-size check functions"
```

---

## Task 4: `trading/portfolio.py` — bakiye/pozisyon muhasebesi

**Files:**
- Create: `trading/portfolio.py`
- Create: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `pipeline.db.ConnWrapper` (Task 1), `trading.config.TradingConfig` (Task 2), `trading.costs.calculate_fee`/`check_position_size` (Task 3)
- Produces: `trading.portfolio.Position` (dataclass), `trading.portfolio.ClosedTrade` (dataclass), `trading.portfolio.PortfolioState` (dataclass: `balance: float`, `starting_balance: float`, `open_positions: list[Position]`), `trading.portfolio.ensure_portfolio(conn, starting_balance: float) -> None`, `trading.portfolio.get_state(conn) -> PortfolioState`, `trading.portfolio.get_open_position(conn, symbol_id: int) -> Position | None`, `trading.portfolio.buy(conn, symbol_id: int, price: float, prob_up: float, decision_date: date, cfg: TradingConfig) -> tuple[bool, str]`, `trading.portfolio.sell(conn, symbol_id: int, price: float, exit_reason: str, decision_date: date, cfg: TradingConfig) -> ClosedTrade | None`.

- [ ] **Step 1: Başarısız testleri yaz — `tests/test_portfolio.py`**

```python
from datetime import date

from pipeline.db import upsert_symbol
from trading.config import TradingConfig
from trading.portfolio import buy, ensure_portfolio, get_open_position, get_state, sell

CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.62,
    sell_threshold=0.50,
    max_hold_days=10,
    max_open_positions=2,
    max_position_pct=0.5,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=1.0,
    min_position_value_try=100.0,
)


def _symbol(conn) -> int:
    return upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")


def test_buy_deducts_balance_and_fee(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)

    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    assert ok is True
    state = get_state(conn)
    # pozisyon değeri: 10000 * %50 = 5000; komisyon: 5000*0.05%=2.5, BSMV: 2.5*5%=0.125, toplam 2.625
    assert state.balance == 10000.0 - 5000.0 - 2.625
    assert len(state.open_positions) == 1
    assert state.open_positions[0].symbol_id == symbol_id


def test_buy_rejects_when_balance_insufficient(conn):
    ensure_portfolio(conn, 100.0)
    symbol_id = _symbol(conn)

    ok, reason = buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    assert ok is False
    assert "yetersiz" in reason.lower()


def test_buy_rejects_when_max_open_positions_reached(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    sym1 = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    sym2 = upsert_symbol(conn, "AKBNK.IS", "BIST", "TRY")
    sym3 = upsert_symbol(conn, "GARAN.IS", "BIST", "TRY")

    assert buy(conn, sym1, 10.0, 0.7, date(2026, 9, 10), CFG)[0] is True
    assert buy(conn, sym2, 10.0, 0.7, date(2026, 9, 10), CFG)[0] is True
    ok, reason = buy(conn, sym3, 10.0, 0.7, date(2026, 9, 10), CFG)

    assert ok is False
    assert "limit" in reason.lower()


def test_sell_computes_net_pnl_after_both_fees(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)
    buy(conn, symbol_id, price=100.0, prob_up=0.7, decision_date=date(2026, 9, 10), cfg=CFG)

    trade = sell(conn, symbol_id, price=110.0, exit_reason="prob_düştü", decision_date=date(2026, 9, 11), cfg=CFG)

    assert trade is not None
    assert trade.gross_pnl == 5000.0 * 0.10  # %10 fiyat artışı, 50 adet * 10 TL
    assert trade.net_pnl < trade.gross_pnl  # iki yönlü komisyon düşülmüş olmalı
    assert get_open_position(conn, symbol_id) is None


def test_sell_returns_none_when_no_open_position(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _symbol(conn)

    assert sell(conn, symbol_id, 100.0, "prob_düştü", date(2026, 9, 10), CFG) is None
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.portfolio'`

- [ ] **Step 3: `trading/portfolio.py`'ı yaz**

```python
"""Sanal bakiye + açık/kapalı pozisyonlar — Postgres'teki `portfolio`/
`positions`/`trades` tablolarına karşı okur/yazar. `kripto-trading-bot/
engine/portfolio.py`'daki davranış sözleşmesiyle aynı (yetersiz bakiye/
limit aşımında `(False, insan-okunur sebep)` döner), ama JSON dosyası
yerine bu tablolara karşı çalışır; kilit mekanizması yok çünkü motor günde
bir kez, tek process'te sıralı çalışıyor (kripto bot'un paralel-sembol
işleme ihtiyacı burada yok)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trading.config import TradingConfig
from trading.costs import calculate_fee, check_position_size


@dataclass(frozen=True)
class Position:
    symbol_id: int
    entry_price: float
    quantity: float
    entry_prob_up: float
    opened_at: str
    entry_fee: float


@dataclass(frozen=True)
class ClosedTrade:
    symbol_id: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees_paid: float
    net_pnl: float
    exit_reason: str
    opened_at: str
    closed_at: str


@dataclass(frozen=True)
class PortfolioState:
    balance: float
    starting_balance: float
    open_positions: list[Position]


def ensure_portfolio(conn, starting_balance: float) -> None:
    """Tek satırlık portfolio kaydını yoksa oluşturur (idempotent)."""
    row = conn.execute("SELECT id FROM portfolio WHERE id = 1").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO portfolio (id, starting_balance, balance) VALUES (1, ?, ?)",
            (starting_balance, starting_balance),
        )


def _row_to_position(row) -> Position:
    return Position(
        symbol_id=int(row["symbol_id"]),
        entry_price=float(row["entry_price"]),
        quantity=float(row["quantity"]),
        entry_prob_up=float(row["entry_prob_up"]),
        opened_at=str(row["opened_at"]),
        entry_fee=float(row["entry_fee"]),
    )


def get_state(conn) -> PortfolioState:
    row = conn.execute(
        "SELECT starting_balance, balance FROM portfolio WHERE id = 1"
    ).fetchone()
    positions_rows = conn.execute(
        "SELECT symbol_id, entry_price, quantity, entry_prob_up, opened_at, entry_fee FROM positions"
    ).fetchall()
    return PortfolioState(
        balance=float(row["balance"]),
        starting_balance=float(row["starting_balance"]),
        open_positions=[_row_to_position(p) for p in positions_rows],
    )


def get_open_position(conn, symbol_id: int) -> Position | None:
    row = conn.execute(
        "SELECT symbol_id, entry_price, quantity, entry_prob_up, opened_at, entry_fee "
        "FROM positions WHERE symbol_id = ?",
        (symbol_id,),
    ).fetchone()
    return _row_to_position(row) if row else None


def buy(
    conn,
    symbol_id: int,
    price: float,
    prob_up: float,
    decision_date: date,
    cfg: TradingConfig,
) -> tuple[bool, str]:
    state = get_state(conn)

    if len(state.open_positions) >= cfg.max_open_positions:
        return False, f"Açık pozisyon limiti doldu ({cfg.max_open_positions})"

    position_value = state.balance * cfg.max_position_pct
    size_check = check_position_size(position_value, cfg)
    if not size_check.allowed:
        return False, size_check.reason

    fee = calculate_fee(position_value, cfg)
    total_cost = position_value + fee.total_fee
    if total_cost > state.balance:
        return False, (
            f"Sanal bakiye yetersiz: gereken {total_cost:.2f} TL, "
            f"mevcut {state.balance:.2f} TL"
        )

    currently_locked = sum(p.entry_price * p.quantity for p in state.open_positions)
    projected_pct = (currently_locked + position_value) / state.starting_balance * 100
    limit_pct = cfg.max_portfolio_exposure_pct * 100
    if projected_pct > limit_pct:
        return False, (
            f"Toplam portföy riski aşılırdı: %{projected_pct:.1f} "
            f"(sınır %{limit_pct:.1f})"
        )

    quantity = position_value / price
    new_balance = state.balance - total_cost

    conn.execute("UPDATE portfolio SET balance = ?, updated_at = NOW() WHERE id = 1", (new_balance,))
    conn.execute(
        "INSERT INTO positions (symbol_id, entry_price, quantity, entry_prob_up, opened_at, entry_fee) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (symbol_id, price, quantity, prob_up, decision_date.isoformat(), fee.total_fee),
    )
    return True, (
        f"Alındı: sembol {symbol_id} @ {price:.4f} x {quantity:.6f} "
        f"(ücret {fee.total_fee:.2f} TL)"
    )


def sell(
    conn,
    symbol_id: int,
    price: float,
    exit_reason: str,
    decision_date: date,
    cfg: TradingConfig,
) -> ClosedTrade | None:
    position = get_open_position(conn, symbol_id)
    if position is None:
        return None

    gross_proceeds = price * position.quantity
    fee = calculate_fee(gross_proceeds, cfg)
    net_proceeds = gross_proceeds - fee.total_fee

    entry_cost = position.entry_price * position.quantity
    gross_pnl = gross_proceeds - entry_cost
    total_fees = position.entry_fee + fee.total_fee
    net_pnl = gross_pnl - total_fees

    state = get_state(conn)
    new_balance = state.balance + net_proceeds

    conn.execute("UPDATE portfolio SET balance = ?, updated_at = NOW() WHERE id = 1", (new_balance,))
    conn.execute("DELETE FROM positions WHERE symbol_id = ?", (symbol_id,))
    conn.execute(
        """
        INSERT INTO trades
            (symbol_id, entry_price, exit_price, quantity, gross_pnl, fees_paid, net_pnl,
             exit_reason, opened_at, closed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol_id, position.entry_price, price, position.quantity,
            round(gross_pnl, 2), round(total_fees, 2), round(net_pnl, 2),
            exit_reason, position.opened_at, decision_date.isoformat(),
        ),
    )
    return ClosedTrade(
        symbol_id=symbol_id,
        entry_price=position.entry_price,
        exit_price=price,
        quantity=position.quantity,
        gross_pnl=round(gross_pnl, 2),
        fees_paid=round(total_fees, 2),
        net_pnl=round(net_pnl, 2),
        exit_reason=exit_reason,
        opened_at=position.opened_at,
        closed_at=decision_date.isoformat(),
    )
```

- [ ] **Step 4: Testleri çalıştır**

Run: `pytest tests/test_portfolio.py -v`
Expected: 5/5 PASS

- [ ] **Step 5: Commit**

```bash
git add trading/portfolio.py tests/test_portfolio.py
git commit -m "Add virtual portfolio buy/sell accounting"
```

---

## Task 5: `trading/engine.py` — günlük al/sat/tut kararı

**Files:**
- Create: `trading/engine.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: `trading.config.TradingConfig` (Task 2), `trading.portfolio` (Task 4), mevcut `predictions`/`prices_daily`/`symbols` tabloları
- Produces: `trading.engine.run_once(conn, cfg: TradingConfig, decision_date: date | None = None) -> dict` (dönen dict: `{"bought": int, "sold": int, "held": int, "rejected": int}`)

- [ ] **Step 1: Başarısız testleri yaz — `tests/test_engine.py`**

```python
from datetime import date, timedelta

from pipeline.db import upsert_prediction, upsert_prices, upsert_symbol
from trading.config import TradingConfig
from trading.engine import run_once
from trading.portfolio import ensure_portfolio, get_open_position, get_state

CFG = TradingConfig(
    starting_balance=10000.0,
    buy_threshold=0.62,
    sell_threshold=0.50,
    max_hold_days=3,
    max_open_positions=5,
    max_position_pct=0.3,
    max_portfolio_exposure_pct=0.90,
    commission_pct=0.05,
    bsmv_pct_of_commission=5.0,
    min_commission_try=1.0,
    min_position_value_try=100.0,
)


def _setup_symbol_with_price(conn, ticker: str, close: float, market: str = "BIST") -> int:
    symbol_id = upsert_symbol(conn, ticker, market, "TRY")
    upsert_prices(conn, symbol_id, [("2026-09-10", close, close, close, close, close, 1000.0)])
    return symbol_id


def test_run_once_buys_symbol_above_threshold(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 1
    assert get_open_position(conn, symbol_id) is not None


def test_run_once_ignores_symbol_below_threshold(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.40, 0, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 0
    assert get_open_position(conn, symbol_id) is None


def test_run_once_ignores_non_bist_symbol(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "AAPL", 100.0, market="US")
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.90, 1, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 0


def test_run_once_sells_when_prob_drops_below_sell_threshold(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, CFG, decision_date=date(2026, 9, 10))
    assert get_open_position(conn, symbol_id) is not None

    upsert_prices(conn, symbol_id, [("2026-09-11", 105.0, 105.0, 105.0, 105.0, 105.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.30, 0, "1.0")
    stats = run_once(conn, CFG, decision_date=date(2026, 9, 11))

    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None


def test_run_once_force_sells_after_max_hold_days(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    symbol_id = _setup_symbol_with_price(conn, "THYAO.IS", 100.0)
    upsert_prediction(conn, symbol_id, "2026-09-01", "2026-09-02", 0.75, 1, "1.0")
    run_once(conn, CFG, decision_date=date(2026, 9, 1))
    assert get_open_position(conn, symbol_id) is not None

    # prob_up hâlâ yüksek olsa bile max_hold_days (3) dolunca kapatılmalı
    late_date = date(2026, 9, 1) + timedelta(days=CFG.max_hold_days)
    upsert_prices(conn, symbol_id, [(late_date.isoformat(), 110.0, 110.0, 110.0, 110.0, 110.0, 1000.0)])
    upsert_prediction(conn, symbol_id, late_date.isoformat(), None, 0.80, 1, "1.0")
    stats = run_once(conn, CFG, decision_date=late_date)

    assert stats["sold"] == 1
    assert get_open_position(conn, symbol_id) is None


def test_run_once_rejects_when_max_positions_reached(conn):
    ensure_portfolio(conn, CFG.starting_balance)
    for i in range(5):
        symbol_id = _setup_symbol_with_price(conn, f"SYM{i}.IS", 100.0)
        upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    extra_id = _setup_symbol_with_price(conn, "EXTRA.IS", 100.0)
    upsert_prediction(conn, extra_id, "2026-09-10", "2026-09-11", 0.99, 1, "1.0")

    stats = run_once(conn, CFG, decision_date=date(2026, 9, 10))

    assert stats["bought"] == 5
    assert stats["rejected"] == 1
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.engine'`

- [ ] **Step 3: `trading/engine.py`'ı yaz**

```python
"""Günlük al/sat/tut karar motoru — sadece BIST sembolleri, eşik-bazlı.
Her karar (alım, satım, tutma, red) `trade_decisions`'a insan-okunur bir
gerekçeyle loglanır — `kripto-trading-bot`'taki 'reddedilen işlemler
nedensel loglanmalı' disiplininin aynısı."""

from __future__ import annotations

from datetime import date

from trading import portfolio as pf
from trading.config import TradingConfig


def _latest_prediction(conn, symbol_id: int):
    return conn.execute(
        "SELECT prob_up FROM predictions WHERE symbol_id = ? ORDER BY feature_date DESC LIMIT 1",
        (symbol_id,),
    ).fetchone()


def _latest_close(conn, symbol_id: int) -> float | None:
    row = conn.execute(
        "SELECT close FROM prices_daily WHERE symbol_id = ? ORDER BY date DESC LIMIT 1",
        (symbol_id,),
    ).fetchone()
    return float(row["close"]) if row else None


def _log_decision(
    conn,
    decision_date: date,
    symbol_id: int | None,
    action: str,
    reason: str,
    prob_up: float | None,
) -> None:
    conn.execute(
        "INSERT INTO trade_decisions (decision_date, symbol_id, action, reason, prob_up) "
        "VALUES (?, ?, ?, ?, ?)",
        (decision_date.isoformat(), symbol_id, action, reason, prob_up),
    )


def run_once(conn, cfg: TradingConfig, decision_date: date | None = None) -> dict:
    decision_date = decision_date or date.today()
    pf.ensure_portfolio(conn, cfg.starting_balance)

    stats = {"bought": 0, "sold": 0, "held": 0, "rejected": 0}

    # 1) Açık pozisyonları değerlendir: sat ya da tut
    state = pf.get_state(conn)
    for position in list(state.open_positions):
        pred = _latest_prediction(conn, position.symbol_id)
        current_prob = float(pred["prob_up"]) if pred else None
        opened = date.fromisoformat(position.opened_at)
        held_days = (decision_date - opened).days

        if held_days >= cfg.max_hold_days:
            price = _latest_close(conn, position.symbol_id)
            if price is not None:
                trade = pf.sell(conn, position.symbol_id, price, "max_hold_süresi", decision_date, cfg)
                if trade:
                    stats["sold"] += 1
                    _log_decision(
                        conn, decision_date, position.symbol_id, "sat",
                        "max_hold_süresi doldu", current_prob,
                    )
            continue

        if current_prob is not None and current_prob < cfg.sell_threshold:
            price = _latest_close(conn, position.symbol_id)
            if price is not None:
                trade = pf.sell(conn, position.symbol_id, price, "prob_düştü", decision_date, cfg)
                if trade:
                    stats["sold"] += 1
                    _log_decision(
                        conn, decision_date, position.symbol_id, "sat",
                        f"prob_up {current_prob:.3f} < eşik {cfg.sell_threshold}", current_prob,
                    )
                continue

        stats["held"] += 1
        _log_decision(conn, decision_date, position.symbol_id, "tut", "eşiklerin içinde", current_prob)

    # 2) Yeni alım adayları — sadece BIST, eşik üstü, henüz açık pozisyonu olmayanlar
    open_symbol_ids = {p.symbol_id for p in pf.get_state(conn).open_positions}
    candidates = conn.execute(
        """
        SELECT p.symbol_id, p.prob_up
        FROM predictions p
        JOIN symbols s ON s.id = p.symbol_id
        WHERE s.market = 'BIST'
          AND p.prob_up > ?
          AND p.feature_date = (
              SELECT MAX(p2.feature_date) FROM predictions p2 WHERE p2.symbol_id = p.symbol_id
          )
        ORDER BY p.prob_up DESC
        """,
        (cfg.buy_threshold,),
    ).fetchall()

    for row in candidates:
        symbol_id = int(row["symbol_id"])
        prob_up = float(row["prob_up"])
        if symbol_id in open_symbol_ids:
            continue

        price = _latest_close(conn, symbol_id)
        if price is None:
            stats["rejected"] += 1
            _log_decision(conn, decision_date, symbol_id, "red", "güncel fiyat yok", prob_up)
            continue

        ok, reason = pf.buy(conn, symbol_id, price, prob_up, decision_date, cfg)
        if ok:
            stats["bought"] += 1
            open_symbol_ids.add(symbol_id)
            _log_decision(conn, decision_date, symbol_id, "al", reason, prob_up)
        else:
            stats["rejected"] += 1
            _log_decision(conn, decision_date, symbol_id, "red", reason, prob_up)

    return stats
```

- [ ] **Step 4: Testleri çalıştır**

Run: `pytest tests/test_engine.py -v`
Expected: 6/6 PASS

- [ ] **Step 5: Commit**

```bash
git add trading/engine.py tests/test_engine.py
git commit -m "Add daily buy/sell/hold decision engine"
```

---

## Task 6: Pipeline entegrasyonu — `daily_pipeline.py` + `run_daily.py` + `run_trading.py`

**Files:**
- Modify: `pipeline/daily_pipeline.py`
- Modify: `scripts/run_daily.py`
- Create: `scripts/run_trading.py`
- Create: `tests/test_daily_pipeline_trading_step.py`

**Interfaces:**
- Consumes: `trading.config.load_trading_config` (Task 2), `trading.engine.run_once` (Task 5)
- Produces: `pipeline.daily_pipeline.run(..., skip_trading: bool = False)` (yeni parametre)

- [ ] **Step 1: Başarısız testi yaz — `tests/test_daily_pipeline_trading_step.py`**

```python
from unittest.mock import patch

from pipeline.daily_pipeline import run


def test_daily_pipeline_calls_trading_step_by_default():
    with (
        patch("pipeline.fetch_prices.run", return_value={}),
        patch("pipeline.fetch_news_rss.run", return_value={}),
        patch("pipeline.kap_sync.sync_kap_disclosures", return_value={}),
        patch("pipeline.entity_linker.relink_all_news", return_value={}),
        patch("pipeline.db.rebuild_sentiment_daily", return_value=0),
        patch("pipeline.analyze_sentiment.run", return_value={}),
        patch("pipeline.predict_model.run", return_value={}),
        patch("pipeline.daily_pipeline._run_trading_step", return_value={"bought": 1}) as trading_mock,
    ):
        result = run(skip_train=True)

    trading_mock.assert_called_once()
    assert result["steps"]["trading"]["ok"] is True
    assert result["steps"]["trading"]["result"] == {"bought": 1}


def test_daily_pipeline_skips_trading_when_requested():
    with (
        patch("pipeline.fetch_prices.run", return_value={}),
        patch("pipeline.fetch_news_rss.run", return_value={}),
        patch("pipeline.kap_sync.sync_kap_disclosures", return_value={}),
        patch("pipeline.entity_linker.relink_all_news", return_value={}),
        patch("pipeline.db.rebuild_sentiment_daily", return_value=0),
        patch("pipeline.analyze_sentiment.run", return_value={}),
        patch("pipeline.predict_model.run", return_value={}),
        patch("pipeline.daily_pipeline._run_trading_step") as trading_mock,
    ):
        result = run(skip_train=True, skip_trading=True)

    trading_mock.assert_not_called()
    assert "trading" not in result["steps"]
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_daily_pipeline_trading_step.py -v`
Expected: FAIL — `AttributeError` / `ImportError` (henüz `_run_trading_step` yok, `skip_trading` parametresi yok)

- [ ] **Step 3: `pipeline/daily_pipeline.py`'a trading adımını ekle**

`from pipeline.entity_linker import relink_all_news` satırından sonra, dosyanın en altına şu fonksiyonu ekle:

```python
def _run_trading_step() -> dict:
    from pipeline.db import get_connection, init_schema
    from trading.config import load_trading_config
    from trading.engine import run_once

    cfg = load_trading_config()
    with get_connection() as conn:
        init_schema(conn)
        return run_once(conn, cfg)
```

`run()` imzasını ve gövdesini güncelle (`skip_predict: bool = False,` satırından hemen sonra `skip_trading: bool = False,` ekle; `step("predict", ...)` çağrısından hemen sonra):

```python
def run(
    incremental_days: int = 7,
    skip_sentiment: bool = False,
    skip_train: bool = True,
    skip_predict: bool = False,
    skip_trading: bool = False,
    relink_news: bool = True,
    log_dir: Path | None = None,
) -> dict:
    ...  # (mevcut gövde aynı kalır, sadece imzaya skip_trading eklendi)

    if not skip_predict:
        step("predict", lambda: predict_model())

    if not skip_trading:
        step("trading", _run_trading_step)

    results["total_seconds"] = round(time.time() - started, 1)
```

- [ ] **Step 4: Testleri çalıştır**

Run: `pytest tests/test_daily_pipeline_trading_step.py -v`
Expected: 2/2 PASS

- [ ] **Step 5: `scripts/run_daily.py`'a `--skip-trading` bayrağı ve haftalık otomatik eğitim ekle**

```python
#!/usr/bin/env python3
"""Günlük pipeline — python scripts/run_daily.py"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from pipeline.daily_pipeline import run  # noqa: E402

load_project_env()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Günlük: fiyat + haber + eşleştirme + sentiment + tahmin + alım-satım"
    )
    parser.add_argument("--days", type=int, default=7, help="Fiyat: son N gün")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--train", action="store_true", help="Modeli yeniden eğit (yavaş)")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-trading", action="store_true", help="Alım-satım adımını atla")
    parser.add_argument("--no-relink", action="store_true", help="Haber eşleştirmesini atla")
    args = parser.parse_args()

    # Render Cron Job her gün aynı komutu çalıştırır — haftalık yeniden
    # eğitimi ayrı bir zamanlanmış iş yerine, aynı script içinde gün
    # kontrolüyle tetikliyoruz (tek cron job, tek yerde mantık).
    auto_train = args.train or datetime.now().weekday() == 0  # Pazartesi

    result = run(
        incremental_days=args.days,
        skip_sentiment=args.skip_sentiment,
        skip_train=not auto_train,
        skip_predict=args.skip_predict,
        skip_trading=args.skip_trading,
        relink_news=not args.no_relink,
    )
    print("\n--- Pipeline özeti ---")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: `scripts/run_trading.py` oluştur (tek başına çalıştırma)**

```python
#!/usr/bin/env python3
"""Sanal alım-satım motorunu tek başına çalıştır — python scripts/run_trading.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from pipeline.db import get_connection, init_schema  # noqa: E402
from trading.config import load_trading_config  # noqa: E402
from trading.engine import run_once  # noqa: E402

load_project_env()


def main() -> None:
    cfg = load_trading_config()
    with get_connection() as conn:
        init_schema(conn)
        result = run_once(conn, cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Elle doğrula**

Run: `python scripts/run_trading.py`
Expected: Geçerli bir JSON çıktısı (`{"bought": ..., "sold": ..., "held": ..., "rejected": ...}`) — henüz `predictions` tablosu boşsa hepsi `0` olur, hata FIRLATMAMALI.

- [ ] **Step 8: Commit**

```bash
git add pipeline/daily_pipeline.py scripts/run_daily.py scripts/run_trading.py tests/test_daily_pipeline_trading_step.py
git commit -m "Wire the trading engine into the daily pipeline"
```

---

## Task 7: API endpoint'leri — `/api/portfolio`, `/api/trades`

**Files:**
- Modify: `api/main.py`
- Create: `tests/test_api_portfolio.py`

**Interfaces:**
- Consumes: `trading.portfolio.get_state` (Task 4)
- Produces: `GET /api/portfolio`, `GET /api/trades?limit=N`

- [ ] **Step 1: Başarısız testi yaz — `tests/test_api_portfolio.py`**

```python
from datetime import date

from fastapi.testclient import TestClient

from api.main import app
from pipeline.db import upsert_prediction, upsert_prices, upsert_symbol
from trading.config import load_trading_config
from trading.engine import run_once
from trading.portfolio import ensure_portfolio

client = TestClient(app)


def test_portfolio_endpoint_reflects_open_position(conn):
    cfg = load_trading_config()
    ensure_portfolio(conn, cfg.starting_balance)
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY", "Türk Hava Yolları")
    upsert_prices(conn, symbol_id, [("2026-09-10", 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, cfg, decision_date=date(2026, 9, 10))

    response = client.get("/api/portfolio")

    assert response.status_code == 200
    data = response.json()
    assert data["open_positions"][0]["ticker"] == "THYAO.IS"
    assert data["balance"] < cfg.starting_balance


def test_trades_endpoint_reports_totals_after_a_closed_trade(conn):
    cfg = load_trading_config()
    ensure_portfolio(conn, cfg.starting_balance)
    symbol_id = upsert_symbol(conn, "THYAO.IS", "BIST", "TRY")
    upsert_prices(conn, symbol_id, [("2026-09-10", 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-10", "2026-09-11", 0.75, 1, "1.0")
    run_once(conn, cfg, decision_date=date(2026, 9, 10))

    upsert_prices(conn, symbol_id, [("2026-09-11", 90.0, 90.0, 90.0, 90.0, 90.0, 1000.0)])
    upsert_prediction(conn, symbol_id, "2026-09-11", "2026-09-12", 0.20, 0, "1.0")
    run_once(conn, cfg, decision_date=date(2026, 9, 11))

    response = client.get("/api/trades")

    assert response.status_code == 200
    data = response.json()
    assert data["totals"]["trade_count"] == 1
    assert data["totals"]["total_fees"] > 0
```

> **Not:** `api/main.py`'daki `get_connection()` her istekte kendi bağlantısını açtığı için bu test, `conn` fixture'ının açtığı AYRI transaction'dan bağımsız çalışır. Yerel `docker compose up -d db`'nin ayakta olması ve `.env`'de `DATABASE_URL`'in aynı yerel Postgres'i göstermesi yeterlidir — test kendi verisini oluşturup rollback ile temizler, API çağrısı ise gerçek (commit edilmiş) bağlantı üzerinden okur. Bu yüzden `conn` fixture'ı kullanılsa da, testin ürettiği veri görünür olması için **rollback yerine fixture'ı burada `commit` eden ayrı, ufak bir yardımcı** gerekir:

- [ ] **Step 2: `tests/conftest.py`'a commit eden bir fixture ekle**

`tests/conftest.py`'ın sonuna ekle:

```python
@pytest.fixture
def committed_conn():
    """`conn` fixture'ının aksine, API testleri gerçek (ayrı bağlantılı)
    bir istekle aynı veriyi görmeli — bu yüzden rollback yerine commit eder.
    Test sonunda ilgili tabloları TRUNCATE ederek temizler."""
    raw = psycopg.connect(get_database_url())
    wrapper = ConnWrapper(raw)
    init_schema(wrapper)
    try:
        yield wrapper
        raw.commit()
    finally:
        cur = raw.cursor()
        cur.execute(
            "TRUNCATE trade_decisions, trades, positions, portfolio, "
            "predictions, prices_daily, symbols RESTART IDENTITY CASCADE"
        )
        raw.commit()
        raw.close()
```

`tests/test_api_portfolio.py`'daki her iki testte de fixture parametresini `conn` yerine `committed_conn` olarak değiştir (fonksiyon gövdeleri aynı kalır, sadece parametre adı ve içeride kullanılan değişken adı `committed_conn` olur).

- [ ] **Step 3: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_api_portfolio.py -v`
Expected: FAIL — `404 Not Found` (endpoint'ler henüz yok)

- [ ] **Step 4: `api/main.py`'a endpoint'leri ekle**

Dosyanın başındaki importlara ekle:
```python
from trading.portfolio import get_state as get_trading_state
```

`symbol_chart` fonksiyonundan sonra, `symbol_prices` fonksiyonundan önce ekle:

```python
@app.get("/api/portfolio")
def portfolio_summary():
    with get_connection() as conn:
        init_schema(conn)
        state = get_trading_state(conn)
        positions = []
        positions_value = 0.0
        for p in state.open_positions:
            sym = conn.execute("SELECT ticker, name FROM symbols WHERE id = ?", (p.symbol_id,)).fetchone()
            last = conn.execute(
                "SELECT close FROM prices_daily WHERE symbol_id = ? ORDER BY date DESC LIMIT 1",
                (p.symbol_id,),
            ).fetchone()
            current_price = float(last["close"]) if last else p.entry_price
            market_value = current_price * p.quantity
            positions_value += market_value
            positions.append({
                "ticker": sym["ticker"] if sym else "?",
                "name": sym["name"] if sym else None,
                "entry_price": p.entry_price,
                "quantity": round(p.quantity, 6),
                "current_price": current_price,
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


@app.get("/api/trades")
def trade_history(limit: int = Query(50, ge=1, le=200)):
    with get_connection() as conn:
        init_schema(conn)
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
```

- [ ] **Step 5: Testleri çalıştır**

Run: `pytest tests/test_api_portfolio.py -v`
Expected: 2/2 PASS

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/conftest.py tests/test_api_portfolio.py
git commit -m "Add /api/portfolio and /api/trades endpoints"
```

---

## Task 8: Dashboard — "Portföy" bölümü

**Files:**
- Modify: `web/index.html`
- Modify: `web/static/app.js`

**Interfaces:**
- Consumes: `/api/portfolio`, `/api/trades` (Task 7)

- [ ] **Step 1: `web/index.html`'e Portföy bölümünü ekle**

`</main>` kapanışından sonra, mevcut `<section class="panel news-section">`'dan ÖNCE ekle:

```html
  <section class="panel" id="portfolio-section">
    <h2>Portföy <span class="hint">— sanal, günlük otomatik alım-satım</span></h2>
    <section class="stats" id="portfolio-stats"></section>

    <h3>Açık pozisyonlar</h3>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Hisse</th><th>Giriş</th><th>Güncel</th><th>Adet</th>
            <th>Gerçekleşmemiş K/Z</th><th>Açılış</th>
          </tr>
        </thead>
        <tbody id="positions-body"></tbody>
      </table>
    </div>

    <h3>Kapanan işlemler</h3>
    <p class="hint" id="trades-summary"></p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Hisse</th><th>Giriş</th><th>Çıkış</th><th>Brüt K/Z</th>
            <th>Komisyon</th><th>Net K/Z</th><th>Sebep</th><th>Kapanış</th>
          </tr>
        </thead>
        <tbody id="trades-body"></tbody>
      </table>
    </div>
  </section>
```

- [ ] **Step 2: `web/static/app.js`'e `loadPortfolio`/`loadTrades` ekle**

`loadNewsFeed` fonksiyonundan sonra, `document.querySelectorAll(".filter")` satırından ÖNCE ekle:

```javascript
async function loadPortfolio() {
  const data = await fetchJSON("/api/portfolio");
  const el = document.getElementById("portfolio-stats");
  const cards = [
    ["Nakit", `${data.balance.toLocaleString("tr")} TL`],
    ["Pozisyon değeri", `${data.positions_value.toLocaleString("tr")} TL`],
    ["Toplam varlık", `${data.total_value.toLocaleString("tr")} TL`],
    ["Başlangıç", `${data.starting_balance.toLocaleString("tr")} TL`],
  ];
  el.innerHTML = cards
    .map(
      ([label, value]) => `
    <div class="stat-card">
      <div class="value">${value}</div>
      <div class="label">${label}</div>
    </div>`
    )
    .join("");

  const tbody = document.getElementById("positions-body");
  tbody.innerHTML =
    data.open_positions
      .map((p) => {
        const pnlClass = p.unrealized_pnl >= 0 ? "sentiment-pos" : "sentiment-neg";
        return `
      <tr>
        <td>${p.ticker}</td>
        <td>${p.entry_price.toFixed(2)}</td>
        <td>${p.current_price.toFixed(2)}</td>
        <td>${p.quantity.toFixed(4)}</td>
        <td class="${pnlClass}">${p.unrealized_pnl.toFixed(2)} TL</td>
        <td>${p.opened_at}</td>
      </tr>`;
      })
      .join("") || `<tr><td colspan="6">Açık pozisyon yok.</td></tr>`;
}

async function loadTrades() {
  const data = await fetchJSON("/api/trades?limit=50");
  const tbody = document.getElementById("trades-body");
  tbody.innerHTML =
    data.items
      .map((t) => {
        const pnlClass = t.net_pnl >= 0 ? "sentiment-pos" : "sentiment-neg";
        return `
      <tr>
        <td>${t.ticker}</td>
        <td>${Number(t.entry_price).toFixed(2)}</td>
        <td>${Number(t.exit_price).toFixed(2)}</td>
        <td>${Number(t.gross_pnl).toFixed(2)}</td>
        <td>${Number(t.fees_paid).toFixed(2)}</td>
        <td class="${pnlClass}">${Number(t.net_pnl).toFixed(2)}</td>
        <td>${t.exit_reason}</td>
        <td>${t.closed_at}</td>
      </tr>`;
      })
      .join("") || `<tr><td colspan="8">Henüz kapanan işlem yok.</td></tr>`;

  const summary = document.getElementById("trades-summary");
  if (data.totals.trade_count) {
    summary.textContent =
      `Toplam ${data.totals.trade_count} işlem · ` +
      `brüt ${data.totals.total_gross_pnl.toFixed(2)} TL · ` +
      `komisyon ${data.totals.total_fees.toFixed(2)} TL · ` +
      `net ${data.totals.total_net_pnl.toFixed(2)} TL · ` +
      `kazanma oranı %${data.totals.win_rate ?? "—"}`;
  } else {
    summary.textContent = "";
  }
}
```

`init()` fonksiyonunu güncelle:

```javascript
async function init() {
  try {
    await loadStats();
    await loadPredictions();
    await loadNewsFeed();
    await loadPortfolio();
    await loadTrades();
  } catch (e) {
    console.error(e);
    document.body.insertAdjacentHTML(
      "beforeend",
      `<p style="color:#ef4444;padding:2rem">API baglantisi kurulamadi. Sunucuyu baslatin: python scripts/run_server.py</p>`
    );
  }
}
```

- [ ] **Step 3: Elle doğrula**

Run: `python scripts/run_server.py`, tarayıcıda `http://127.0.0.1:8000` aç.
Expected: Sayfanın altında "Portföy" başlıklı yeni bir bölüm görünür (bakiye kartları + iki tablo), konsol hatası yok. Henüz hiç trading verisi yoksa "Açık pozisyon yok." / "Henüz kapanan işlem yok." mesajları görünmeli.

- [ ] **Step 4: Commit**

```bash
git add web/index.html web/static/app.js
git commit -m "Add Portfolio section to the dashboard"
```

---

## Task 9: Supabase projesi oluştur + şemayı uygula

**Files:** (kod değişikliği yok — altyapı kurulumu, MCP araçlarıyla)

- [ ] **Step 1: Supabase organizasyonunu listele**

Tool: `mcp__plugin_supabase_supabase__list_organizations` (veya `mcp__supabase__*` eşleniği) — çıktıdan bir `organization_id` seç.

- [ ] **Step 2: Yeni proje oluştur**

Tool: `mcp__plugin_supabase_supabase__create_project` — `name: "borsa-ai"`, `organization_id`, uygun bir `region` (ör. `eu-central-1`, Türkiye'ye en yakın). Proje `INACTIVE`/kuruluyor durumunda dönebilir; hazır olana kadar `get_project` ile durumu kontrol et.

- [ ] **Step 3: Şemayı uygula**

Tool: `mcp__plugin_supabase_supabase__apply_migration` — `pipeline/db.py::SCHEMA_SQL` içeriğinin AYNISINI (Task 1'de yazılan, Postgres-uyumlu hâli) bir migration olarak gönder (`name: "initial_schema"`).

- [ ] **Step 4: Bağlantı bilgilerini al**

Tool: `mcp__plugin_supabase_supabase__get_project_url` ve `get_publishable_keys` — ama asıl gereken doğrudan Postgres bağlantı dizesi (connection string), Supabase dashboard'ında Project Settings → Database → Connection string (URI) altında. Bunu `DATABASE_URL` olarak not al (bir sonraki task'ta Render'a girilecek, koda YAZILMAYACAK).

- [ ] **Step 5: Doğrula**

Yerel makineden, `.env`'i geçici olarak Supabase `DATABASE_URL`'ine çevirip:
Run: `python scripts/run_trading.py`
Expected: Hatasız çalışır, Supabase'teki `portfolio` tablosunda 1 satır oluşur (Supabase dashboard → Table Editor'dan kontrol edilebilir).
Sonra `.env`'i tekrar yerel `postgresql://borsa:borsa@localhost:5432/borsa`'ya döndür.

---

## Task 10: `render.yaml` + README güncellemesi

**Files:**
- Create: `render.yaml`
- Modify: `README.md`

- [ ] **Step 1: `render.yaml` oluştur**

```yaml
# Render Blueprint — https://render.com/docs/blueprint-spec
# Kullanım: Render Dashboard → New → Blueprint → bu repo'yu seç.
# sync: false olan değişkenlerin DEĞERLERİ dashboard'dan elle girilir.
services:
  - type: web
    name: borsa-ai-dashboard
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: DATABASE_URL
        sync: false
      - key: HF_TOKEN
        sync: false

  - type: cron
    name: borsa-ai-daily
    runtime: python
    plan: starter
    schedule: "0 16 * * 1-5"
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/run_daily.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: DATABASE_URL
        sync: false
      - key: HF_TOKEN
        sync: false
```

- [ ] **Step 2: `README.md`'ye "Canlı Dağıtım" ve "Sanal Alım-Satım" bölümlerini ekle**

`## Sonraki fazlar` bölümünden HEMEN ÖNCE ekle:

```markdown
## Sanal alım-satım (F5)

Mevcut tahminleri (`prob_up`) kullanarak SADECE BIST hisselerinde, sahte
100.000 TL ile eşik-bazlı otomatik alım-satım yapar — gerçek emir
göndermez. Her işlemde komisyon + BSMV gerçekçi şekilde hesaba katılır.
Parametreler: `config/trading.yaml`.

```bash
python scripts/run_trading.py
```

Dashboard'daki "Portföy" bölümü güncel bakiyeyi, açık pozisyonları ve
kapanan işlemleri (brüt/net kâr, ödenen komisyon ayrı ayrı) gösterir.

**Uyarı:** Tamamen simülasyondur; yatırım tavsiyesi değildir.

## Canlı dağıtım

1. Bir Supabase projesi oluştur, `pipeline/db.py::SCHEMA_SQL`'i uygula.
2. Render'da bu repoyu Blueprint (`render.yaml`) ile bağla.
3. Her iki serviste de (`borsa-ai-dashboard`, `borsa-ai-daily`)
   `DATABASE_URL` (Supabase connection string) ve `HF_TOKEN`'ı elle gir.
4. `borsa-ai-daily` Cron Job'ı hafta içi her gün BIST kapanışından sonra
   (19:00 İstanbul = 16:00 UTC) otomatik çalışır: fiyat → haber → eşleştirme
   → sentiment → tahmin → alım-satım. Pazartesi günleri model de otomatik
   yeniden eğitilir.
```

`## Kurulum` bölümündeki SQLite referansını güncelle: `Veritabanı varsayılan: data/borsa.db (SQLite).` satırını şu şekilde değiştir:

```markdown
Veritabanı: PostgreSQL. Yerelde `docker compose up -d db`, canlıda Supabase.
`.env`'de `DATABASE_URL` (bkz. `.env.example`).
```

- [ ] **Step 3: Commit**

```bash
git add render.yaml README.md
git commit -m "Add Render blueprint and deployment docs"
```

---

## Task 11: Uçtan uca canlı doğrulama

**Files:** (kod değişikliği yok)

- [ ] **Step 1: Render Blueprint'i uygula**

Render Dashboard → New → Blueprint → `UmutErayAltay/BorsaSite` reposu, `feature/borsa-ai-platform` dalı → `render.yaml` algılanır → Apply.

- [ ] **Step 2: Ortam değişkenlerini gir**

Her iki serviste de `DATABASE_URL` (Task 9'daki Supabase connection string) ve `HF_TOKEN` değerlerini Render dashboard'undan elle gir.

- [ ] **Step 3: Cron Job'ı elle bir kez tetikle**

Render Dashboard → `borsa-ai-daily` → "Trigger Run" (ilk veriyi doldurmak için otomatik zamanlamayı beklemeden).
Expected: Log'larda `=== trading ===` adımı hatasız tamamlanır.

- [ ] **Step 4: Dashboard'ı canlıda doğrula**

`borsa-ai-dashboard` servisinin verdiği URL'i tarayıcıda aç.
Expected: "Portföy" bölümü Supabase'teki gerçek veriyle dolu görünür (en az bakiye kartları; ilk çalıştırmada `predictions` tablosu henüz boşsa "Açık pozisyon yok." normal).

- [ ] **Step 5: Vault'a not düş**

`🔮 850-Companion/Threads.md`'deki "Freelance + tam zamanlı iş hazırlığı" thread'ine bu işin tamamlandığını, canlı URL'i ve Cron Job zamanlamasını ekle.
