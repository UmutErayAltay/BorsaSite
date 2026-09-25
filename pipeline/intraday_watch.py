"""Açık pozisyonlar için gün içi haber izleme — YENİ ALIM yapmaz, sadece
mevcut bir pozisyonda güçlü olumsuz bir KAP bildirimi gelirse erken satar
(risk azaltma). Günlük tahmin/alım döngüsüne (pipeline/daily_pipeline.py)
paralel, gün içinde birkaç kez çalıştırılmak üzere tasarlandı (bkz.
scripts/run_intraday_watch.py).

Bilinçli sınır: proje sadece günlük bar kullanıyor, gün içi fiyat akışı yok
— erken çıkış en son bilinen kapanış fiyatından yapılmış gibi işaretlenir
(`trading/portfolio.py::current_price`'ın kendisi de aynı yaklaşımı
kullanıyor). Gerçek gün içi fiyatla yürütme ayrı bir veri kaynağı gerektirir.
"""
from __future__ import annotations

import logging
from datetime import date

from pipeline.db import upsert_news_sentiment
from pipeline.kap_sync import sync_kap_disclosures
from pipeline.sentiment_models import analyze_batch
from trading.config import TradingConfig, load_trading_config
from trading.portfolio import current_price, get_state, sell

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVE_THRESHOLD = -0.5


def _pending_news_for_symbols(conn, symbol_ids: set[int]) -> list[dict]:
    """`symbol_ids`'den en az biriyle eşleşen, henüz sentiment skoru
    olmayan haberler."""
    if not symbol_ids:
        return []
    rows = conn.execute(
        """
        SELECT n.id AS news_id, n.title, n.summary, n.language, l.symbol_id
        FROM news_raw n
        JOIN news_symbol_links l ON l.news_id = n.id
        LEFT JOIN news_sentiment s ON s.news_id = n.id
        WHERE s.news_id IS NULL
        """
    ).fetchall()
    return [dict(r) for r in rows if int(r["symbol_id"]) in symbol_ids]


def _score_pending_news(conn, pending: list[dict]) -> int:
    """Bekleyen haberleri dil bazında toplu skorlar, `news_sentiment`'e yazar.
    Aynı haber birden fazla sembole bağlıysa (N:M) tekilleştirip bir kez skorlar."""
    by_id = {row["news_id"]: row for row in pending}
    unique = list(by_id.values())
    if not unique:
        return 0

    texts = [((r["title"] or "") + "\n" + (r["summary"] or "")).strip() for r in unique]
    langs = [(r["language"] or "tr") for r in unique]

    results: list[tuple | None] = [None] * len(unique)
    for lang_group in ("tr", "en"):
        idx = [i for i, l in enumerate(langs) if (l.lower()[:2] == "en") == (lang_group == "en")]
        if not idx:
            continue
        scored = analyze_batch([texts[i] for i in idx], lang_group)
        for j, i in enumerate(idx):
            results[i] = scored[j]

    for row, result in zip(unique, results):
        if result is None:
            continue
        score, label, pos, neg, neu, model = result
        upsert_news_sentiment(conn, int(row["news_id"]), score, label, pos, neg, neu, model)
    return len(unique)


def _worst_score_since(conn, symbol_id: int, since_date: str) -> float | None:
    row = conn.execute(
        """
        SELECT MIN(s.score) AS worst
        FROM news_symbol_links l
        JOIN news_sentiment s ON s.news_id = l.news_id
        JOIN news_raw n ON n.id = l.news_id
        WHERE l.symbol_id = ? AND (n.published_at IS NULL OR n.published_at >= ?)
        """,
        (symbol_id, since_date),
    ).fetchone()
    return float(row["worst"]) if row and row["worst"] is not None else None


def react_to_negative_news(
    conn,
    held_symbol_ids: set[int],
    trading_cfg: TradingConfig,
    negative_threshold: float,
    decision_date: date | None = None,
) -> dict:
    """Ağ çağrısı yapmaz — `news_raw`/`news_symbol_links` üzerinden çalışır,
    bu yüzden gerçek DB'ye karşı mock'suz test edilebilir (bkz.
    tests/test_intraday_watch.py). `sync_kap_disclosures` çağrısı bilerek
    burada değil, `run_intraday_watch`'ta — ağ bağımlı kısım ayrı tutulur."""
    decision_date = decision_date or date.today()
    stats = {"analyzed": 0, "sold": 0, "sold_symbols": []}

    pending = _pending_news_for_symbols(conn, held_symbol_ids)
    stats["analyzed"] = _score_pending_news(conn, pending)

    state = get_state(conn)
    for position in state.open_positions:
        if position.symbol_id not in held_symbol_ids:
            continue
        worst = _worst_score_since(conn, position.symbol_id, position.opened_at)
        if worst is None or worst >= negative_threshold:
            continue

        price = current_price(conn, position.symbol_id, position.entry_price)
        trade = sell(
            conn,
            position.symbol_id,
            price=price,
            exit_reason=f"gun_ici_olumsuz_haber(skor={worst:.2f})",
            decision_date=decision_date,
            cfg=trading_cfg,
        )
        if trade:
            stats["sold"] += 1
            stats["sold_symbols"].append(position.symbol_id)
            logger.info(
                "Gün içi erken çıkış: symbol_id=%s skor=%.2f net_pnl=%.2f",
                position.symbol_id, worst, trade.net_pnl,
            )
    return stats


def run_intraday_watch(
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    trading_cfg: TradingConfig | None = None,
) -> dict:
    from pipeline.db import get_connection, init_schema

    trading_cfg = trading_cfg or load_trading_config()
    stats = {
        "checked_positions": 0, "kap_fetched": 0, "kap_new_news": 0,
        "analyzed": 0, "sold": 0, "sold_symbols": [],
    }

    # `sync_kap_disclosures()` kendi bağlantısını açıp kapatıyor — burada AÇIK
    # bir bağlantının İÇİNDEN çağrılırsa, dıştaki bağlantının commit edilmemiş
    # `init_schema()` kilitleriyle kilitlenip sonsuza kadar bekler (gerçek bir
    # deadlock ile tespit edildi). Bu yüzden iki adım kesinlikle AYRI
    # bağlantılarda, art arda çalıştırılır.
    with get_connection() as conn:
        init_schema(conn)
        state = get_state(conn)
        if not state.open_positions:
            return stats
        held_symbol_ids = {p.symbol_id for p in state.open_positions}
        stats["checked_positions"] = len(held_symbol_ids)

    kap_stats = sync_kap_disclosures(days=1, enrich_existing_urls=False)
    stats["kap_fetched"] = kap_stats["kap_fetched"]
    stats["kap_new_news"] = kap_stats["new_news"]

    with get_connection() as conn:
        reaction = react_to_negative_news(conn, held_symbol_ids, trading_cfg, negative_threshold)
        stats["analyzed"] = reaction["analyzed"]
        stats["sold"] = reaction["sold"]
        stats["sold_symbols"] = reaction["sold_symbols"]

    return stats
