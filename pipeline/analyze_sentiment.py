"""Haberler için duygu analizi ve günlük hisse özeti."""

from __future__ import annotations

import logging
from collections import defaultdict

from pipeline.db import (
    count_sentiment,
    get_connection,
    init_schema,
    rebuild_sentiment_daily,
    upsert_news_sentiment,
)
from pipeline.sentiment_models import analyze_batch

logger = logging.getLogger(__name__)


def _news_text(title: str, summary: str | None) -> str:
    parts = [title.strip()]
    if summary and summary.strip():
        parts.append(summary.strip())
    return "\n".join(parts)


def fetch_pending_news(conn, limit: int | None, force: bool) -> list[dict]:
    sql = """
        SELECT n.id, n.title, n.summary, n.language
        FROM news_raw n
    """
    if not force:
        sql += """
        LEFT JOIN news_sentiment s ON s.news_id = n.id
        WHERE s.news_id IS NULL
        """
    sql += " ORDER BY n.published_at DESC, n.id DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def run(
    limit: int | None = None,
    force: bool = False,
    skip_aggregate: bool = False,
) -> dict[str, int | float]:
    stats: dict[str, int | float] = {
        "analyzed": 0,
        "errors": 0,
        "daily_rows": 0,
    }

    with get_connection() as conn:
        init_schema(conn)
        pending = fetch_pending_news(conn, limit, force)
        if not pending:
            logger.info("Analiz edilecek haber yok.")
            stats["total_sentiment_db"] = count_sentiment(conn)
            if not skip_aggregate:
                stats["daily_rows"] = rebuild_sentiment_daily(conn)
            return stats

        by_lang: dict[str, list[dict]] = defaultdict(list)
        for row in pending:
            lang = (row.get("language") or "tr").lower()[:2]
            by_lang[lang].append(row)

        for lang, items in by_lang.items():
            logger.info("%s dili: %d haber", lang, len(items))
            texts = [_news_text(r["title"], r["summary"]) for r in items]
            try:
                scores = analyze_batch(texts, lang)
            except Exception as e:
                logger.exception("Batch hata (%s): %s", lang, e)
                stats["errors"] += len(items)
                continue

            for row, (score, label, pos, neg, neu, model) in zip(items, scores):
                try:
                    upsert_news_sentiment(
                        conn,
                        news_id=int(row["id"]),
                        score=score,
                        label=label,
                        positive_prob=pos,
                        negative_prob=neg,
                        neutral_prob=neu,
                        model=model,
                    )
                    stats["analyzed"] += 1
                except Exception as e:
                    logger.exception("Kayıt hatası news_id=%s: %s", row["id"], e)
                    stats["errors"] += 1

        if not skip_aggregate:
            stats["daily_rows"] = rebuild_sentiment_daily(conn)

        stats["total_sentiment_db"] = count_sentiment(conn)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run(limit=20))
