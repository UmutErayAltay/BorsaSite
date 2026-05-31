#!/usr/bin/env python3
"""Duygu analizi özeti."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.db import get_connection, init_schema  # noqa: E402


def main() -> None:
    with get_connection() as conn:
        init_schema(conn)

        total = conn.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0]
        pending = conn.execute(
            """
            SELECT COUNT(*) FROM news_raw n
            LEFT JOIN news_sentiment s ON s.news_id = n.id
            WHERE s.news_id IS NULL
            """
        ).fetchone()[0]
        daily = conn.execute("SELECT COUNT(*) FROM sentiment_daily").fetchone()[0]

        print(f"Analiz edilmiş haber: {total}")
        print(f"Bekleyen haber: {pending}")
        print(f"Günlük hisse özeti satırı: {daily}")

        print("\nSkor dağılımı:")
        for row in conn.execute(
            """
            SELECT
                CASE
                    WHEN score >= 0.3 THEN 'pozitif (>=0.3)'
                    WHEN score <= -0.3 THEN 'negatif (<=-0.3)'
                    ELSE 'nötr'
                END AS bucket,
                COUNT(*) AS n
            FROM news_sentiment
            GROUP BY bucket
            """
        ):
            print(f"  {row['bucket']}: {row['n']}")

        print("\nÖrnek haberler (skor | başlık):")
        for row in conn.execute(
            """
            SELECT n.title, s.score, s.label, s.model
            FROM news_sentiment s
            JOIN news_raw n ON n.id = s.news_id
            ORDER BY ABS(s.score) DESC
            LIMIT 8
            """
        ):
            print(f"  {row['score']:+.2f} [{row['label']}] {row['title'][:65]}")

        print("\nHisse günlük sentiment (son 5):")
        for row in conn.execute(
            """
            SELECT sym.ticker, d.date, d.avg_score, d.news_count
            FROM sentiment_daily d
            JOIN symbols sym ON sym.id = d.symbol_id
            ORDER BY d.date DESC, d.avg_score DESC
            LIMIT 5
            """
        ):
            print(
                f"  {row['ticker']} {row['date']}: "
                f"avg={row['avg_score']:+.3f} ({row['news_count']} haber)"
            )


if __name__ == "__main__":
    main()
