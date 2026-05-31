#!/usr/bin/env python3
"""Haber veritabanı özeti."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.db import get_connection, init_schema  # noqa: E402


def main() -> None:
    with get_connection() as conn:
        init_schema(conn)

        total = conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
        links = conn.execute("SELECT COUNT(*) FROM news_symbol_links").fetchone()[0]
        print(f"Toplam haber: {total}")
        print(f"Hisse bağlantısı: {links}")

        print("\nKaynak bazında:")
        for row in conn.execute(
            """
            SELECT source, COUNT(*) AS n
            FROM news_raw GROUP BY source ORDER BY n DESC
            """
        ):
            print(f"  {row['source']}: {row['n']}")

        print("\nEn çok haberle eşleşen hisseler (ilk 10):")
        for row in conn.execute(
            """
            SELECT s.ticker, COUNT(*) AS n
            FROM news_symbol_links l
            JOIN symbols s ON s.id = l.symbol_id
            GROUP BY s.id ORDER BY n DESC LIMIT 10
            """
        ):
            print(f"  {row['ticker']}: {row['n']}")

        print("\nSon haberler (5):")
        for row in conn.execute(
            """
            SELECT n.title, n.source, n.published_at,
                   GROUP_CONCAT(s.ticker) AS symbols
            FROM news_raw n
            LEFT JOIN news_symbol_links l ON l.news_id = n.id
            LEFT JOIN symbols s ON s.id = l.symbol_id
            GROUP BY n.id
            ORDER BY n.published_at DESC NULLS LAST, n.id DESC
            LIMIT 5
            """
        ):
            sym = row["symbols"] or "—"
            print(f"  [{row['source']}] {row['title'][:70]}...")
            print(f"    {row['published_at']} | {sym}")


if __name__ == "__main__":
    main()
