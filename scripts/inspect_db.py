#!/usr/bin/env python3
"""Veritabanı özetini gösterir."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.db import get_connection, init_schema  # noqa: E402


def main() -> None:
    with get_connection() as conn:
        init_schema(conn)
        symbols = conn.execute(
            "SELECT market, COUNT(*) AS n FROM symbols GROUP BY market"
        ).fetchall()
        prices = conn.execute(
            """
            SELECT s.ticker, COUNT(p.id) AS bars, MAX(p.date) AS last_date
            FROM symbols s
            LEFT JOIN prices_daily p ON p.symbol_id = s.id
            GROUP BY s.id
            ORDER BY s.market, s.ticker
            LIMIT 10
            """
        ).fetchall()

        print("Sembol sayısı (piyasa):")
        for row in symbols:
            print(f"  {row['market']}: {row['n']}")

        total = conn.execute("SELECT COUNT(*) AS n FROM prices_daily").fetchone()["n"]
        print(f"\nToplam fiyat satırı: {total}")
        print("\nÖrnek (ilk 10):")
        for row in prices:
            print(f"  {row['ticker']}: {row['bars']} bar, son={row['last_date']}")


if __name__ == "__main__":
    main()
