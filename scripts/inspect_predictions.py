#!/usr/bin/env python3
"""Tahmin özeti — yükselme olasılığına göre sıralı."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.db import get_connection, init_schema  # noqa: E402


def main() -> None:
    with get_connection() as conn:
        init_schema(conn)
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        print(f"Toplam tahmin: {total}\n")

        print("En yüksek yükselme olasılığı (10):")
        for row in conn.execute(
            """
            SELECT s.ticker, p.feature_date, p.prob_up, p.predicted_up
            FROM predictions p
            JOIN symbols s ON s.id = p.symbol_id
            ORDER BY p.prob_up DESC
            LIMIT 10
            """
        ):
            direction = "YUKARI" if row["predicted_up"] else "ASAGI"
            print(
                f"  {row['ticker']}: %{row['prob_up']*100:.1f} {direction} "
                f"(veri: {row['feature_date']})"
            )

        print("\nEn düşük yükselme olasılığı (5):")
        for row in conn.execute(
            """
            SELECT s.ticker, p.feature_date, p.prob_up, p.predicted_up
            FROM predictions p
            JOIN symbols s ON s.id = p.symbol_id
            ORDER BY p.prob_up ASC
            LIMIT 5
            """
        ):
            direction = "YUKARI" if row["predicted_up"] else "ASAGI"
            print(
                f"  {row['ticker']}: %{row['prob_up']*100:.1f} {direction} "
                f"(veri: {row['feature_date']})"
            )


if __name__ == "__main__":
    main()
