#!/usr/bin/env python3
"""Tüm haberleri hisselerle yeniden eşleştir."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.db import get_connection, init_schema, rebuild_sentiment_daily, count_news_links  # noqa: E402
from pipeline.entity_linker import relink_all_news  # noqa: E402


def main() -> None:
    with get_connection() as conn:
        init_schema(conn)
        before = count_news_links(conn)
        stats = relink_all_news(conn)
        daily = rebuild_sentiment_daily(conn)
        after = count_news_links(conn)

    print("--- Eşleştirme ---")
    print(f"  Haber işlendi: {stats['news']}")
    print(f"  Haber (en az 1 hisse): {stats['news_with_link']}")
    print(f"  Toplam bağlantı: {stats['links']}")
    print(f"  Önceki bağlantı: {before} -> şimdi: {after}")
    print(f"  sentiment_daily satır: {daily}")


if __name__ == "__main__":
    main()
