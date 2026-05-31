#!/usr/bin/env python3
"""RSS haber çekme — python scripts/run_fetch_news.py"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.fetch_news_rss import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RSS haber kaynaklarını tara")
    parser.add_argument(
        "--max-per-feed",
        type=int,
        metavar="N",
        help="Feed başına en fazla N haber (test için)",
    )
    parser.add_argument(
        "--relink",
        action="store_true",
        help="Mevcut haberler için sembol eşleşmesini yeniden çalıştır",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = run(max_per_feed=args.max_per_feed, relink_existing=args.relink)
    print("\n--- Özet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
