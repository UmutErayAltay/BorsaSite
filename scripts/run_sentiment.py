#!/usr/bin/env python3
"""Duygu analizi — python scripts/run_sentiment.py"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from pipeline.analyze_sentiment import run  # noqa: E402

load_project_env()


def main() -> None:
    parser = argparse.ArgumentParser(description="Haber duygu analizi (FinBERT + TR BERT)")
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="En fazla N haber analiz et (ilk çalıştırma testi)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Zaten analiz edilmiş haberleri yeniden işle",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Günlük hisse sentiment özetini güncelleme",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = run(
        limit=args.limit,
        force=args.force,
        skip_aggregate=args.skip_aggregate,
    )
    print("\n--- Özet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
