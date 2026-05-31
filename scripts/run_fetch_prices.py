#!/usr/bin/env python3
"""Fiyat verisi çekme — proje kökünden: python scripts/run_fetch_prices.py"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.fetch_prices import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="BIST + ABD günlük fiyat verisi çek")
    parser.add_argument(
        "--period",
        default="2y",
        help="yfinance period (örn. 1y, 2y, max). --incremental ile birlikte kullanılmaz.",
    )
    parser.add_argument(
        "--incremental",
        type=int,
        metavar="DAYS",
        help="Son N günü güncelle (günlük görev için, örn. 7)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = run(period=args.period, incremental_days=args.incremental)
    print("\n--- Özet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
