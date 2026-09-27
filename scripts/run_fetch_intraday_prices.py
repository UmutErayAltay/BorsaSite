#!/usr/bin/env python3
"""Intraday fiyat verisi çekme — proje kökünden:
python scripts/run_fetch_intraday_prices.py --interval 15m"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.fetch_intraday_prices import (  # noqa: E402
    MAX_PERIOD_BY_INTERVAL,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="BIST + ABD intraday fiyat verisi çek")
    parser.add_argument(
        "--interval",
        default="15m",
        choices=sorted(MAX_PERIOD_BY_INTERVAL),
        help="Bar aralığı. Yfinance sınırları: 1m ~7g, 5m/15m/30m 60g, 1h 730g.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = run(interval=args.interval)
    print("\n--- Özet ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
