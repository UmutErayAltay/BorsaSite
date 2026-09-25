#!/usr/bin/env python3
"""Gün içi haber izleme — açık pozisyonlarda güçlü olumsuz bir KAP bildirimi
varsa erken satar, YENİ ALIM yapmaz. python scripts/run_intraday_watch.py"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from pipeline.intraday_watch import DEFAULT_NEGATIVE_THRESHOLD, run_intraday_watch  # noqa: E402

load_project_env()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=DEFAULT_NEGATIVE_THRESHOLD,
        help=f"Bu skorun ALTINDAKİ (-1..1) haberler erken çıkışı tetikler (varsayılan {DEFAULT_NEGATIVE_THRESHOLD})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = run_intraday_watch(negative_threshold=args.negative_threshold)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
