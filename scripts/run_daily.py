#!/usr/bin/env python3
"""Günlük pipeline — python scripts/run_daily.py"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from pipeline.daily_pipeline import run  # noqa: E402

load_project_env()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Günlük: fiyat + haber + eşleştirme + sentiment + tahmin"
    )
    parser.add_argument("--days", type=int, default=7, help="Fiyat: son N gün")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--train", action="store_true", help="Modeli yeniden eğit (yavaş)")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--no-relink", action="store_true", help="Haber eşleştirmesini atla")
    args = parser.parse_args()

    result = run(
        incremental_days=args.days,
        skip_sentiment=args.skip_sentiment,
        skip_train=not args.train,
        skip_predict=args.skip_predict,
        relink_news=not args.no_relink,
    )
    print("\n--- Pipeline özeti ---")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
