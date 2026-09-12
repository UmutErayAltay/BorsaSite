#!/usr/bin/env python3
"""Günlük pipeline — python scripts/run_daily.py"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_env  # noqa: E402
from pipeline.daily_pipeline import run  # noqa: E402

load_project_env()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Günlük: fiyat + haber + eşleştirme + sentiment + tahmin + alım-satım"
    )
    parser.add_argument("--days", type=int, default=7, help="Fiyat: son N gün")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--train", action="store_true", help="Modeli yeniden eğit (yavaş)")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-trading", action="store_true", help="Alım-satım adımını atla")
    parser.add_argument("--no-relink", action="store_true", help="Haber eşleştirmesini atla")
    args = parser.parse_args()

    # Render Cron Job her gün aynı komutu çalıştırır — haftalık yeniden
    # eğitimi ayrı bir zamanlanmış iş yerine, aynı script içinde gün
    # kontrolüyle tetikliyoruz (tek cron job, tek yerde mantık).
    auto_train = args.train or datetime.now().weekday() == 0  # Pazartesi

    result = run(
        incremental_days=args.days,
        skip_sentiment=args.skip_sentiment,
        skip_train=not auto_train,
        skip_predict=args.skip_predict,
        skip_trading=args.skip_trading,
        relink_news=not args.no_relink,
    )
    print("\n--- Pipeline özeti ---")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
