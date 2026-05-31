#!/usr/bin/env python3
"""KAP bildirimlerini çek ve hisselerle eşleştir."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.kap_sync import sync_kap_disclosures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="KAP bildirim senkronizasyonu")
    parser.add_argument("--days", type=int, default=7, help="Son N gün (varsayılan 7)")
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Mevcut KAP URL'lerini zenginleştirme",
    )
    args = parser.parse_args()

    stats = sync_kap_disclosures(
        days=args.days,
        enrich_existing_urls=not args.no_enrich,
    )
    print("\n--- KAP özeti ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
