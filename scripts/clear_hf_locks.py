#!/usr/bin/env python3
"""Yarım kalmış Hugging Face indirme kilitlerini temizler."""

import os
import sys
from pathlib import Path

CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))


def main() -> None:
    if not CACHE.exists():
        print(f"Önbellek yok: {CACHE}")
        return

    removed = 0
    for lock in CACHE.rglob("*.lock"):
        try:
            lock.unlink()
            removed += 1
            print(f"Silindi: {lock}")
        except OSError as e:
            print(f"Atlandı {lock}: {e}", file=sys.stderr)

    print(f"\nToplam {removed} kilit dosyası silindi.")
    print("Ardından tek bir terminalde çalıştırın:")
    print("  python scripts/run_sentiment.py --limit 20")


if __name__ == "__main__":
    main()
