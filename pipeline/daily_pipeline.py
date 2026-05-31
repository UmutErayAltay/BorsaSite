"""Günlük otomatik veri + tahmin pipeline."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def run(
    incremental_days: int = 7,
    skip_sentiment: bool = False,
    skip_train: bool = True,
    skip_predict: bool = False,
    relink_news: bool = True,
    log_dir: Path | None = None,
) -> dict:
    log_dir = log_dir or LOG_DIR
    log_file = log_dir / f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _setup_logging(log_file)

    logger = logging.getLogger(__name__)
    started = time.time()
    results: dict = {"log_file": str(log_file), "steps": {}}

    def step(name: str, fn):
        logger.info("=== %s ===", name)
        t0 = time.time()
        try:
            out = fn()
            results["steps"][name] = {"ok": True, "result": out, "seconds": round(time.time() - t0, 1)}
            logger.info("%s tamamlandı (%.1fs): %s", name, time.time() - t0, out)
        except Exception as e:
            results["steps"][name] = {"ok": False, "error": str(e), "seconds": round(time.time() - t0, 1)}
            logger.exception("%s başarısız: %s", name, e)
            raise

    from pipeline.fetch_prices import run as fetch_prices
    from pipeline.fetch_news_rss import run as fetch_news
    from pipeline.analyze_sentiment import run as analyze_sentiment
    from pipeline.train_model import run as train_model
    from pipeline.predict_model import run as predict_model
    from pipeline.db import get_connection, init_schema, rebuild_sentiment_daily
    from pipeline.entity_linker import relink_all_news

    step("prices", lambda: fetch_prices(incremental_days=incremental_days))
    step("news", lambda: fetch_news())

    from pipeline.kap_sync import sync_kap_disclosures

    step(
        "kap",
        lambda: sync_kap_disclosures(
            days=max(incremental_days, 7),
            enrich_existing_urls=True,
        ),
    )

    if relink_news:
        def _relink():
            with get_connection() as conn:
                init_schema(conn)
                stats = relink_all_news(conn)
                rebuild_sentiment_daily(conn)
                return stats

        step("relink", _relink)

    if not skip_sentiment:
        step("sentiment", lambda: analyze_sentiment())

    if not skip_train:
        step("train", lambda: train_model())

    if not skip_predict:
        step("predict", lambda: predict_model())

    results["total_seconds"] = round(time.time() - started, 1)
    logger.info("Pipeline bitti (%.1fs). Log: %s", results["total_seconds"], log_file)
    return results
