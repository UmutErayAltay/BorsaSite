"""RSS kaynaklarından haber çeker ve hisselerle eşleştirir."""

from __future__ import annotations

import logging
import re
from calendar import timegm
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

import feedparser
import yaml

from pipeline.db import (
    count_news,
    count_news_links,
    get_connection,
    get_news_id_by_url,
    init_schema,
    insert_news,
    link_news_symbols,
)
from pipeline.entity_linker import build_symbol_index, match_symbols

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEEDS_PATH = PROJECT_ROOT / "config" / "news_feeds.yaml"

USER_AGENT = "BorsaAI/0.1 (educational; RSS reader)"


def load_feeds_config(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or FEEDS_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("feeds", []))


def normalize_url(url: str) -> str:
    url, _frag = urldefrag(url.strip())
    return url


def parse_published(entry: dict) -> str | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                dt = datetime.fromtimestamp(timegm(parsed), tz=timezone.utc)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OverflowError, OSError):
                pass

    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue
    return None


def entry_summary(entry: dict) -> str | None:
    if entry.get("summary"):
        text = entry.summary
    elif entry.get("description"):
        text = entry.description
    else:
        return None
    # HTML etiketlerini kaba temizle
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000] if text else None


def fetch_feed(url: str) -> feedparser.FeedParserDict:
    return feedparser.parse(
        url,
        agent=USER_AGENT,
        request_headers={"User-Agent": USER_AGENT},
    )


def process_entry(
    entry: dict,
    feed: dict[str, Any],
    symbol_index: list,
    by_base: dict[str, int],
    blocklist,
    conn,
    stats: dict[str, int],
) -> None:
    title = (entry.get("title") or "").strip()
    link = entry.get("link") or entry.get("id")
    if not title or not link:
        stats["skipped"] += 1
        return

    url = normalize_url(link)
    summary = entry_summary(entry)
    published_at = parse_published(entry)
    language = feed.get("language", "tr")

    news_id = insert_news(
        conn,
        source=feed["name"],
        feed_id=feed["id"],
        title=title,
        summary=summary,
        url=url,
        published_at=published_at,
        language=language,
        content=summary,
    )

    if news_id is None:
        news_id = get_news_id_by_url(conn, url)
        if news_id is None:
            stats["skipped"] += 1
            return
        stats["duplicates"] += 1
    else:
        stats["new_articles"] += 1

    text = f"{title}\n{summary or ''}"
    links = match_symbols(text, symbol_index, by_base, blocklist)
    if links:
        link_news_symbols(conn, news_id, links)
        stats["linked_articles"] += 1
        stats["symbol_links"] += len(links)


def run(
    feeds_path: Path | None = None,
    max_per_feed: int | None = None,
    relink_existing: bool = False,
) -> dict[str, int]:
    """
    Tüm RSS kaynaklarını tara.

    relink_existing: URL'si DB'de olan haberler için sembol eşleşmesini yenile.
    """
    feeds = load_feeds_config(feeds_path)
    stats: dict[str, int] = {
        "feeds_ok": 0,
        "feeds_failed": 0,
        "new_articles": 0,
        "duplicates": 0,
        "skipped": 0,
        "linked_articles": 0,
        "symbol_links": 0,
        "entries_processed": 0,
    }

    with get_connection() as conn:
        init_schema(conn)
        symbol_index, by_base = build_symbol_index(conn)
        blocklist = None
        try:
            from pipeline.entity_linker import load_blocklist

            blocklist = load_blocklist()
        except Exception:
            pass

        if relink_existing:
            from pipeline.entity_linker import relink_all_news

            relink_stats = relink_all_news(conn)
            stats["relink"] = relink_stats
            return stats

        for feed in feeds:
            url = feed["url"]
            try:
                parsed = fetch_feed(url)
                if parsed.bozo and not parsed.entries:
                    logger.warning(
                        "Feed hatası %s: %s", feed["id"], parsed.bozo_exception
                    )
                    stats["feeds_failed"] += 1
                    continue

                entries = parsed.entries
                if max_per_feed is not None:
                    entries = entries[:max_per_feed]

                for entry in entries:
                    process_entry(entry, feed, symbol_index, by_base, blocklist, conn, stats)
                    stats["entries_processed"] += 1

                stats["feeds_ok"] += 1
                logger.info(
                    "%s: %d girdi işlendi",
                    feed["id"],
                    len(entries),
                )
            except Exception as e:
                logger.exception("Feed başarısız %s: %s", feed["id"], e)
                stats["feeds_failed"] += 1

        stats["total_news_db"] = count_news(conn)
        stats["total_links_db"] = count_news_links(conn)

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())
