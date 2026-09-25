"""KAP (kamuyu aydınlatma) bildirimlerini çeker — resmi JSON API (ücretsiz web uç noktası)."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import requests

from pipeline.db import (
    get_connection,
    init_schema,
    insert_news,
    link_news_symbols,
    rebuild_sentiment_daily,
)
from pipeline.entity_linker import build_symbol_index

logger = logging.getLogger(__name__)

KAP_CRITERIA_URL = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"
KAP_BILDIRIM_URL = "https://www.kap.org.tr/tr/Bildirim/{index}"
BILDIRIM_RE = re.compile(r"/Bildirim/(\d+)", re.I)

HEADERS = {
    "User-Agent": "BorsaAI/0.1 (educational)",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
}


def _criteria_body(from_date: str, to_date: str) -> dict[str, Any]:
    return {
        "fromDate": from_date,
        "toDate": to_date,
        "memberType": "",
        "mkkMemberOidList": [],
        "inactiveMkkMemberOidList": [],
        "disclosureClass": "",
        "subjectList": [],
        "isLate": "",
        "mainSector": "",
        "sector": "",
        "subSector": "",
        "marketOid": "",
        "index": "",
        "bdkReview": "",
        "bdkMemberOidList": [],
        "year": "",
        "term": "",
        "ruleType": "",
    }


def fetch_disclosures(days: int = 7, end_date: date | None = None) -> list[dict[str, Any]]:
    """`end_date`'den geriye N gün tüm şirket bildirimleri (varsayılan: bugüne kadar).
    Geniş aralıklar (aylar/yıllar) API'de 500 ile patlıyor — geçmiş bir dönemi
    doldurmak için `days`'i küçük (örn. 7) tutup `end_date`'i kaydırarak
    parça parça çağır (bkz. sync_kap_disclosures)."""
    end = end_date or date.today()
    start = end - timedelta(days=days)
    resp = requests.post(
        KAP_CRITERIA_URL,
        json=_criteria_body(start.isoformat(), end.isoformat()),
        headers=HEADERS,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Beklenmeyen KAP yanıtı: {type(data)}")
    logger.info("KAP: %d bildirim (%s — %s)", len(data), start, end)
    return data


def parse_kap_datetime(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None


def stock_codes_to_symbol_ids(
    stock_codes: str | None,
    by_base: dict[str, int],
) -> list[tuple[int, str]]:
    if not stock_codes:
        return []
    links: list[tuple[int, str]] = []
    for code in stock_codes.split(","):
        base = code.strip().upper()
        if not base:
            continue
        sid = by_base.get(base)
        if sid:
            links.append((sid, f"kap:{base}"))
    return links


def _disclosure_title(row: dict[str, Any]) -> str:
    company = (row.get("kapTitle") or "").strip()
    subject = (row.get("subject") or row.get("summary") or "Bildirim").strip()
    if company and subject:
        return f"{company} — {subject}"
    return company or subject or "KAP Bildirimi"


def sync_kap_disclosures(
    days: int = 7,
    enrich_existing_urls: bool = True,
    end_date: date | None = None,
) -> dict[str, int]:
    """KAP bildirimlerini news_raw'a yazar ve doğrudan hisse bağlar."""
    rows = fetch_disclosures(days, end_date=end_date)
    by_index = {int(r["disclosureIndex"]): r for r in rows if r.get("disclosureIndex")}

    stats = {
        "kap_fetched": len(rows),
        "new_news": 0,
        "updated_links": 0,
        "symbol_links": 0,
        "skipped_no_code": 0,
    }

    with get_connection() as conn:
        init_schema(conn)
        _, by_base = build_symbol_index(conn)

        for row in rows:
            idx = row.get("disclosureIndex")
            if not idx:
                continue
            url = KAP_BILDIRIM_URL.format(index=idx)
            title = _disclosure_title(row)
            summary = (row.get("summary") or row.get("subject") or "")[:2000]
            published = parse_kap_datetime(row.get("publishDate"))

            news_id = insert_news(
                conn,
                source="KAP",
                feed_id="kap_byCriteria",
                title=title,
                summary=summary or None,
                url=url,
                published_at=published,
                language="tr",
                content=summary or None,
            )
            if news_id is None:
                from pipeline.db import get_news_id_by_url

                news_id = get_news_id_by_url(conn, url)
            else:
                stats["new_news"] += 1

            if news_id is None:
                continue

            links = stock_codes_to_symbol_ids(row.get("stockCodes"), by_base)
            if links:
                link_news_symbols(conn, news_id, links)
                stats["symbol_links"] += len(links)
            else:
                stats["skipped_no_code"] += 1

        if enrich_existing_urls and by_index:
            stats["updated_links"] += _enrich_existing_kap_urls(conn, by_index, by_base)

        rebuild_sentiment_daily(conn)

    return stats


def _enrich_existing_kap_urls(
    conn,
    by_index: dict[int, dict],
    by_base: dict[str, int],
) -> int:
    """Google News vb. kaynaklardan gelen KAP URL'lerine kod eşlemesi ekler."""
    from pipeline.db import get_news_id_by_url

    updated = 0
    kap_news = conn.execute(
        "SELECT id, url FROM news_raw WHERE url LIKE '%kap.org.tr%Bildirim%'"
    ).fetchall()

    for item in kap_news:
        m = BILDIRIM_RE.search(item["url"] or "")
        if not m:
            continue
        idx = int(m.group(1))
        row = by_index.get(idx)
        if not row:
            continue
        links = stock_codes_to_symbol_ids(row.get("stockCodes"), by_base)
        if not links:
            continue
        news_id = int(item["id"])
        link_news_symbols(conn, news_id, links)
        updated += len(links)
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(sync_kap_disclosures())
