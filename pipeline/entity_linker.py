"""Haber metnini izlediğimiz hisselerle eşleştirir (geliştirilmiş)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import sqlite3
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALIASES_PATH = PROJECT_ROOT / "config" / "symbol_aliases.yaml"
BLOCKLIST_PATH = PROJECT_ROOT / "config" / "entity_blocklist.yaml"

MIN_NAME_LEN = 5
NAME_STOPWORDS = frozenset(
    {
        "inc",
        "inc.",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "co",
        "company",
        "holding",
        "group",
        "a.ş",
        "a.s",
        "a.s.",
        "as",
        "ve",
        "the",
        "and",
        "class",
        "hisseleri",
        "hisse",
        "sanayi",
        "ticaret",
    }
)

# BIST: 3-6 harf; ABD: 1-5 büyük harf
RE_TICKER_PARENS = re.compile(r"[\(\[]([A-Z]{1,5}(?:\.IS)?)[\)\]]")
RE_TICKER_DOLLAR = re.compile(r"\$([A-Z]{1,5})\b")
RE_TICKER_BIST = re.compile(r"\b([A-Z]{3,6})(?:\.IS)?\b")
RE_TICKER_EXPLICIT = re.compile(r"\b([A-Z]{1,5}\.IS)\b")


@dataclass
class SymbolPattern:
    symbol_id: int
    ticker: str
    base: str
    patterns: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    # casefold("İ") -> "i" + BİRLEŞEN NOKTA (U+0307) üretir, bu da "MİGROS"
    # (KAP'ın büyük harfli başlıkları) ile "Migros"un asla eşleşmemesine
    # yol açıyordu — Türkçe büyük harfleri önce elle normal harfe çevir.
    text = text.replace("İ", "i").replace("I", "ı")
    text = unicodedata.normalize("NFKC", text)
    return text.casefold()


def _base_ticker(ticker: str) -> str:
    return ticker.replace(".IS", "").upper()


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    path = path or ALIASES_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("aliases", {}) or {}


def load_blocklist() -> frozenset[str]:
    if not BLOCKLIST_PATH.exists():
        return frozenset()
    with open(BLOCKLIST_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = {_base_ticker(x) for x in data.get("blocklist", []) if x}
    return frozenset(items)


def _aliases_from_company_name(name: str) -> list[str]:
    """DB şirket adından otomatik kısa anahtar kelimeler."""
    if not name or len(name.strip()) < MIN_NAME_LEN:
        return []
    raw = name.strip()
    results = {raw}

    # Parantez içi kısaltma: "Türk Hava Yolları (THYAO)"
    paren = re.search(r"\(([A-Z]{2,6}(?:\.IS)?)\)", raw)
    if paren:
        results.add(paren.group(1))

    cleaned = re.sub(r"\([^)]*\)", "", raw)
    cleaned = re.sub(r"[,.]", " ", cleaned)
    tokens = [t for t in cleaned.split() if t]
    norm_tokens = [_normalize(t) for t in tokens]

    significant = [
        t
        for t, nt in zip(tokens, norm_tokens)
        if len(t) >= 4 and nt not in NAME_STOPWORDS
    ]
    if len(significant) >= 2:
        results.add(" ".join(significant[:3]))
    elif significant:
        results.add(significant[0])

    return [r for r in results if len(r) >= 4]


def build_symbol_index(conn: sqlite3.Connection) -> tuple[list[SymbolPattern], dict[str, int]]:
    """
    Döner: (pattern listesi, base_ticker -> symbol_id haritası)
    """
    aliases_cfg = load_aliases()
    rows = conn.execute(
        "SELECT id, ticker, name, market FROM symbols ORDER BY ticker"
    ).fetchall()

    by_base: dict[str, int] = {}
    index: list[SymbolPattern] = []

    for row in rows:
        symbol_id = int(row["id"])
        ticker = row["ticker"]
        base = _base_ticker(ticker)
        by_base[base] = symbol_id
        patterns: set[str] = set()

        # <4 harfli ticker'lar gevşek kelime eşleşmesine girmez ("V" -> "Hemi
        # V-8 engine" gibi alakasız metinlere bağlanıyordu) — regex yolundaki
        # (extract_tickers_from_text) aynı len>=4 kuralıyla tutarlı; kısa
        # ticker'lar $V / (V) / V.IS regex'leri ve manuel alias ile yakalanır.
        if len(base) >= 4:
            patterns.add(base)
            patterns.add(ticker)
            patterns.add(ticker.casefold())

        name = row["name"]
        if name:
            patterns.add(name.strip())
            patterns.update(_aliases_from_company_name(name))

        for alias in aliases_cfg.get(ticker, []):
            if alias and len(alias.strip()) >= 3:
                patterns.add(alias.strip())

        index.append(
            SymbolPattern(
                symbol_id=symbol_id,
                ticker=ticker,
                base=base,
                patterns=sorted(patterns, key=len, reverse=True),
            )
        )
    return index, by_base


def _word_match(haystack: str, needle: str) -> bool:
    needle_n = _normalize(needle)
    if len(needle_n) <= 5:
        return bool(re.search(rf"\b{re.escape(needle_n)}\b", haystack, re.UNICODE))
    return needle_n in haystack


def extract_tickers_from_text(text: str, by_base: dict[str, int], blocklist: frozenset[str]) -> list[tuple[str, str]]:
    """Regex ile metinden ticker adayları. Döner: (base_ticker, reason)."""
    if not text:
        return []

    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(base: str, reason: str) -> None:
        base = _base_ticker(base)
        if base in blocklist or base in seen:
            return
        if base in by_base:
            seen.add(base)
            found.append((base, reason))

    for m in RE_TICKER_PARENS.finditer(text):
        add(m.group(1), f"regex:paren:{m.group(1)}")
    for m in RE_TICKER_DOLLAR.finditer(text):
        add(m.group(1), f"regex:dollar:{m.group(1)}")
    for m in RE_TICKER_EXPLICIT.finditer(text):
        add(m.group(1), f"regex:explicit:{m.group(1)}")
    for m in RE_TICKER_BIST.finditer(text):
        cand = m.group(1)
        if len(cand) >= 4:
            add(cand, f"regex:bist:{cand}")

    return found


def match_symbols(
    text: str,
    index: list[SymbolPattern],
    by_base: dict[str, int] | None = None,
    blocklist: frozenset[str] | None = None,
) -> list[tuple[int, str]]:
    if not text.strip():
        return []

    if by_base is None:
        by_base = {item.base: item.symbol_id for item in index}
    if blocklist is None:
        blocklist = load_blocklist()

    haystack = _normalize(text)
    found: dict[int, str] = {}

    # 1) Regex ile doğrudan ticker
    for base, reason in extract_tickers_from_text(text, by_base, blocklist):
        sid = by_base[base]
        found[sid] = reason

    # 2) Şirket adı / alias
    for item in index:
        if item.symbol_id in found:
            continue
        for pattern in item.patterns:
            if _word_match(haystack, pattern):
                found[item.symbol_id] = f"{item.base}:alias:{pattern[:40]}"
                break

    return list(found.items())


def relink_all_news(conn: sqlite3.Connection) -> dict[str, int]:
    """Metin tabanlı eşleşmeleri yeniden kurar. KAP kaynaklı haberlere
    DOKUNMAZ: onların bağlantıları kap_sync tarafından bildirimin kendi
    `stockCodes` alanından kuruluyor (match_reason='kap:*'), bu metin
    tahmininden çok daha kesin bir kaynak."""
    index, by_base = build_symbol_index(conn)
    blocklist = load_blocklist()

    conn.execute(
        "DELETE FROM news_symbol_links WHERE news_id IN "
        "(SELECT id FROM news_raw WHERE source != 'KAP')"
    )
    rows = conn.execute(
        "SELECT id, title, summary FROM news_raw WHERE source != 'KAP'"
    ).fetchall()

    stats = {"news": 0, "links": 0, "news_with_link": 0}
    for row in rows:
        text = f"{row['title']}\n{row['summary'] or ''}"
        links = match_symbols(text, index, by_base, blocklist)
        if links:
            stats["news_with_link"] += 1
            from pipeline.db import link_news_symbols

            link_news_symbols(conn, int(row["id"]), links)
            stats["links"] += len(links)
        stats["news"] += 1
    return stats
