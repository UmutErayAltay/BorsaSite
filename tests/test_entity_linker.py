import pytest

from pipeline.db import insert_news, link_news_symbols, upsert_symbol
from pipeline.entity_linker import (
    _aliases_from_company_name,
    _normalize,
    build_symbol_index,
    match_symbols,
    relink_all_news,
)

TUPRS_NAME = "Türkiye Petrol Rafinerileri A.Ş."
IBM_NAME = "International Business Machines Corporation"


def test_normalize_turkish_dotted_i_is_plain_i():
    assert _normalize("MİGROS") == _normalize("Migros")


def test_normalize_uppercase_dotless_i():
    assert _normalize("VAKIFBANK") == _normalize("Vakıfbank")


def test_normalize_ascii_unchanged():
    assert _normalize("COCA-COLA") == _normalize("Coca-Cola")


def test_multiword_name_drops_bare_first_word():
    aliases = _aliases_from_company_name(TUPRS_NAME)
    assert "Türkiye" not in aliases
    assert "Türkiye Petrol Rafinerileri" in aliases


def test_english_multiword_name_drops_bare_first_word():
    aliases = _aliases_from_company_name(IBM_NAME)
    assert "International" not in aliases
    assert "International Business Machines" in aliases


def test_single_significant_token_keeps_bare_word():
    aliases = _aliases_from_company_name("Migros Ticaret A.Ş.")
    assert "Migros" in aliases


def test_paren_ticker_still_extracted():
    aliases = _aliases_from_company_name("Türk Hava Yolları (THYAO)")
    assert "THYAO" in aliases


@pytest.fixture
def index(conn):
    upsert_symbol(conn, "TUPRS.IS", "BIST", "TRY", TUPRS_NAME)
    upsert_symbol(conn, "MGROS.IS", "BIST", "TRY", "Migros Ticaret A.Ş.")
    upsert_symbol(conn, "V", "US", "USD", "Visa Inc.")
    upsert_symbol(conn, "IBM", "US", "USD", IBM_NAME)
    return build_symbol_index(conn)


def _tickers(conn, found, index_by_id):
    return {index_by_id[sid] for sid, _ in found}


def test_swimming_news_does_not_match_tuprs(conn, index):
    idx, by_base = build_symbol_index(conn)
    found = match_symbols(
        "Türkiye, Avrupa Para Yüzme Şampiyonası'nı 16 madalyayla tamamladı", idx, by_base
    )
    assert found == []


def test_company_phrase_still_matches_tuprs(conn, index):
    idx, by_base = build_symbol_index(conn)
    found = match_symbols(
        "Türkiye Petrol Rafinerileri üçüncü çeyrek bilançosunu açıkladı", idx, by_base
    )
    assert by_base["TUPRS"] in dict(found)


def test_manual_alias_still_matches_tuprs(conn, index):
    idx, by_base = build_symbol_index(conn)
    found = match_symbols("Tüpraş kâr payı dağıtacak", idx, by_base)
    assert by_base["TUPRS"] in dict(found)


def test_international_news_does_not_match_ibm(conn, index):
    idx, by_base = build_symbol_index(conn)
    found = match_symbols("Marriott International opens new hotel in Antalya", idx, by_base)
    assert by_base["IBM"] not in dict(found)


def test_bare_short_ticker_letter_not_matched(conn, index):
    idx, by_base = build_symbol_index(conn)
    for text in ("The Hemi V-8 engine roared to life", "ASML Holding N.V. reported earnings"):
        found = match_symbols(text, idx, by_base)
        assert by_base["V"] not in dict(found)


def test_short_ticker_matches_via_dollar_and_parens(conn, index):
    idx, by_base = build_symbol_index(conn)
    found = match_symbols("$V rose 3% today", idx, by_base)
    d = dict(found)
    assert by_base["V"] in d
    assert d[by_base["V"]].startswith("regex:")

    found2 = match_symbols("Visa Inc. (V) reported strong earnings", idx, by_base)
    assert by_base["V"] in dict(found2)


def test_short_ticker_matches_via_company_alias(conn, index):
    idx, by_base = build_symbol_index(conn)
    found = match_symbols("Visa beat earnings estimates this quarter", idx, by_base)
    assert by_base["V"] in dict(found)


def test_build_index_short_ticker_patterns(conn, index):
    idx, by_base = build_symbol_index(conn)
    v_pattern = next(p for p in idx if p.base == "V")
    ibm_pattern = next(p for p in idx if p.base == "IBM")
    tuprs_pattern = next(p for p in idx if p.base == "TUPRS")

    assert "V" not in v_pattern.patterns
    assert "Visa" in v_pattern.patterns
    assert "IBM" in ibm_pattern.patterns  # manuel alias kasıtlı olarak geri ekliyor
    assert "TUPRS" in tuprs_pattern.patterns


def test_kap_allcaps_turkish_title_matches(conn, index):
    idx, by_base = build_symbol_index(conn)
    found = match_symbols("MİGROS TİCARET A.Ş. — PAY ALIM SATIM İŞLEMLERİ", idx, by_base)
    assert by_base["MGROS"] in dict(found)


def test_relink_preserves_kap_links(conn, index):
    tuprs_id = next(p.symbol_id for p in index[0] if p.base == "TUPRS")
    migros_id = next(p.symbol_id for p in index[0] if p.base == "MGROS")

    kap_news_id = insert_news(
        conn, "KAP", None, "ÖZEL DURUM AÇIKLAMASI", None,
        "https://kap.example/1", "2026-09-12", "tr",
    )
    link_news_symbols(conn, kap_news_id, [(tuprs_id, "kap:TUPRS")])

    rss_news_id = insert_news(
        conn, "dunya", None, "Migros Ticaret A.Ş. üçüncü çeyrek sonuçlarını açıkladı", None,
        "https://dunya.example/1", "2026-09-12", "tr",
    )
    link_news_symbols(conn, rss_news_id, [(tuprs_id, "bogus:stale")])

    relink_all_news(conn)

    kap_links = conn.execute(
        "SELECT symbol_id, match_reason FROM news_symbol_links WHERE news_id = ?",
        (kap_news_id,),
    ).fetchall()
    assert [(r["symbol_id"], r["match_reason"]) for r in kap_links] == [(tuprs_id, "kap:TUPRS")]

    rss_links = conn.execute(
        "SELECT symbol_id FROM news_symbol_links WHERE news_id = ?", (rss_news_id,)
    ).fetchall()
    assert [r["symbol_id"] for r in rss_links] == [migros_id]


def test_relink_stats_exclude_kap_rows(conn, index):
    insert_news(
        conn, "KAP", None, "ÖZEL DURUM AÇIKLAMASI", None,
        "https://kap.example/2", "2026-09-12", "tr",
    )
    insert_news(
        conn, "dunya", None, "Genel bir haber", None,
        "https://dunya.example/2", "2026-09-12", "tr",
    )
    insert_news(
        conn, "dunya", None, "Bir başka genel haber", None,
        "https://dunya.example/3", "2026-09-12", "tr",
    )

    stats = relink_all_news(conn)

    assert stats["news"] == 2
