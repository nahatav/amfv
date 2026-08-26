"""Tests for StatPearls scraping helpers."""

import json

import httpx

from amfv_datasets.scraping.html import LinkMode
from amfv_datasets.scraping.statpearls import (
    ChapterRef,
    StatpearlsFetchError,
    list_chapters,
    scrape_chapter,
    scrape_chapter_by_url,
    search_section_uids,
    summarize_chapters,
)


def _bookinfo(chapter_title: str) -> str:
    return (
        "<Info><Path>"
        '<Parent id="statpearls" role="source" type="book" uid="4403668"><Title>StatPearls</Title></Parent>'
        f'<Parent id="article-1" role="document" type="chapter" uid="1"><Title>{chapter_title}</Title></Parent>'
        '<Self id="article-1.s1" role="object" type="sec" uid="111"><Title>Introduction</Title></Self>'
        "</Path></Info>"
    )


def _esummary_payload(records: dict[str, dict]) -> dict:
    return {"result": {"uids": list(records), **records}}


def test_search_section_uids_returns_the_result_idlist() -> None:
    """The section search page returns the raw E-utilities id list."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/entrez/eutils/esearch.fcgi"
        assert request.url.params["db"] == "books"
        assert request.url.params["term"] == "statpearls[book]"
        assert request.url.params["retstart"] == "0"
        # E-utilities policy asks callers to identify themselves.
        assert request.url.params["tool"] == "amfv-datasets"
        payload = {"esearchresult": {"idlist": ["111", "112"]}}
        return httpx.Response(200, text=json.dumps(payload))

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert search_section_uids(client, retstart=0) == ["111", "112"]


def test_summarize_chapters_extracts_accession_and_chapter_title() -> None:
    """Section summaries resolve to their parent chapter's accession and title."""
    payload = _esummary_payload(
        {
            "111": {"chapteraccessionid": "NBK111", "bookinfo": _bookinfo("Bilateral Vocal Cord Paralysis")},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/entrez/eutils/esummary.fcgi"
        assert request.url.params["id"] == "111"
        assert request.url.params["tool"] == "amfv-datasets"
        return httpx.Response(200, text=json.dumps(payload))

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert summarize_chapters(client, ["111"]) == [
        ChapterRef(accession="NBK111", title="Bilateral Vocal Cord Paralysis")
    ]


def test_summarize_chapters_skips_records_without_a_chapter_accession() -> None:
    """Section records missing a chapter accession are skipped rather than erroring."""
    payload = _esummary_payload({"111": {"bookinfo": _bookinfo("Orphan Section")}})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps(payload))

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert summarize_chapters(client, ["111"]) == []


def test_summarize_chapters_returns_empty_for_no_uids() -> None:
    """No section UIDs means no E-utilities request and no chapters."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not fetch summaries for an empty UID list")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert summarize_chapters(client, []) == []


def test_list_chapters_deduplicates_chapters_seen_on_earlier_pages() -> None:
    """Chapters already seen are skipped even if their sections reappear."""
    payload = _esummary_payload(
        {
            "111": {"chapteraccessionid": "NBK111", "bookinfo": _bookinfo("Chapter One")},
            "112": {"chapteraccessionid": "NBK111", "bookinfo": _bookinfo("Chapter One")},
            "113": {"chapteraccessionid": "NBK222", "bookinfo": _bookinfo("Chapter Two")},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/entrez/eutils/esearch.fcgi":
            return httpx.Response(200, text=json.dumps({"esearchresult": {"idlist": ["111", "112", "113"]}}))
        return httpx.Response(200, text=json.dumps(payload))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    seen: set[str] = set()

    first_page = list_chapters(client, 1, seen=seen)

    assert first_page == [
        ChapterRef(accession="NBK111", title="Chapter One"),
        ChapterRef(accession="NBK222", title="Chapter Two"),
    ]
    assert seen == {"NBK111", "NBK222"}


def test_scrape_chapter_converts_body_content_to_markdown() -> None:
    """A chapter's `itemprop="text"` body is converted to markdown."""
    html = """
        <html><body>
          <div id="maincontent">
            <div class="body-content" itemprop="text">
              <div id="article-1.s1">
                <h2>Introduction</h2>
                <p>Bilateral vocal cord paralysis is a rare condition.</p>
                <div id="article-1.s1.1"><p>A nested subsection.</p></div>
              </div>
              <div id="article-1.s2"><h2>Etiology</h2><p>Causes vary.</p></div>
            </div>
          </div>
        </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/books/NBK111/"
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    document = scrape_chapter(client, ChapterRef(accession="NBK111", title="Bilateral Vocal Cord Paralysis"))

    assert document.source == "statpearls"
    assert document.external_id == "statpearls-NBK111"
    assert document.title == "Bilateral Vocal Cord Paralysis"
    assert document.url == "https://www.ncbi.nlm.nih.gov/books/NBK111/"
    assert document.content == (
        "## Introduction\n\nBilateral vocal cord paralysis is a rare condition.\n\n"
        "A nested subsection.\n\n"
        "## Etiology\n\nCauses vary."
    )
    # Nested subsections roll up into their parent section.
    assert document.section_count == 2
    assert document.metadata == {"accession": "NBK111"}


def test_scrape_chapter_raises_without_readable_content() -> None:
    """A chapter with no `itemprop="text"` body raises rather than yielding an empty document."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body><p>Not a chapter page.</p></body></html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        scrape_chapter(client, ChapterRef(accession="NBK999", title="Missing"))
        raise AssertionError("expected StatpearlsFetchError")
    except StatpearlsFetchError:
        pass


def test_scrape_chapter_by_url_extracts_title_and_content() -> None:
    """A chapter URL is resolved to its accession, title, and markdown content."""
    html = """
        <html><head><title>Bilateral Vocal Cord Paralysis - StatPearls - NCBI Bookshelf</title></head>
        <body>
          <div id="maincontent">
            <div class="body-content" itemprop="text"><p>Summary text.</p></div>
          </div>
        </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/books/NBK111/"
        return httpx.Response(200, text=html)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    document = scrape_chapter_by_url(client, "https://www.ncbi.nlm.nih.gov/books/NBK111/", link_mode=LinkMode.KEEP)

    assert document.title == "Bilateral Vocal Cord Paralysis"
    assert document.external_id == "statpearls-NBK111"
    assert document.content == "Summary text."


def test_scrape_chapter_by_url_rejects_a_non_bookshelf_url() -> None:
    """URLs without an NBK accession are rejected with an actionable message."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    try:
        scrape_chapter_by_url(client, "https://www.ncbi.nlm.nih.gov/pubmed/12345")
        raise AssertionError("expected StatpearlsFetchError")
    except StatpearlsFetchError as error:
        assert "NBK430685" in str(error)
