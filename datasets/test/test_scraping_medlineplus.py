"""Tests for MedlinePlus scraping helpers."""

import logging

import httpx
import pytest
from typer.testing import CliRunner

from amfv_datasets.scraping.base import ScrapedDocument, ScrapeRun
from amfv_datasets.scraping.cli import SCRAPERS, app
from amfv_datasets.scraping.html import LinkMode
from amfv_datasets.scraping.medlineplus import (
    BASE_URL,
    MedlineplusFetchError,
    list_topic_urls,
    scrape_medlineplus,
    scrape_topic,
    topic_slug_from_url,
)

_TOPIC_HTML = """
<html>
  <head>
    <meta name="DC.Title" content="A1C" />
    <meta name="DC.Title.Alternate" content="Hemoglobin A1c" />
    <meta name="DC.Title.Alternate" content="HbA1c" />
    <meta name="DC.Subject.MeSH" content="Glycated Hemoglobin" />
    <meta name="DC.Date.Created" content="2015-12-22" />
    <meta name="DC.Date.Modified" content="2026-08-01" />
    <meta name="DC.Publisher" content="National Library of Medicine" />
  </head>
  <body>
    <div id="topic">
      <h1>A1C</h1>
      <div id="topic-summary" class="syndicate">
        <p>A1C tests for <a href="/diabetestype2.html">type 2 diabetes</a>.</p>
      </div>
    </div>
  </body>
</html>
"""

_TOPIC_HTML_ASTHMA = """
<html>
  <head><meta name="DC.Title" content="Asthma" /></head>
  <body>
    <div id="topic">
      <h1>Asthma</h1>
      <div id="topic-summary" class="syndicate"><p>Asthma affects the airways.</p></div>
    </div>
  </body>
</html>
"""

_NON_TOPIC_HTML = """
<html><head><meta name="DC.Title" content="Health Check Tools" /></head>
  <body><div id="topic"><h1>Health Check Tools</h1><p>An index page.</p></div></body>
</html>
"""

_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://medlineplus.gov/a1c.html</loc></url>
  <url><loc>https://medlineplus.gov/healthchecktools.html</loc></url>
  <url><loc>https://medlineplus.gov/a1c.html</loc></url>
  <url><loc>https://medlineplus.gov/spanish/a1c.html</loc></url>
  <url><loc>https://medlineplus.gov/ency/article/003640.htm</loc></url>
  <url><loc>https://medlineplus.gov/druginfo/herb_All.html</loc></url>
  <url><loc>https://medlineplus.gov/lab-tests/a1c-test/</loc></url>
</urlset>
"""


def test_list_topic_urls_keeps_only_flat_english_topic_pages() -> None:
    """Translations, nested corpora, and duplicate sitemap entries are filtered out."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sitemap.xml"
        return httpx.Response(200, content=_SITEMAP_XML.encode("utf-8"))

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)

    assert list_topic_urls(client) == [
        "https://medlineplus.gov/a1c.html",
        "https://medlineplus.gov/healthchecktools.html",
    ]


def test_list_topic_urls_raises_when_the_sitemap_has_no_topics() -> None:
    """An unexpected sitemap shape fails loudly rather than scraping nothing."""
    empty = b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'

    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=empty)))

    with pytest.raises(MedlineplusFetchError, match="No topic URLs"):
        list_topic_urls(client)


def test_scrape_topic_converts_summary_and_collects_metadata() -> None:
    """A topic page becomes a normalized document with markdown content."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/a1c.html"
        return httpx.Response(200, text=_TOPIC_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    document = scrape_topic(client, "https://medlineplus.gov/a1c.html")

    assert document is not None
    assert document.source == "medlineplus"
    assert document.external_id == "medlineplus-a1c"
    assert document.title == "A1C"
    assert document.url == "https://medlineplus.gov/a1c.html"
    assert document.content == "A1C tests for [type 2 diabetes](https://medlineplus.gov/diabetestype2.html)."
    assert document.metadata == {
        "attribution": "Courtesy of MedlinePlus from the National Library of Medicine",
        "also_called": ["Hemoglobin A1c", "HbA1c"],
        "date_created": "2015-12-22",
        "date_modified": "2026-08-01",
        "mesh_headings": ["Glycated Hemoglobin"],
        "publisher": "National Library of Medicine",
    }


def test_scrape_topic_strips_links_in_strip_mode() -> None:
    """Link stripping mode drops markdown link syntax from the summary."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=_TOPIC_HTML)))

    document = scrape_topic(client, "https://medlineplus.gov/a1c.html", link_mode=LinkMode.STRIP)

    assert document is not None
    assert document.content == "A1C tests for type 2 diabetes."


def test_scrape_topic_skips_pages_without_a_summary() -> None:
    """Index and tool pages that share the topic URL shape are skipped."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=_NON_TOPIC_HTML)))

    assert scrape_topic(client, "https://medlineplus.gov/healthchecktools.html") is None


def test_scrape_topic_skips_and_logs_a_page_that_fails_to_fetch(caplog: pytest.LogCaptureFixture) -> None:
    """One unreachable page is skipped rather than aborting the whole crawl."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))

    with caplog.at_level(logging.WARNING):
        document = scrape_topic(client, "https://medlineplus.gov/a1c.html")

    assert document is None
    assert "a1c.html" in caplog.text


def test_list_topic_urls_raises_medlineplus_fetch_error_when_the_sitemap_fails() -> None:
    """A sitemap fetch failure raises this module's error type, not a raw httpx error."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))

    with pytest.raises(MedlineplusFetchError, match="Could not fetch"):
        list_topic_urls(client)


@pytest.mark.parametrize(
    ("url", "expected_message"),
    [
        ("https://example.com/a1c.html", "from medlineplus.gov"),
        ("https://medlineplus.gov/spanish/a1c.html", "English topic URL"),
        ("https://medlineplus.gov/ency/article/003640.htm", "English topic URL"),
    ],
    ids=["wrong-host", "spanish-translation", "encyclopedia-article"],
)
def test_topic_slug_from_url_rejects_non_topic_urls(url: str, expected_message: str) -> None:
    """Non-topic URLs are rejected with a message naming the expected shape."""
    with pytest.raises(MedlineplusFetchError, match=expected_message):
        topic_slug_from_url(url)


def test_topic_slug_from_url_accepts_a_topic_url() -> None:
    """A topic URL resolves to its slug."""
    assert topic_slug_from_url("https://medlineplus.gov/a1c.html") == "a1c"


@pytest.mark.parametrize(
    "url",
    [
        "https://medlineplus.gov/ency/article/003640.htm",
        "https://medlineplus.gov/druginfo/meds/a693048.html",
    ],
    ids=["adam-encyclopedia", "ashp-drug-monograph"],
)
def test_third_party_licensed_corpora_are_never_scraped(url: str) -> None:
    """NLM licenses /ency/ and /druginfo/ from vendors, so they stay out of the corpus."""
    sitemap = (
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{url}</loc></url></urlset>"
    )
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=sitemap.encode("utf-8")))
    )

    # Excluded from discovery, and rejected outright when passed as --url.
    with pytest.raises(MedlineplusFetchError, match="No topic URLs"):
        list_topic_urls(client)
    with pytest.raises(MedlineplusFetchError, match="English topic URL"):
        topic_slug_from_url(url)


def test_scrape_medlineplus_scrapes_every_topic_and_fetches_the_sitemap_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SCRAPERS entry point discovers topics from the sitemap and scrapes each one."""
    sitemap = (
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<url><loc>https://medlineplus.gov/a1c.html</loc></url>"
        "<url><loc>https://medlineplus.gov/asthma.html</loc></url>"
        "</urlset>"
    )
    request_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_paths.append(request.url.path)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=sitemap.encode("utf-8"))
        if request.url.path == "/a1c.html":
            return httpx.Response(200, text=_TOPIC_HTML)
        if request.url.path == "/asthma.html":
            return httpx.Response(200, text=_TOPIC_HTML_ASTHMA)
        raise AssertionError(f"unexpected request to {request.url}")

    # scrape_medlineplus opens a fresh client per default_client() call (one for
    # discovery, one inside scrape_listing_documents), so this must be a factory
    # rather than a single shared client, which scrape_listing_documents closes
    # after its own use.
    monkeypatch.setattr(
        "amfv_datasets.scraping.medlineplus.default_client",
        lambda: httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL),
    )
    monkeypatch.setattr("amfv_datasets.scraping.base.time.sleep", lambda seconds: None)

    scrape_run = scrape_medlineplus(documents=None, link_mode=LinkMode.KEEP)
    documents = list(scrape_run.documents)

    assert scrape_run.total == 2
    assert {document.title for document in documents} == {"A1C", "Asthma"}
    assert request_paths.count("/sitemap.xml") == 1


def test_scrape_medlineplus_url_mode_normalizes_a_www_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A www.-prefixed --url still resolves to the canonical medlineplus.gov page."""
    request_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_paths.append(request.url.path)
        assert request.url.host == "medlineplus.gov"
        return httpx.Response(200, text=_TOPIC_HTML)

    monkeypatch.setattr(
        "amfv_datasets.scraping.medlineplus.default_client",
        lambda: httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL),
    )

    scrape_run = scrape_medlineplus(documents=None, url="https://www.medlineplus.gov/a1c.html")
    documents = list(scrape_run.documents)

    assert scrape_run.total == 1
    assert len(documents) == 1
    assert documents[0].url == "https://medlineplus.gov/a1c.html"
    assert request_paths == ["/a1c.html"]


def test_cli_dispatches_the_medlineplus_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """--source medlineplus resolves through the CLI's SCRAPERS registry to this module."""

    def fake_scrape_medlineplus(*, documents: int | None, link_mode: LinkMode, url: str | None = None) -> ScrapeRun:
        assert documents == 2
        assert url is None
        document = ScrapedDocument(
            source="medlineplus",
            external_id="medlineplus-a1c",
            title="A1C",
            url="https://medlineplus.gov/a1c.html",
            content="content",
        )
        return ScrapeRun([document], total=1)

    monkeypatch.setitem(SCRAPERS, "medlineplus", fake_scrape_medlineplus)

    result = CliRunner().invoke(app, ["--source", "medlineplus", "--documents", "2", "--no-progress"])

    assert result.exit_code == 0
    assert '"external_id": "medlineplus-a1c"' in result.stdout
