"""Scrape StatPearls chapters into normalized markdown documents.

StatPearls (NCBI Bookshelf, U.S. National Library of Medicine) is a
continuously updated, clinician-reviewed medical encyclopedia. There is no
public listing of chapter accessions, so chapters are discovered through
NCBI's E-utilities search API: searching the `books` database for
`statpearls[book]` returns one hit per chapter *section* (Introduction,
Treatment, Review Questions, etc.), so we deduplicate by each section's
parent chapter accession to build the list of chapters to scrape. Chapter
content itself is scraped from its NCBI Bookshelf HTML page.

NCBI's robots.txt allows `/books/NBK*` with a 5 second crawl delay, which we
apply between chapter fetches. E-utilities usage policy (see
https://www.ncbi.nlm.nih.gov/books/NBK25497/) asks for at most 3 requests per
second without an API key and for callers to identify themselves via the `tool`
parameter, which we send as `EUTILS_TOOL`; discovery issues two sequential
requests per listing page, well inside that limit. Anyone running a full-corpus
scrape should additionally register a `tool`/`email` pair with NCBI and run it
outside US peak hours, as that policy requests for large jobs.

Licensing: unlike MedlinePlus, StatPearls chapters are not public domain.
NCBI's own copyright dialog on the book page states "Copyright (c) 2026,
StatPearls Publishing LLC," distributed under CC BY-NC-ND 4.0
(https://creativecommons.org/licenses/by-nc-nd/4.0/): NonCommercial, and
NoDerivatives. Every document carries `LICENSE` and `LICENSE_URL` in its
metadata so this travels downstream. The ND term is worth reading carefully
before this corpus feeds claim decomposition or training data, both of which
are derivative uses of the source text; that call belongs to whoever is
building those pipelines, not to this scraper.

Typical chapter is substantially larger than a MedlinePlus topic: live
samples run 27,500-65,400 characters across 16-20 top-level sections, versus
a MedlinePlus topic's ~1,600-character median.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from lxml import etree
from lxml import html as lxml_html

from amfv_datasets.scraping.base import (
    ScrapedDocument,
    ScrapeError,
    ScrapeRun,
    default_client,
    scrape_listing_documents,
)
from amfv_datasets.scraping.html import LinkMode, document_title, html_to_markdown

BOOKS_BASE_URL = "https://www.ncbi.nlm.nih.gov/books"
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
STATPEARLS_QUERY = "statpearls[book]"
SECTION_BATCH_SIZE = 200
DOCUMENT_DELAY_SECONDS = 5.0
EUTILS_TOOL = "amfv-datasets"
"""Identifies this client to NCBI, as E-utilities usage policy asks callers to do."""
STATPEARLS_DATASET_NAME = "statpearls-webscrape"
STATPEARLS_DATASET_DISPLAY_NAME = "StatPearls Webscrape"
LICENSE = "CC BY-NC-ND 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"

logger = logging.getLogger(__name__)

_ACCESSION_URL_RE = re.compile(r"/books/(NBK\d+)")
# NCBI Bookshelf pages are served as XHTML with a leading <?xml ...?> declaration,
# which lxml.html.fromstring rejects for str input ("Unicode strings with encoding
# declaration are not supported"), so it is stripped before parsing.
_XML_DECLARATION_RE = re.compile(r"^\s*<\?xml[^>]*\?>\s*")
# Top-level chapter sections are `article-<id>.s<n>`; nested subsections carry a
# further `.<n>` and are not counted separately.
_TOP_LEVEL_SECTION_RE = re.compile(r"article-\d+\.s\d+")


class StatpearlsFetchError(ScrapeError):
    """Raised when a StatPearls chapter cannot be discovered, fetched, or parsed."""


@dataclass(frozen=True)
class ChapterRef:
    """A StatPearls chapter discovered from the section search index."""

    accession: str
    title: str


def chapter_url(accession: str) -> str:
    """Return the NCBI Bookshelf URL for a chapter accession.

    Args:
        accession: Chapter accession, e.g. `NBK560852`.
    """
    return f"{BOOKS_BASE_URL}/{accession}/"


def search_section_uids(client: httpx.Client, *, retstart: int, retmax: int = SECTION_BATCH_SIZE) -> list[str]:
    """Return one page of StatPearls section UIDs from the books search index.

    Args:
        client: HTTP client used to query E-utilities.
        retstart: Offset into the search result set.
        retmax: Maximum number of section UIDs to return (default: SECTION_BATCH_SIZE).
    """
    try:
        response = client.get(
            f"{EUTILS_BASE_URL}/esearch.fcgi",
            params={
                "db": "books",
                "term": STATPEARLS_QUERY,
                "retstart": retstart,
                "retmax": retmax,
                "retmode": "json",
                "tool": EUTILS_TOOL,
            },
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise StatpearlsFetchError(f"Could not search StatPearls sections at retstart {retstart}") from exc
    return response.json()["esearchresult"]["idlist"]


def summarize_chapters(client: httpx.Client, section_uids: list[str]) -> list[ChapterRef]:
    """Resolve section UIDs to their parent chapter's accession and title.

    Args:
        client: HTTP client used to query E-utilities.
        section_uids: Section UIDs returned by `search_section_uids`.
    """
    if not section_uids:
        return []
    try:
        response = client.get(
            f"{EUTILS_BASE_URL}/esummary.fcgi",
            params={"db": "books", "id": ",".join(section_uids), "retmode": "json", "tool": EUTILS_TOOL},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise StatpearlsFetchError(f"Could not summarize {len(section_uids)} StatPearls sections") from exc
    result = response.json()["result"]
    refs: list[ChapterRef] = []
    for uid in result.get("uids", []):
        record = result[uid]
        accession = record.get("chapteraccessionid")
        if not accession:
            continue
        refs.append(ChapterRef(accession=accession, title=_chapter_title(record) or accession))
    return refs


def list_chapters(client: httpx.Client, page: int, *, seen: set[str]) -> list[ChapterRef]:
    """Return chapters newly discovered on one page of the section search index.

    Chapters already present in `seen` are skipped. Sections for a chapter are
    returned by NCBI clustered together, so a search page is only empty of new
    chapters when the underlying section search itself is exhausted.

    Args:
        client: HTTP client used to query E-utilities.
        page: 1-indexed search page; each page covers `SECTION_BATCH_SIZE` sections.
        seen: Chapter accessions already yielded on earlier pages; updated in place.
    """
    section_uids = search_section_uids(client, retstart=(page - 1) * SECTION_BATCH_SIZE)
    new_refs = []
    for ref in summarize_chapters(client, section_uids):
        if ref.accession in seen:
            continue
        seen.add(ref.accession)
        new_refs.append(ref)
    return new_refs


def scrape_chapter(
    client: httpx.Client, ref: ChapterRef, *, link_mode: LinkMode = LinkMode.KEEP
) -> ScrapedDocument | None:
    """Scrape one StatPearls chapter into a normalized document.

    Returns None when the chapter cannot be fetched or has no readable
    content, which is logged rather than raised so one bad chapter in a
    multi-hour crawl costs a document instead of the whole run.

    Args:
        client: HTTP client used to fetch the chapter page.
        ref: Chapter accession and title to scrape.
        link_mode: Whether links are kept as markdown links or stripped to
            their visible text (default: LinkMode.KEEP).
    """
    try:
        response = client.get(chapter_url(ref.accession))
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Skipping StatPearls chapter %s: %s", ref.accession, exc)
        return None
    content, section_count = _chapter_content(response.text, link_mode=link_mode)
    if not content:
        logger.warning("Skipping StatPearls chapter %s: no readable content", ref.accession)
        return None
    return ScrapedDocument(
        source="statpearls",
        external_id=f"statpearls-{ref.accession}",
        title=ref.title,
        url=chapter_url(ref.accession),
        content=content,
        section_count=section_count,
        metadata={"accession": ref.accession, "license": LICENSE, "license_url": LICENSE_URL},
    )


def scrape_chapter_by_url(client: httpx.Client, url: str, *, link_mode: LinkMode = LinkMode.KEEP) -> ScrapedDocument:
    """Scrape a StatPearls chapter from its NCBI Bookshelf URL.

    Unlike `scrape_chapter`, this raises on failure rather than returning
    None: a single `--url` request has no other document to fall back on.

    Args:
        client: HTTP client used to fetch the chapter page.
        url: NCBI Bookshelf chapter URL, e.g. `https://www.ncbi.nlm.nih.gov/books/NBK560852/`.
        link_mode: Whether links are kept as markdown links or stripped to
            their visible text (default: LinkMode.KEEP).
    """
    match = _ACCESSION_URL_RE.search(url)
    if not match:
        raise StatpearlsFetchError(f"Enter a StatPearls chapter URL like {BOOKS_BASE_URL}/NBK430685/; got {url!r}")
    accession = match.group(1)

    try:
        response = client.get(chapter_url(accession))
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise StatpearlsFetchError(f"Could not fetch StatPearls chapter '{accession}'") from exc
    content, section_count = _chapter_content(response.text, link_mode=link_mode)
    if not content:
        raise StatpearlsFetchError(f"No readable content for StatPearls chapter '{accession}'")
    title = document_title(
        _XML_DECLARATION_RE.sub("", response.text),
        fallback=accession,
        suffixes=(" - StatPearls - NCBI Bookshelf",),
    )
    return ScrapedDocument(
        source="statpearls",
        external_id=f"statpearls-{accession}",
        title=title,
        url=chapter_url(accession),
        content=content,
        section_count=section_count,
        metadata={"accession": accession, "license": LICENSE, "license_url": LICENSE_URL},
    )


def scrape_statpearls(
    *,
    documents: int | None,
    link_mode: LinkMode = LinkMode.KEEP,
    url: str | None = None,
) -> ScrapeRun:
    """Scrape StatPearls chapters discovered through the NCBI books search index.

    Args:
        documents: Number of chapters to scrape. Ignored when `url` is set.
            When unset, chapters are discovered and scraped until the search
            index is exhausted (default: None).
        link_mode: Whether links are kept as markdown links or stripped to
            their visible text (default: LinkMode.KEEP).
        url: NCBI Bookshelf chapter URL to scrape as a single document
            (default: None).
    """
    if url is not None:

        def scrape_url() -> Iterable[ScrapedDocument]:
            with default_client() as client:
                yield scrape_chapter_by_url(client, url, link_mode=link_mode)

        return ScrapeRun(documents=scrape_url(), total=1)

    seen: set[str] = set()
    return ScrapeRun(
        # Unlike MedlinePlus's sitemap, chapter discovery here is itself
        # incremental (one search page at a time), so there is no cheap
        # upper bound to report before a full run other than `documents`.
        total=documents,
        documents=scrape_listing_documents(
            documents=documents,
            client_factory=default_client,
            list_page=lambda client, page: list_chapters(client, page, seen=seen),
            scrape_item=lambda client, ref: scrape_chapter(client, ref, link_mode=link_mode),
            document_delay_seconds=DOCUMENT_DELAY_SECONDS,
        ),
    )


def _chapter_title(record: dict[str, Any]) -> str:
    bookinfo = record.get("bookinfo")
    if not bookinfo:
        return ""
    try:
        info = etree.fromstring(bookinfo.encode("utf-8"))
    except etree.XMLSyntaxError:
        return ""
    titles = info.xpath(".//Parent[@type='chapter']/Title/text()")
    return titles[0].strip() if titles else ""


def _chapter_content(html_text: str, *, link_mode: LinkMode) -> tuple[str, int]:
    """Return a chapter body as markdown and its top-level section count."""
    doc = lxml_html.fromstring(_XML_DECLARATION_RE.sub("", html_text))
    bodies = doc.xpath('//div[@itemprop="text"]')
    if not bodies:
        return "", 0
    body = bodies[0]
    sections = [child for child in body.xpath("./div[@id]") if _TOP_LEVEL_SECTION_RE.fullmatch(child.get("id") or "")]
    body_html = lxml_html.tostring(body, encoding="unicode")
    markdown = html_to_markdown(body_html, link_mode=link_mode, base_url=BOOKS_BASE_URL)
    return markdown, max(len(sections), 1)


__all__ = [
    "BOOKS_BASE_URL",
    "DOCUMENT_DELAY_SECONDS",
    "EUTILS_BASE_URL",
    "EUTILS_TOOL",
    "LICENSE",
    "LICENSE_URL",
    "SECTION_BATCH_SIZE",
    "STATPEARLS_DATASET_DISPLAY_NAME",
    "STATPEARLS_DATASET_NAME",
    "STATPEARLS_QUERY",
    "ChapterRef",
    "StatpearlsFetchError",
    "chapter_url",
    "list_chapters",
    "scrape_chapter",
    "scrape_chapter_by_url",
    "scrape_statpearls",
    "search_section_uids",
    "summarize_chapters",
]
