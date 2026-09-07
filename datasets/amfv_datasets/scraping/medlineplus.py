"""Scrape MedlinePlus health topic summaries into normalized markdown documents.

MedlinePlus (U.S. National Library of Medicine) publishes clinician-reviewed
health topic summaries. Topics are discovered from the sitemap that
MedlinePlus advertises in its own robots.txt, then each topic page is scraped
for its summary and Dublin Core metadata.

MedlinePlus also publishes a daily bulk XML export of every topic, which would
be one request instead of one per topic. We deliberately do not use it: it is
served from `/xml/`, which MedlinePlus robots.txt disallows. The topic pages
and the sitemap are both allowed, and the pages carry the same summary text
plus MeSH headings, alternate titles, and creation dates as metadata.

Summary content lives in a `topic-summary` container that NLM marks
`syndicate`, its convention for content offered for reuse. Pages without that
container (indexes, tools, directories) are not health topics and are skipped.

Licensing: NLM places health topic summaries in the public domain and allows
redistribution with the acknowledgement carried on every scraped document as
`ATTRIBUTION`, alongside `LICENSE`/`LICENSE_URL` so a corpus can be filtered
by source rights without parsing prose. Content NLM licenses from third
parties is deliberately out of scope:
A.D.A.M. encyclopedia articles (`/ency/`) and ASHP drug monographs
(`/druginfo/`) cannot be redistributed without licensing from those vendors,
and neither matches the flat topic path this scraper accepts. See
https://medlineplus.gov/about/using/usingcontent/.

Documents are short: live samples run 500-6,500 characters, median ~1,600,
since these are patient-facing summaries rather than clinical guidelines. For
comparison, the NICE scraper's guidelines run a median of 23,817 characters.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from urllib.parse import urlparse

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
from amfv_datasets.scraping.html import LinkMode, clean_text, html_to_markdown

BASE_URL = "https://medlineplus.gov"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
MEDLINEPLUS_DATASET_NAME = "medlineplus-webscrape"
MEDLINEPLUS_DATASET_DISPLAY_NAME = "MedlinePlus Webscrape"
ATTRIBUTION = "Courtesy of MedlinePlus from the National Library of Medicine"
"""Acknowledgement NLM asks redistributors of public domain content to carry."""
LICENSE = "Public Domain (U.S. Government work)"
LICENSE_URL = "https://medlineplus.gov/about/using/usingcontent/"
DOCUMENT_DELAY_SECONDS = 1.0
"""Delay between topic pages. MedlinePlus robots.txt sets no Crawl-delay, so
this is a politeness floor rather than a required interval."""

logger = logging.getLogger(__name__)

_SITEMAP_NS = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
# English topic pages are a single flat slug, e.g. /a1c.html. Anything nested
# (/spanish/..., /ency/..., /druginfo/...) is a translation or another corpus.
_TOPIC_PATH_RE = re.compile(r"^/(?P<slug>[a-z0-9]+)\.html$")


class MedlineplusFetchError(ScrapeError):
    """Raised when a MedlinePlus topic cannot be fetched or parsed."""


def topic_slug_from_url(url: str) -> str:
    """Return the topic slug for a MedlinePlus topic URL.

    Args:
        url: MedlinePlus topic URL, e.g. `https://medlineplus.gov/a1c.html`.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {
        "medlineplus.gov",
        "www.medlineplus.gov",
    }:
        raise MedlineplusFetchError(f"Enter a MedlinePlus topic URL from medlineplus.gov; got {url!r}")
    match = _TOPIC_PATH_RE.match(parsed.path)
    if not match:
        raise MedlineplusFetchError(f"Enter an English topic URL like {BASE_URL}/a1c.html; got {url!r}")
    return match.group("slug")


def list_topic_urls(client: httpx.Client) -> list[str]:
    """Return candidate English topic page URLs from the MedlinePlus sitemap.

    The sitemap also lists indexes, tools, and directory pages that share the
    topic URL shape; those are filtered out when scraped, since only real
    topics carry a summary container.

    Args:
        client: HTTP client used to fetch the sitemap.
    """
    try:
        response = client.get(SITEMAP_URL)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MedlineplusFetchError(f"Could not fetch the MedlinePlus sitemap at {SITEMAP_URL}") from exc
    try:
        root = etree.fromstring(response.content)
    except etree.XMLSyntaxError as exc:
        raise MedlineplusFetchError(f"Could not parse the MedlinePlus sitemap at {SITEMAP_URL}") from exc

    urls: list[str] = []
    seen: set[str] = set()
    for location in root.findall(".//sitemap:url/sitemap:loc", _SITEMAP_NS):
        url = (location.text or "").strip()
        if not url or url in seen:
            continue
        parsed = urlparse(url)
        if parsed.netloc.lower() != "medlineplus.gov" or not _TOPIC_PATH_RE.match(parsed.path):
            continue
        seen.add(url)
        urls.append(url)
    if not urls:
        raise MedlineplusFetchError(f"No topic URLs found in the MedlinePlus sitemap at {SITEMAP_URL}")
    return urls


def scrape_topic(client: httpx.Client, url: str, *, link_mode: LinkMode = LinkMode.KEEP) -> ScrapedDocument | None:
    """Scrape one MedlinePlus topic page into a normalized document.

    Returns None when the page carries no summary container, which is how
    non-topic pages in the sitemap are skipped, or when the page could not be
    fetched, which is logged rather than raised so one unreachable page in a
    large crawl costs a document instead of the whole run.

    Args:
        client: HTTP client used to fetch the topic page.
        url: MedlinePlus topic URL to scrape.
        link_mode: Whether links are kept as markdown links or stripped to
            their visible text (default: LinkMode.KEEP).
    """
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Skipping MedlinePlus topic %s: %s", url, exc)
        return None
    doc = lxml_html.fromstring(response.text)

    summaries = doc.xpath('//div[@id="topic-summary"]')
    if not summaries:
        return None
    content = html_to_markdown(lxml_html.tostring(summaries[0], encoding="unicode"), link_mode=link_mode, base_url=url)
    if not content:
        return None

    title = _meta_value(doc, "DC.Title") or _first_text(doc, '//div[@id="topic"]//h1//text()')
    if not title:
        return None

    return ScrapedDocument(
        source="medlineplus",
        external_id=f"medlineplus-{topic_slug_from_url(url)}",
        title=title,
        url=url,
        content=content,
        metadata={
            "attribution": ATTRIBUTION,
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "also_called": _meta_values(doc, "DC.Title.Alternate"),
            "date_created": _meta_value(doc, "DC.Date.Created"),
            "date_modified": _meta_value(doc, "DC.Date.Modified"),
            "mesh_headings": _meta_values(doc, "DC.Subject.MeSH"),
            "publisher": _meta_value(doc, "DC.Publisher"),
        },
    )


def scrape_medlineplus(
    *,
    documents: int | None,
    link_mode: LinkMode = LinkMode.KEEP,
    url: str | None = None,
) -> ScrapeRun:
    """Scrape MedlinePlus health topics discovered from the sitemap.

    Args:
        documents: Number of topics to scrape. Ignored when `url` is set. When
            unset, every topic in the sitemap is scraped (default: None).
        link_mode: Whether links are kept as markdown links or stripped to
            their visible text (default: LinkMode.KEEP).
        url: MedlinePlus topic URL to scrape as a single document (default:
            None).
    """
    if url is not None:
        topic_url = f"{BASE_URL}/{topic_slug_from_url(url)}.html"

        def scrape_url() -> Iterable[ScrapedDocument]:
            with default_client() as client:
                document = scrape_topic(client, topic_url, link_mode=link_mode)
            if document is None:
                raise MedlineplusFetchError(f"No topic summary found at '{topic_url}'")
            yield document

        return ScrapeRun(documents=scrape_url(), total=1)

    with default_client() as client:
        topic_urls = list_topic_urls(client)

    return ScrapeRun(
        total=documents if documents is not None else len(topic_urls),
        documents=scrape_listing_documents(
            documents=documents,
            client_factory=default_client,
            first_page_items=topic_urls,
            list_page=lambda client, page: [],
            scrape_item=lambda client, topic_url: scrape_topic(client, topic_url, link_mode=link_mode),
            document_delay_seconds=DOCUMENT_DELAY_SECONDS,
        ),
    )


def _meta_values(doc: lxml_html.HtmlElement, name: str) -> list[str]:
    values = [clean_text(value) for value in doc.xpath(f'//meta[@name="{name}"]/@content')]
    return [value for value in values if value]


def _meta_value(doc: lxml_html.HtmlElement, name: str) -> str:
    values = _meta_values(doc, name)
    return values[0] if values else ""


def _first_text(doc: lxml_html.HtmlElement, xpath: str) -> str:
    values = [clean_text(value) for value in doc.xpath(xpath)]
    return next((value for value in values if value), "")


__all__ = [
    "ATTRIBUTION",
    "BASE_URL",
    "DOCUMENT_DELAY_SECONDS",
    "LICENSE",
    "LICENSE_URL",
    "MEDLINEPLUS_DATASET_DISPLAY_NAME",
    "MEDLINEPLUS_DATASET_NAME",
    "SITEMAP_URL",
    "MedlineplusFetchError",
    "list_topic_urls",
    "scrape_medlineplus",
    "scrape_topic",
    "topic_slug_from_url",
]
