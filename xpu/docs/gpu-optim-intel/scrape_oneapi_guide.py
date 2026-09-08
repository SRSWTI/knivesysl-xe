#!/usr/bin/env python3
"""Recursively archive the text of an Intel oneAPI documentation guide.

The crawler starts at one guide page, follows every HTML link below that
version's URL prefix, extracts the article text with Trafilatura, and writes one
Markdown file per page plus an index and a machine-readable manifest.

Run from the repository root:
    uv run --with trafilatura --with requests \
      xpu/docs/gpu-optim-intel/scrape_oneapi_guide.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

try:
    import requests
    import trafilatura
    from lxml import html
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Run with: uv run --with trafilatura --with requests "
        f"{Path(__file__).name} ... ({exc})"
    ) from exc

DEFAULT_START_URL = (
    "https://www.intel.com/content/www/us/en/docs/oneapi/"
    "optimization-guide-gpu/2025-2/overview.html"
)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


@dataclass(frozen=True)
class PageRecord:
    title: str
    url: str
    file: str
    sha256: str
    discovered_links: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively scrape one Intel documentation guide to Markdown."
    )
    parser.add_argument("start_url", nargs="?", default=DEFAULT_START_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "archive",
        help="archive directory (default: %(default)s)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help="minimum delay between requests in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="optional safety limit; by default crawl the entire guide",
    )
    return parser.parse_args()


def canonicalize(url: str, base_url: str) -> str:
    """Resolve a link and remove fragments, queries, and redundant slashes."""
    absolute = urljoin(base_url, url.strip())
    parts = urlsplit(absolute)
    path = re.sub(r"/{2,}", "/", parts.path)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def guide_prefix(start_url: str) -> str:
    """Return the directory URL that bounds the recursive crawl."""
    clean = canonicalize(start_url, start_url)
    return clean.rsplit("/", 1)[0] + "/"


def is_guide_page(url: str, prefix: str) -> bool:
    return url.startswith(prefix) and url.endswith(".html")


def discover_links(document: str, page_url: str, prefix: str) -> list[str]:
    """Return unique in-guide article links in document order."""
    try:
        tree = html.fromstring(document)
    except (ValueError, html.etree.ParserError):
        return []

    found: list[str] = []
    seen: set[str] = set()
    for href in tree.xpath("//a[@href]/@href"):
        candidate = canonicalize(href, page_url)
        if is_guide_page(candidate, prefix) and candidate not in seen:
            seen.add(candidate)
            found.append(candidate)
    return found


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.5,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def fetch(session: requests.Session, url: str, timeout: float) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type:
        raise RuntimeError(f"expected HTML from {url}, received {content_type or 'unknown'}")
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def extract_markdown(document: str, url: str) -> tuple[str, str]:
    """Extract the main article as Markdown and return (title, body)."""
    metadata = trafilatura.extract_metadata(document, default_url=url)
    title = metadata.title.strip() if metadata and metadata.title else Path(url).stem
    body = trafilatura.extract(
        document,
        url=url,
        output_format="markdown",
        include_links=True,
        include_images=False,
        include_tables=True,
        include_formatting=True,
        favor_precision=True,
        deduplicate=True,
    )
    if not body or len(body.strip()) < 40:
        body = trafilatura.extract(
            document,
            url=url,
            output_format="markdown",
            include_links=True,
            include_images=False,
            include_tables=True,
            include_formatting=True,
            favor_recall=True,
            deduplicate=True,
        )
    if not body:
        raise RuntimeError(f"Trafilatura extracted no article text from {url}")
    return title, body.strip()


def page_filename(url: str) -> str:
    stem = Path(urlsplit(url).path).stem
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-") or "page"
    suffix = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"{safe}-{suffix}.md"


def write_page(path: Path, title: str, url: str, body: str) -> str:
    content = f"# {title}\n\nSource: {url}\n\n{body}\n"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()




def save_archive_metadata(
    output: Path,
    start_url: str,
    prefix: str,
    records: Iterable[PageRecord],
    failures: dict[str, str],
) -> None:
    ordered = list(records)
    manifest = {
        "start_url": start_url,
        "guide_prefix": prefix,
        "page_count": len(ordered),
        "failure_count": len(failures),
        "pages": [asdict(record) for record in ordered],
        "failures": failures,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = ["# Intel oneAPI GPU Optimization Guide archive", "", f"Source: {start_url}", ""]
    lines.extend(f"- [{record.title}](pages/{record.file})" for record in ordered)
    if failures:
        lines.extend(("", "## Failed pages", ""))
        lines.extend(f"- {url}: `{error}`" for url, error in sorted(failures.items()))
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def crawl(args: argparse.Namespace) -> int:
    start_url = canonicalize(args.start_url, args.start_url)
    prefix = guide_prefix(start_url)
    output: Path = args.output
    pages_dir = output / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    records_by_url: dict[str, PageRecord] = {}
    visited: set[str] = set()
    failures: dict[str, str] = {}
    pending = [start_url]
    queued = {start_url}
    session = make_session()
    last_request_at = 0.0

    while pending and (args.max_pages is None or len(visited) < args.max_pages):
        url = pending.pop()
        if url in visited:
            continue

        wait = args.delay - (time.monotonic() - last_request_at)
        if wait > 0:
            time.sleep(wait + random.uniform(0.0, min(0.25, args.delay / 3)))

        print(f"[{len(visited) + 1}] {url}", flush=True)
        try:
            document = fetch(session, url, args.timeout)
            last_request_at = time.monotonic()
            links = discover_links(document, url, prefix)
            title, body = extract_markdown(document, url)
            filename = page_filename(url)
            digest = write_page(pages_dir / filename, title, url, body)
            record = PageRecord(title, url, filename, digest, len(links))
            records_by_url[url] = record
            visited.add(url)

            # Reverse push preserves the page's link order in a depth-first crawl.
            for link in reversed(links):
                if link not in visited and link not in queued:
                    pending.append(link)
                    queued.add(link)
        except Exception as exc:  # Continue so one withdrawn article does not lose the guide.
            last_request_at = time.monotonic()
            failures[url] = f"{type(exc).__name__}: {exc}"
            visited.add(url)
            print(f"  failed: {failures[url]}", file=sys.stderr, flush=True)

        save_archive_metadata(
            output,
            start_url,
            prefix,
            records_by_url.values(),
            failures,
        )

    print(
        f"Archived {len(records_by_url)} pages to {output}; "
        f"{len(failures)} page(s) failed.",
        flush=True,
    )
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.delay < 0:
        raise SystemExit("--delay must be non-negative")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_pages is not None and args.max_pages <= 0:
        raise SystemExit("--max-pages must be positive")
    return crawl(args)


if __name__ == "__main__":
    raise SystemExit(main())
