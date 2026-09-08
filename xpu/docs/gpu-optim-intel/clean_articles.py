#!/usr/bin/env python3
"""Strip Intel navigation/footer boilerplate from archived Markdown pages.

The archive reader output has this stable shape:
  ...site metadata and navigation...
  ## Actual article title
  ...article...
  [Level Two Title](#)
  ...site footer...

This script keeps only the title, source URL, and text between those boundaries.
Failed-download placeholder pages are removed because they contain no article.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "archive"
ARTICLE_RE_TEMPLATE = r"(?ms)^##\s+{title}\s*$\n(?P<body>.*?)(?=^\[Level Two Title\]\(#\)\s*$|\Z)"
IMAGE_RE = re.compile(r"(?m)^\s*!\[[^\]]*\]\([^\n]*\)\s*$\n?")
EXCESS_BLANKS_RE = re.compile(r"\n{3,}")
NAVIGATION_ONLY_TITLES = {"oneAPI GPU Optimization Guide"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only actual article content in an Intel guide archive."
    )
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="archive directory (default: %(default)s)",
    )
    return parser.parse_args()


def extract_article(raw: str, title: str, source: str) -> str:
    cleaned_prefix = f"# {title}\n\nSource: {source}\n\n"
    if raw.startswith(cleaned_prefix):
        return raw

    pattern = re.compile(ARTICLE_RE_TEMPLATE.format(title=re.escape(title)))
    matches = list(pattern.finditer(raw))
    if not matches:
        raise ValueError(f"article heading not found: {title}")

    # The first matching heading may occur in Intel's table of contents. The
    # actual article is the final exact heading before the footer marker.
    body = matches[-1].group("body").strip()
    body = IMAGE_RE.sub("", body)
    body = EXCESS_BLANKS_RE.sub("\n\n", body).strip()
    if not body:
        raise ValueError(f"article body is empty: {title}")
    return f"# {title}\n\nSource: {source}\n\n{body}\n"


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    kept: list[dict[str, object]] = []
    removed: list[str] = []
    for page in manifest["pages"]:
        path = archive / page["file"]
        if page["title"] in NAVIGATION_ONLY_TITLES:
            path.unlink(missing_ok=True)
            removed.append(page["url"])
            continue

        if not page.get("complete", True):
            path.unlink(missing_ok=True)
            removed.append(page["url"])
            continue

        cleaned = extract_article(
            path.read_text(encoding="utf-8"), page["title"], page["url"]
        )
        path.write_text(cleaned, encoding="utf-8")
        page["sha256"] = hashlib.sha256(cleaned.encode()).hexdigest()
        kept.append(page)

    manifest["pages"] = kept
    manifest["page_count"] = len(kept)
    manifest["complete_page_count"] = len(kept)
    manifest["removed_incomplete_pages"] = sorted(
        set(manifest.get("removed_incomplete_pages", [])) | set(removed)
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    index = [
        "# Intel oneAPI GPU Optimization Guide 2025.2",
        "",
        f"Source: {manifest['source']}",
        "",
        f"Archived articles: {len(kept)}",
        "",
    ]
    index.extend(f"- [{page['title']}]({page['file']})" for page in kept)
    (archive / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    print(f"Kept {len(kept)} articles; removed {len(removed)} non-article pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
