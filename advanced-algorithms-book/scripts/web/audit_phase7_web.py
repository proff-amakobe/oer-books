#!/usr/bin/env python3
"""Audit the rendered Second Edition HTML book."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "_book"
SITE = "https://proff-amakobe.github.io/oer-books/advanced-algorithms-book/"
CHAPTERS = [
    "01-introduction.html", "02-Divide-and-Conquer.html",
    "03-Data-Structures-for-Efficiency.html", "04-Greedy-Algorithms.html",
    "05-Dynamic-Programming.html", "06-Randomized-Algorithms.html",
    "07-Computational-Complexity.html", "08-Approximation-Algorithms.html",
    "09-Advanced-Graph-Algorithms.html", "10-String-Processing-Algorithms.html",
    "11-Numerical-Algorithms.html", "12-Advanced-Data-Structures.html",
    "13-Research-Industry-Applications.html", "14-Project-Development.html",
    "15-Final-Presentations.html",
]


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links, self.images, self.meta, self.scripts = [], [], [], []
        self.ids, self.visible = set(), []
        self.lang, self.title, self._script = "", "", None
        self._suppressed = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = {key: value or "" for key, value in attrs}
        if tag == "html": self.lang = attrs.get("lang", "")
        if tag in {"script", "style", "pre"}: self._suppressed += 1
        if tag == "script": self._script = [attrs, []]
        if tag == "title": self._in_title = True
        if tag == "a" and attrs.get("href"): self.links.append(attrs["href"])
        if attrs.get("id"): self.ids.add(attrs["id"])
        if tag == "img": self.images.append(attrs)
        if tag in {"meta", "link"}: self.meta.append(attrs)

    def handle_endtag(self, tag):
        if tag == "title": self._in_title = False
        if tag == "script":
            self.scripts.append((self._script or [{}, []])[0:1] + ["".join((self._script or [{}, []])[1])])
            self._script = None
        if tag in {"script", "style", "pre"}: self._suppressed = max(0, self._suppressed - 1)

    def handle_data(self, data):
        if self._script is not None: self._script[1].append(data)
        if self._in_title: self.title += data
        if not self._suppressed and data.strip(): self.visible.append(data.strip())


def load(path: Path) -> Page:
    page = Page(); page.feed(path.read_text(encoding="utf-8")); return page


def canonical(path: Path) -> str:
    relative = path.relative_to(BOOK).as_posix()
    return SITE if relative == "index.html" else SITE + relative


def audit() -> dict:
    files = sorted(BOOK.rglob("*.html"))
    pages = {path.resolve(): load(path) for path in files}
    errors, residue, external = [], Counter(), set()
    chapter_paths = [(BOOK / "chapters" / name).resolve() for name in CHAPTERS]

    numbering = []
    for expected, path in enumerate(chapter_paths, 1):
        if not path.exists():
            errors.append(f"missing chapter URL: {path.name}"); numbering.append(None); continue
        match = re.search(r'<h1 class="title"><span class="chapter-number">(\d+)</span>', path.read_text(encoding="utf-8"))
        found = int(match.group(1)) if match else None
        numbering.append(found)
        if found != expected: errors.append(f"chapter {path.name}: expected {expected}, found {found}")

    for path, page in pages.items():
        visible = " ".join(page.visible)
        patterns = {
            "first_edition": r"\bFirst Edition\b",
            "current_2025": r"First Edition.{0,20}2025|©\s*2025|Second Edition.{0,20}2025",
            "admin_placeholder": r"\[(?:Insert appropriate date|Specify:|Insert schedule|Insert link/platform|Your Name|Your Email)\]",
            "raw_source": r"```(?:python|bash)|\{\.class\}|date:\s*today",
        }
        for name, pattern in patterns.items():
            count = len(re.findall(pattern, visible, flags=re.I))
            residue[name] += count
            if count: errors.append(f"{name}: {path.relative_to(BOOK.resolve())} ({count})")

        if page.lang != "en-US": errors.append(f"language: {path.relative_to(BOOK.resolve())} = {page.lang!r}")
        tags = [item.get("href") for item in page.meta if item.get("rel") == "canonical"]
        if tags != [canonical(path)]: errors.append(f"canonical: {path.relative_to(BOOK.resolve())} = {tags}")
        if not any(item.get("property") == "og:title" for item in page.meta): errors.append(f"OpenGraph missing: {path.name}")
        try:
            objects = [json.loads(text) for attrs, text in page.scripts if attrs.get("type") == "application/ld+json"]
            book = next(item for item in objects if item.get("@type") == "Book")
            assert book.get("bookEdition") == "Second Edition" and str(book.get("datePublished")) == "2026"
            assert not any(key.lower() == "isbn" for key in book)
            paperback = book.get("workExample", {})
            assert paperback.get("bookFormat") == "https://schema.org/Paperback"
            assert paperback.get("isbn") == "979-8-1827-2111-0"
        except Exception as exc: errors.append(f"JSON-LD: {path.name}: {exc}")

    checked = broken = 0
    for path, page in pages.items():
        for href in page.links:
            parts = urlsplit(href)
            if parts.scheme in {"http", "https"}: external.add(href); continue
            if parts.scheme or href.startswith("#quarto-search"): continue
            checked += 1
            target = path if not parts.path else (path.parent / unquote(parts.path)).resolve()
            if target.is_dir(): target /= "index.html"
            if not target.exists():
                errors.append(f"broken link: {path.relative_to(BOOK.resolve())} -> {href}"); broken += 1; continue
            if parts.fragment and target.suffix == ".html":
                target_page = pages.get(target) or load(target)
                if unquote(parts.fragment) not in target_page.ids:
                    errors.append(f"broken fragment: {path.name} -> {href}"); broken += 1

    missing_alt = sum(
        1 for page in pages.values() for image in page.images
        if ".svg" in image.get("src", "") and not image.get("alt", "").strip()
    )
    if missing_alt: errors.append(f"SVG figures without alt: {missing_alt}")
    all_html = "\n".join(path.read_text(encoding="utf-8") for path in files)
    technical = len(re.findall(r'class="technical-block(?:\s|\")', all_html))
    figures = len(re.findall(r'<img[^>]+assets/figures/[^>]+\.svg', all_html))
    if figures != 14: errors.append(f"figures: {figures}/14")
    sitemap = BOOK / "sitemap.xml"; robots = BOOK / "robots.txt"
    if not sitemap.exists() or "localhost" in sitemap.read_text(encoding="utf-8"): errors.append("invalid sitemap")
    if not robots.exists() or "Allow: /" not in robots.read_text(encoding="utf-8"): errors.append("invalid robots.txt")

    search_terms = ["Master Theorem", "QuickSort", "Huffman", "NP-complete", "dynamic programming", "segment tree", "reproducibility"]
    search_items = json.loads((BOOK / "search.json").read_text(encoding="utf-8"))
    search_results = {}
    for term in search_terms:
        matches = [item["href"] for item in search_items if term.lower() in " ".join(str(item.get(key, "")) for key in ("title", "section", "text", "crumbs")).lower()]
        search_results[term] = len(matches)
        if not matches: errors.append(f"search term absent: {term}")

    return {
        "html_pages": len(files), "chapter_urls": f"{sum(path.exists() for path in chapter_paths)}/15",
        "chapter_numbering": numbering, "technical_blocks_before_math_audit": 512,
        "math_blocks_reclassified_net": 512 - technical,
        "technical_blocks_html": technical, "figures_svg": figures,
        "residue": dict(residue), "internal_links_checked": checked,
        "broken_internal": broken, "external_links_discovered": len(external),
        "search_results": search_results,
        "missing_svg_alt": missing_alt, "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--report", type=Path); args = parser.parse_args()
    result = audit()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "PASS")


if __name__ == "__main__": main()
