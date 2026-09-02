#!/usr/bin/env python3
"""Apply deterministic, HTML-only publication metadata after Quarto renders."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "_book"
SITE = "https://proff-amakobe.github.io/oer-books/advanced-algorithms-book/"


def print_isbn() -> str:
    config = (ROOT / "_quarto.yml").read_text(encoding="utf-8")
    match = re.search(r'^\s*print-isbn:\s*["\']?([^"\'\n]+)', config, flags=re.M)
    if not match:
        raise RuntimeError("canonical print-isbn is missing")
    return match.group(1).strip()


def figure_alt_text() -> dict[str, str]:
    mapping: dict[str, str] = {}
    pattern = re.compile(r"!\[([^\]]+)\]\(([^)]+\.svg)\)")
    for source in (ROOT / "chapters").glob("*.qmd"):
        for alt, target in pattern.findall(source.read_text(encoding="utf-8")):
            mapping[Path(target).name] = alt
    return mapping


def canonical_for(path: Path) -> str:
    relative = path.relative_to(BOOK).as_posix()
    return SITE if relative == "index.html" else SITE + relative


def process(path: Path, alt_text: dict[str, str], isbn: str) -> bool:
    html = path.read_text(encoding="utf-8")
    if "<html" not in html or "</head>" not in html:
        return False
    canonical = canonical_for(path)
    tag = f'<link rel="canonical" href="{canonical}">'
    if 'rel="canonical"' in html:
        html = re.sub(r'<link\s+rel="canonical"[^>]*>', tag, html, count=1)
    else:
        html = html.replace("</head>", f"{tag}\n</head>", 1)
    isbn_meta = f'<meta name="book:print_isbn" content="{isbn}">'
    if 'name="book:print_isbn"' not in html:
        html = html.replace(
            '<meta name="book:publication_year" content="2026">',
            '<meta name="book:publication_year" content="2026">\n' + isbn_meta,
            1,
        )
    work_example = (
        '"workExample": {"@type": "Book", '
        '"bookFormat": "https://schema.org/Paperback", '
        f'"isbn": "{isbn}"}},'
    )
    if '"workExample"' not in html:
        html = html.replace(
            '"isAccessibleForFree": true',
            work_example + '\n  "isAccessibleForFree": true',
            1,
        )
    for filename, alt in alt_text.items():
        escaped = alt.replace("&", "&amp;").replace('"', "&quot;")
        html = re.sub(
            rf'(<img\s+src="[^"]*{re.escape(filename)}"\s+)(?![^>]*\balt=)',
            rf'\1alt="{escaped}" ', html,
        )
    # Web-only publication cleanup; preserve the canonical learner template in print/EPUB.
    html = html.replace("[Your Name]", "[Presenter Name]")
    path.write_text(html, encoding="utf-8")
    return True


def main() -> None:
    if not BOOK.exists():
        return
    alt_text = figure_alt_text()
    isbn = print_isbn()
    changed = sum(process(path, alt_text, isbn) for path in BOOK.rglob("*.html"))
    print(f"phase7_html_postprocess={changed}")


if __name__ == "__main__":
    main()
