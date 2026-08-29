#!/usr/bin/env python3
"""Verify that rendered HTML chapters are numbered consecutively 1 through 15."""

from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "_book" / "chapters"
files = sorted(BOOK.glob("*.html"))
expected = list(range(1, 16))
actual: list[int] = []

if len(files) != 15:
    print(f"FAIL: expected 15 chapter HTML files, found {len(files)}")
    raise SystemExit(1)

for path in files:
    text = path.read_text(encoding="utf-8")
    title_block = re.search(r'<h1 class="title">(.*?)</h1>', text, re.S)
    if not title_block:
        print(f"FAIL: no title block in {path.relative_to(ROOT)}")
        raise SystemExit(1)
    number = re.search(r'<span class="chapter-number">(\d+)</span>', title_block.group(1))
    if not number:
        print(f"FAIL: no chapter number in {path.relative_to(ROOT)}")
        raise SystemExit(1)
    actual.append(int(number.group(1)))

if actual != expected:
    print(f"FAIL: expected {expected}, found {actual}")
    raise SystemExit(1)

print(f"PASS: rendered chapter sequence is {actual[0]} through {actual[-1]}")

epub = ROOT / "_book" / "Advanced-Computational-Algorithms.epub"
if epub.exists():
    epub_numbers: list[int] = []
    with zipfile.ZipFile(epub) as archive:
        for name in sorted(archive.namelist()):
            if not re.match(r"EPUB/text/ch\d+\.xhtml$", name):
                continue
            page = archive.read(name).decode("utf-8", errors="replace")
            heading = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
            if not heading:
                continue
            number = re.search(r'class="chapter-number"[^>]*>(\d+)</span>', heading.group(1))
            if number:
                epub_numbers.append(int(number.group(1)))
    if epub_numbers != expected:
        print(f"FAIL: EPUB expected {expected}, found {epub_numbers}")
        raise SystemExit(1)
    print(f"PASS: EPUB chapter sequence is {epub_numbers[0]} through {epub_numbers[-1]}")
