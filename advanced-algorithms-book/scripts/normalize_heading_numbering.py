#!/usr/bin/env python3
"""Normalize chapter-owned numbering while leaving fenced content untouched."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIRST_NUMBERED_SECTION = {
    1: "What Is an Algorithm, Really?",
    2: "The Divide and Conquer Paradigm",
    3: "Heaps and Priority Queues",
    4: "The Greedy Choice Property",
    5: "The Problem with Naive Recursion",
    6: "Fundamentals of Randomized Algorithms",
    7: "Understanding Computational Complexity",
    8: "The Fundamentals of Approximation",
}


def normalize(path: Path) -> tuple[str, int]:
    chapter = int(path.name[:2])
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    fenced = False
    changes = 0
    chapter_title_seen = False
    numbered_sections_started = chapter not in FIRST_NUMBERED_SECTION

    for line_number, line in enumerate(lines, 1):
        if line.startswith("```"):
            fenced = not fenced
            output.append(line)
            continue
        if fenced:
            output.append(line)
            continue

        # Chapters 1, 2, and 5 contain a generic duplicate H1 immediately
        # before their real, manually numbered chapter title.
        if chapter in {1, 2, 5} and line_number == 1 and line.startswith(
            "# Advanced Algorithms: A Journey Through Computational Problem Solving"
        ):
            changes += 1
            continue

        match = re.match(r"^(#{1,6})\s+(.*?)(\r?\n)?$", line)
        if not match:
            output.append(line)
            continue

        marks, title, newline = match.group(1), match.group(2), match.group(3) or ""
        original = title

        chapter_prefix = re.match(
            rf"^Chapter\s+{chapter}(?::|\s+-|\s+—|\s+–|\s+)(.*)$", title, re.I
        )
        if chapter_prefix:
            title = chapter_prefix.group(1).strip()
            if title.startswith((':', '-', '—', '–')):
                title = title[1:].strip()
            if not chapter_title_seen:
                marks = "#"
                chapter_title_seen = True

        section_prefix = re.match(
            rf"^Section\s+{chapter}(?:\.\d+)+(?:\s*[:.—–-]\s*|\s+)(.*)$", title, re.I
        )
        if section_prefix:
            title = section_prefix.group(1).strip()

        numeric_prefix = re.match(
            rf"^{chapter}(?:\.\d+)+(?:\s*[:.—–-]\s*|\s+)(.*)$", title
        )
        if numeric_prefix:
            title = numeric_prefix.group(1).strip()

        if title == FIRST_NUMBERED_SECTION.get(chapter):
            numbered_sections_started = True
        elif len(marks) >= 2 and not numbered_sections_started and "{.unnumbered}" not in title:
            title = f"{title} {{.unnumbered}}"

        if title != original or marks != match.group(1):
            changes += 1
        output.append(f"{marks} {title}{newline}")

    return "".join(output), changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write normalized headings")
    args = parser.parse_args()
    total = 0
    for path in sorted((ROOT / "chapters").glob("*.qmd")):
        normalized, changes = normalize(path)
        total += changes
        if args.write and changes:
            path.write_text(normalized, encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {changes}")
    print(f"total: {total}")
    return 0 if args.write or total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
