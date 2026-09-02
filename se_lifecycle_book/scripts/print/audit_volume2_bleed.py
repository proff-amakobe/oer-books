#!/usr/bin/env python3
"""Audit the locked Volume II Ingram bleed geometry.

Uses pdfinfo for per-page MediaBox, TrimBox, and BleedBox inspection. An
independent count can be supplied by pdfseparate in the calling QA workflow.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

EXPECTED_PAGES = 409
MEDIA = (0.0, 0.0, 621.0, 810.0)
ODD_TRIM = (0.0, 9.0, 612.0, 801.0)
EVEN_TRIM = (9.0, 9.0, 621.0, 801.0)
TOLERANCE = 0.02


def close_box(actual, expected) -> bool:
    return len(actual) == 4 and all(
        abs(float(a) - e) <= TOLERANCE for a, e in zip(actual, expected)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf",
        nargs="?",
        default="output/volume2/Software-Delivery-Operations-and-Evolution.pdf",
    )
    args = parser.parse_args()
    path = Path(args.pdf)

    info = subprocess.run(
        ["pdfinfo", str(path)], check=True, text=True, capture_output=True
    ).stdout
    pages_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", info, re.MULTILINE
    )
    failures: list[str] = []
    if not pages_match or int(pages_match.group(1)) != EXPECTED_PAGES:
        failures.append("pdfinfo page count is not 409")
    if not size_match or any(
        abs(float(v) - e) > TOLERANCE
        for v, e in zip(size_match.groups(), MEDIA[2:])
    ):
        failures.append("pdfinfo page size is not 621 x 810 pt")

    boxes = subprocess.run(
        ["pdfinfo", "-box", "-f", "1", "-l", str(EXPECTED_PAGES), str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    patterns = {
        "MediaBox": re.compile(r"^Page\s+(\d+) MediaBox:\s+(.+)$", re.MULTILINE),
        "TrimBox": re.compile(r"^Page\s+(\d+) TrimBox:\s+(.+)$", re.MULTILINE),
        "BleedBox": re.compile(r"^Page\s+(\d+) BleedBox:\s+(.+)$", re.MULTILINE),
    }
    found = {
        name: {
            int(page): tuple(float(value) for value in values.split())
            for page, values in pattern.findall(boxes)
        }
        for name, pattern in patterns.items()
    }
    for page_no in range(1, EXPECTED_PAGES + 1):
        expected_trim = ODD_TRIM if page_no % 2 else EVEN_TRIM
        for name, expected in (
            ("MediaBox", MEDIA),
            ("TrimBox", expected_trim),
            ("BleedBox", MEDIA),
        ):
            actual = found[name].get(page_no, ())
            if not close_box(actual, expected):
                failures.append(f"page {page_no}: {name} {actual}, expected {expected}")

    if failures:
        print("FAIL")
        for failure in failures[:30]:
            print(f"- {failure}")
        if len(failures) > 30:
            print(f"- ... {len(failures) - 30} additional failures")
        return 1
    print("PASS: 409 pages; MediaBox/BleedBox 621 x 810 pt; mirrored 612 x 792 pt TrimBox")
    print("PASS: odd outside bleed right; even outside bleed left; gutter bleed 0 pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
