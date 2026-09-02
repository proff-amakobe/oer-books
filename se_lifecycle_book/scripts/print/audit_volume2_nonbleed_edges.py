#!/usr/bin/env python3
"""Flag non-white pixels within 0.10 inch of Volume II trim edges."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

PAGES = 409
WIDTH, HEIGHT = 612, 792
REVIEW_BAND = 7  # 0.0972 inch at the 72 dpi audit resolution


def read_ppm(path: Path):
    with path.open("rb") as handle:
        if handle.readline() != b"P6\n":
            raise ValueError(f"unexpected PPM format: {path}")
        dimensions = handle.readline()
        while dimensions.startswith(b"#"):
            dimensions = handle.readline()
        width, height = map(int, dimensions.split())
        if int(handle.readline()) != 255:
            raise ValueError(f"unexpected PPM depth: {path}")
        return width, height, handle.read()


def nonwhite(data: bytes, width: int, xs, ys) -> bool:
    for y in ys:
        for x in xs:
            start = (y * width + x) * 3
            if data[start : start + 3] != b"\xff\xff\xff":
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf",
        nargs="?",
        default="output/volume2/Software-Delivery-Operations-and-Evolution.pdf",
    )
    args = parser.parse_args()
    findings = []
    with tempfile.TemporaryDirectory(prefix="volume2-edge-audit-") as tmp:
        prefix = str(Path(tmp) / "page")
        subprocess.run(
            ["pdftoppm", "-r", "72", "-f", "1", "-l", str(PAGES), args.pdf, prefix],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for page_no in range(1, PAGES + 1):
            path = Path(f"{prefix}-{page_no:03d}.ppm")
            width, height, data = read_ppm(path)
            if (width, height) != (WIDTH, HEIGHT):
                raise SystemExit(f"page {page_no}: {width} x {height}, expected 612 x 792")
            edges = []
            if nonwhite(data, width, range(width), range(REVIEW_BAND)):
                edges.append("top")
            if nonwhite(data, width, range(width), range(height - REVIEW_BAND, height)):
                edges.append("bottom")
            outside_x = range(width - REVIEW_BAND, width) if page_no % 2 else range(REVIEW_BAND)
            if nonwhite(data, width, outside_x, range(height)):
                edges.append("outside")
            if edges:
                findings.append((page_no, "+".join(edges)))
    if findings:
        print("REVIEW: non-white artwork occurs within 0.10 inch of trim")
        for page_no, edges in findings:
            print(f"- page {page_no}: {edges}")
        return 1
    print("PASS: 409 pages; no non-white artwork within 0.10 inch of top, bottom, or outside trim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
