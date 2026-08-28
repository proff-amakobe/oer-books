#!/usr/bin/env python3
"""Audit physical-page and instructional-figure vector bounds in a print PDF.

Requires PyMuPDF (`python -m pip install pymupdf`). Exits nonzero when an
instructional vector crosses the physical page or the print-safe text area.
Chapter-opening decorative backgrounds are reported separately and excluded.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pymupdf as fitz


PAGE_W = 8.5 * 72
PAGE_H = 11 * 72
# Geometry from print/preamble.tex plus 0.075 inch internal figure clearance.
SAFE_TOP = (0.70 + 0.075) * 72
SAFE_BOTTOM = PAGE_H - (0.81 + 0.075) * 72
SAFE_EVEN_LEFT = (0.73 + 0.075) * 72
SAFE_EVEN_RIGHT = PAGE_W - (1.00 + 0.10 + 0.075) * 72
SAFE_ODD_LEFT = (1.00 + 0.10 + 0.075) * 72
SAFE_ODD_RIGHT = PAGE_W - (0.73 + 0.075) * 72
CAPTION_RE = re.compile(r"^Figure\s+\d+(?:\.\d+)+:", re.I)
CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)", re.I)


def overflow(rect: fitz.Rect, bounds: fitz.Rect) -> float:
    return max(
        bounds.x0 - rect.x0,
        rect.x1 - bounds.x1,
        bounds.y0 - rect.y0,
        rect.y1 - bounds.y1,
        0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--report", type=Path, default=Path("editorial/figure-overflow-audit.md"))
    args = parser.parse_args()
    doc = fitz.open(args.pdf)
    physical = []
    instructional = []
    chapter = "Front matter"
    section = ""

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        page_rect = page.rect
        text = page.get_text("text")
        match = CHAPTER_RE.search(text)
        if match:
            chapter = f"Chapter {match.group(1)}"
        blocks = page.get_text("blocks")
        headings = [b[4].strip() for b in blocks if b[4].strip() and len(b[4].strip()) < 90]
        if headings:
            section = headings[0].splitlines()[0]
        is_opener = "CHAPTER" in text[:500] and page_no > 1
        drawings = page.get_drawings()
        for drawing in drawings:
            rect = fitz.Rect(drawing["rect"])
            amount = overflow(rect, page_rect)
            if amount > 0.01 and not is_opener:
                physical.append((page_no, chapter, section, rect, amount))

        captions = [fitz.Rect(b[:4]) for b in blocks if CAPTION_RE.match(b[4].strip())]
        if not captions:
            continue
        left = SAFE_ODD_LEFT if page_no % 2 else SAFE_EVEN_LEFT
        right = SAFE_ODD_RIGHT if page_no % 2 else SAFE_EVEN_RIGHT
        safe = fitz.Rect(left, SAFE_TOP, right, SAFE_BOTTOM)
        for cap in captions:
            # Quarto places captions directly below figures. Group nearby
            # substantive paths vertically and select the group nearest the
            # caption; this avoids treating an earlier terminal box, table, or
            # running rule as part of the figure.
            candidates = []
            for drawing in drawings:
                rect = fitz.Rect(drawing["rect"])
                if rect.y1 <= cap.y0 + 2 and rect.y1 >= cap.y0 - 0.72 * PAGE_H:
                    if rect.width > 2 and rect.height > 2:
                        candidates.append(rect)
            if not candidates:
                continue
            candidates.sort(key=lambda r: r.y1, reverse=True)
            bottom = candidates[0].y1
            top = candidates[0].y0
            cluster = [candidates[0]]
            for candidate in candidates[1:]:
                if top - candidate.y1 > 30:
                    break
                cluster.append(candidate)
                top = min(top, candidate.y0)
            union = fitz.Rect(cluster[0])
            for candidate in cluster[1:]:
                union.include_rect(candidate)
            amount = overflow(union, safe)
            if amount > 0.01:
                instructional.append((page_no, chapter, section, union, amount))

    lines = [
        "# Figure Overflow Audit",
        "",
        f"PDF: `{args.pdf}`",
        f"Pages: {doc.page_count}",
        "",
        "## Result",
        "",
        f"- Physical-page vector overflows (excluding chapter openers): **{len(physical)}**",
        f"- Instructional figure text-area violations: **{len(instructional)}**",
        "- Chapter-opening decorative backgrounds are intentionally excluded.",
        "",
        "## Findings",
        "",
        "| PDF page | Chapter | Section / figure context | Maximum overflow | Resolution |",
        "|---:|---|---|---:|---|",
    ]
    findings = physical + instructional
    if findings:
        for page_no, chap, sec, rect, amount in findings:
            lines.append(f"| {page_no} | {chap} | {sec or 'Figure'} | {amount / 72:.3f} in | Requires correction |")
    else:
        lines.append("| - | All | All instructional figures | 0.000 in | Pass - no correction required |")
    lines.extend([
        "",
        "## Method",
        "",
        "PyMuPDF drawing paths are checked against the 8.5 x 11 inch media box. Figure-adjacent vectors are also checked against the mirrored print text area with 0.075 inch internal clearance. The audit is intended to run after every print build.",
        "",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.report}: {len(physical)} physical, {len(instructional)} instructional")
    return 1 if physical or instructional else 0


if __name__ == "__main__":
    raise SystemExit(main())
