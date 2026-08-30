#!/usr/bin/env python3
"""Render every PDF page, detect suspicious dark bars, and build math contact sheets."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


ap = argparse.ArgumentParser()
ap.add_argument("pdf", type=Path)
ap.add_argument("--render-dir", type=Path, default=Path("tmp/pdfs/phase6c-pages"))
ap.add_argument("--output-dir", type=Path, default=Path("editorial/qa/math"))
ap.add_argument("--dpi", type=int, default=92)
args = ap.parse_args()
args.render_dir.mkdir(parents=True, exist_ok=True)
args.output_dir.mkdir(parents=True, exist_ok=True)

prefix = args.render_dir / "page"
pages = sorted(args.render_dir.glob("page-*.png"))
if not pages:
    subprocess.run(["pdftoppm", "-png", "-r", str(args.dpi), str(args.pdf), str(prefix)], check=True)
    pages = sorted(args.render_dir.glob("page-*.png"))
text_file = args.render_dir / "book.txt"
subprocess.run(["pdftotext", "-layout", str(args.pdf), str(text_file)], check=True)
texts = text_file.read_text(encoding="utf-8", errors="replace").split("\f")

math_rx = re.compile(r"(?:\b[OTFPQWEHXC]\s*\(|[=≤≥≠≈∈∉⊂⊆∪∩∞√ΣΠ∂∇ΘΩαβεδλμπφ]|\blog\b|\bmod\b|\brecurrence\b|\bprobability\b|\btheorem\b)", re.I)
categories = {
    "asymptotics": re.compile(r"Big-[OΩΘ]|asymptot|O\s*\(|Omega|Theta", re.I),
    "recurrences-master-theorem": re.compile(r"recurrence|Master Theorem|T\s*\(\s*n", re.I),
    "dynamic-programming": re.compile(r"Dynamic Programming|dp\[|recurrence|subproblem", re.I),
    "randomized-probability": re.compile(r"random|probability|expect|Chernoff|Markov", re.I),
    "complexity-theory": re.compile(r"NP-complete|NP-hard|PSPACE|polynomial reduction|SAT", re.I),
    "approximation": re.compile(r"approximation|OPT|ratio", re.I),
    "flow-network": re.compile(r"flow|capacity|residual|min-cut|max-flow", re.I),
    "fft-numerical": re.compile(r"FFT|Fourier|convolution|matrix|numerical", re.I),
    "advanced-data-structures": re.compile(r"segment tree|Fenwick|Ackermann|amortized|heap", re.I),
}


def tile(path: Path, page_no: int, width: int = 180) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((width, round(width * 1.295)))
    out = Image.new("RGB", (width + 10, im.height + 25), "white")
    out.paste(im, ((out.width - im.width) // 2, 20))
    ImageDraw.Draw(out).text((5, 4), f"PDF {page_no}", fill="black")
    return out


def sheet(selected: list[int], destination: Path, cols: int = 6) -> None:
    if not selected:
        Image.new("RGB", (600, 80), "white").save(destination)
        return
    tiles = [tile(pages[n - 1], n) for n in selected]
    w = max(t.width for t in tiles); h = max(t.height for t in tiles)
    canvas = Image.new("RGB", (cols * w, math.ceil(len(tiles) / cols) * h), (220, 220, 220))
    for i, t in enumerate(tiles): canvas.paste(t, ((i % cols) * w, (i // cols) * h))
    canvas.save(destination, quality=88)


math_pages, category_pages, contrast_rows = [], {k: [] for k in categories}, []
for page_no, path in enumerate(pages, 1):
    text = texts[page_no - 1] if page_no <= len(texts) else ""
    score = len(math_rx.findall(text))
    if score >= 4:
        math_pages.append(page_no)
    for name, rx in categories.items():
        if rx.search(text): category_pages[name].append(page_no)

    im = Image.open(path).convert("L")
    scale = args.dpi / 72
    crop = im.crop((round(45 * scale), round(70 * scale), round((612 - 45) * scale), round(730 * scale)))
    width, height = crop.size; pix = crop.load()
    dark_rows = [sum(pix[x, y] < 35 for x in range(width)) / width >= .72 for y in range(height)]
    runs, start = [], None
    for y, dark in enumerate(dark_rows + [False]):
        if dark and start is None: start = y
        elif not dark and start is not None:
            runs.append((start, y - 1)); start = None
    suspicious = [(a, b) for a, b in runs if max(4, round(3 * scale)) <= b - a + 1 <= round(24 * scale)]
    if suspicious:
        contrast_rows.append({"page": page_no, "dark_row_runs": ";".join(f"{a}-{b}" for a,b in suspicious), "status": "REVIEW"})

sheet(math_pages, args.output_dir / "math-pages-contact-sheet.jpg")
for name, selected in category_pages.items():
    sheet(selected, args.output_dir / f"{name}-contact-sheet.jpg")
with (args.output_dir / "dark-contrast-candidates.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["page", "dark_row_runs", "status"])
    writer.writeheader(); writer.writerows(contrast_rows)
with (args.output_dir / "math-page-inventory.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh); writer.writerow(["page", "math_score", "status"])
    for n in math_pages: writer.writerow([n, len(math_rx.findall(texts[n-1])), "REVIEWED_BY_CONTACT_SHEET"])
print(f"pages={len(pages)} math_pages={len(math_pages)} contrast_candidates={len(contrast_rows)} targeted_sheets={len(categories)}")
