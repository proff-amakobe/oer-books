#!/usr/bin/env python3
"""Render every PDF page, build contact sheets, and measure meaningful body use.

The scanner deliberately ignores the running-head and folio bands. It measures
vertical occupancy by visible ink rows in the live body, so headers, page
numbers, rules, and empty listing borders do not make an empty page pass.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
parser.add_argument("--render-dir", type=Path, default=Path("tmp/pdfs/phase6b-pages"))
parser.add_argument("--contact-dir", type=Path, default=Path("output/print/phase6b-contact-sheets"))
parser.add_argument("--report", type=Path, default=Path("editorial/phase6b-low-utilization-pages.csv"))
parser.add_argument("--dpi", type=int, default=110)
parser.add_argument("--batch", type=int, default=25)
args = parser.parse_args()

args.render_dir.mkdir(parents=True, exist_ok=True)
args.contact_dir.mkdir(parents=True, exist_ok=True)
prefix = args.render_dir / "page"
if not list(args.render_dir.glob("page-*.png")):
    subprocess.run(["pdftoppm", "-png", "-r", str(args.dpi), str(args.pdf), str(prefix)], check=True)

text_path = args.render_dir / "book.txt"
subprocess.run(["pdftotext", "-layout", str(args.pdf), str(text_path)], check=True)
text_pages = text_path.read_text(encoding="utf-8", errors="replace").split("\f")
page_files = sorted(args.render_dir.glob("page-*.png"))

rows = []
thumbs = []
for page_no, path in enumerate(page_files, 1):
    image = Image.open(path).convert("L")
    scale = args.dpi / 72
    # Remove running heads, footer/folio, and the outer trim margins.
    crop = image.crop((round(48 * scale), round(66 * scale),
                       round((612 - 48) * scale), round(735 * scale)))
    width, height = crop.size
    pix = crop.load()
    dark_background_rows = 0
    occupied = []
    for y in range(height):
        dark = sum(1 for x in range(width) if pix[x, y] < 190)
        if dark > int(width * .70): dark_background_rows += 1
        # Ignore blank rows and long decorative/border rules.
        occupied.append(8 <= dark <= int(width * .70))
    expanded = occupied[:]
    radius = max(2, round(2.5 * scale))
    for y, value in enumerate(occupied):
        if value:
            for yy in range(max(0, y - radius), min(height, y + radius + 1)):
                expanded[yy] = True
    utilization = sum(expanded) / height
    compact = " ".join((text_pages[page_no - 1] if page_no <= len(text_pages) else "").split())
    next_compact = " ".join((text_pages[page_no] if page_no < len(text_pages) else "").split())
    if re.search(r"\bPART [IVX]+\b", compact):
        classification = "part opener"
    elif dark_background_rows > height * .18:
        utilization = max(utilization, dark_background_rows / height)
        classification = "full-page technical panel"
    elif page_no == 1: classification = "title page"
    elif page_no == 9: classification = "half-title"
    elif page_no == 10: classification = "edition notice"
    elif re.search(r"\bCHAPTER \d+\b", compact): classification = "chapter opener"
    elif "Copyright" in compact and len(compact) < 1200: classification = "copyright"
    elif "Dedication" in compact and len(compact) < 600: classification = "dedication"
    elif utilization < .20 and re.search(r"\b(?:PART [IVX]+|CHAPTER \d+)\b", next_compact):
        classification = "chapter/section closing page"
    else: classification = "content"
    severity = "CRITICAL REVIEW" if utilization < .10 else "REVIEW" if utilization < .20 else "PASS"
    if severity != "PASS":
        batch_start = ((page_no - 1) // args.batch) * args.batch + 1
        batch_end = min(batch_start + args.batch - 1, len(page_files))
        intentional = classification in {"title page", "half-title", "edition notice", "part opener", "copyright", "dedication", "chapter opener", "chapter/section closing page"}
        rows.append({"page": page_no, "meaningful_utilization": f"{utilization:.3f}",
                     "severity": severity, "classification": classification,
                     "contact_sheet": f"pages-{batch_start:03d}-{batch_end:03d}.png",
                     "reason": "deliberately sparse opener" if intentional else "low visible body occupancy",
                     "action": "retain" if intentional else "manual visual review",
                     "final_status": "INTENTIONAL" if intentional else "OPEN"})
    rgb = Image.open(path).convert("RGB")
    rgb.thumbnail((245, 317))
    tile = Image.new("RGB", (261, 345), "white")
    tile.paste(rgb, ((261 - rgb.width) // 2, 22))
    ImageDraw.Draw(tile).text((8, 5), f"PDF {page_no}", fill="black")
    thumbs.append(tile)

for start in range(0, len(thumbs), args.batch):
    group = thumbs[start:start + args.batch]
    cols = 5
    contact = Image.new("RGB", (cols * 261, math.ceil(len(group) / cols) * 345), (220, 220, 220))
    for index, thumb in enumerate(group):
        contact.paste(thumb, ((index % cols) * 261, (index // cols) * 345))
    first, last = start + 1, start + len(group)
    contact.save(args.contact_dir / f"pages-{first:03d}-{last:03d}.png")

with args.report.open("w", newline="", encoding="utf-8") as handle:
    fields = ["page", "meaningful_utilization", "severity", "classification", "contact_sheet",
              "reason", "action", "final_status"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)

open_rows = sum(row["final_status"] == "OPEN" for row in rows)
print(f"pages={len(page_files)} contact_sheets={math.ceil(len(page_files)/args.batch)} "
      f"flagged={len(rows)} open={open_rows}")
