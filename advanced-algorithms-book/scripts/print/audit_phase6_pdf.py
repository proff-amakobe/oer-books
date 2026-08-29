#!/usr/bin/env python3
"""Audit the Phase 6 print PDF and emit reviewable QA artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
parser.add_argument("--report", type=Path, default=Path("editorial/phase6-print-qa.json"))
parser.add_argument("--low", type=Path, default=Path("editorial/phase6-low-utilization-pages.csv"))
parser.add_argument("--blank", type=Path, default=Path("editorial/phase6-blank-pages.csv"))
args = parser.parse_args()

with tempfile.TemporaryDirectory() as tmp:
    bbox = Path(tmp) / "book.html"
    text_file = Path(tmp) / "book.txt"
    subprocess.run(["pdftotext", "-bbox-layout", str(args.pdf), str(bbox)], check=True)
    subprocess.run(["pdftotext", "-layout", str(args.pdf), str(text_file)], check=True)
    root = ET.parse(bbox).getroot()
    text = text_file.read_text(encoding="utf-8", errors="replace")

text_pages = text.split("\f")
page_rows, physical_overflow, text_area_overflow, stranded = [], [], [], []
for page_no, page in enumerate(root.findall(".//{*}page"), 1):
    width, height = float(page.attrib["width"]), float(page.attrib["height"])
    words = list(page.findall(".//{*}word"))
    boxes = [{k: float(w.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax")} |
             {"text": "".join(w.itertext())} for w in words]
    if boxes:
        ink_area = max(b["xMax"] for b in boxes) - min(b["xMin"] for b in boxes)
        ink_area *= max(b["yMax"] for b in boxes) - min(b["yMin"] for b in boxes)
        utilization = ink_area / (width * height)
    else:
        utilization = 0
    page_text = text_pages[page_no - 1] if page_no <= len(text_pages) else ""
    compact = " ".join(page_text.split())
    if page_no == 1: kind = "title"
    elif re.search(r"\bPART [IVX]+\b", compact): kind = "part opener"
    elif re.search(r"\bCHAPTER \d{2}\b", compact): kind = "chapter opener"
    elif "Table of contents" in compact: kind = "contents"
    elif len(words) == 0: kind = "blank"
    else: kind = "content"
    page_rows.append({"page": page_no, "words": len(words), "utilization": round(utilization, 4), "classification": kind})
    for box in boxes:
        if box["xMin"] < -.5 or box["yMin"] < -.5 or box["xMax"] > width + .5 or box["yMax"] > height + .5:
            physical_overflow.append({"page": page_no, **box})
        # Running heads and folios are deliberately outside the live body rectangle.
        if 48 < box["yMin"] < 735:
            left = 53.0 if page_no % 2 == 0 else 66.0
            right = 546.0 if page_no % 2 == 0 else 559.0
            # Allow 25 pt for optical punctuation and inline mathematical runs.
            if page_no > 1 and (box["xMin"] < left - 25 or box["xMax"] > right + 25):
                text_area_overflow.append({"page": page_no, **box})
    for line in page.findall(".//{*}line"):
        line_words = list(line.findall(".//{*}word"))
        if not line_words: continue
        value = " ".join("".join(w.itertext()) for w in line_words)
        y = min(float(w.attrib["yMin"]) for w in line_words)
        if y > 735 and re.match(r"^\d+(?:\.\d+)+\s+[A-Z][^.!?]{2,80}$", value):
            stranded.append({"page": page_no, "heading": value, "y": round(y, 2)})

intentional = {"title", "part opener", "chapter opener", "contents"}
low_rows = [row for row in page_rows if row["utilization"] < .12]
blank_rows = [row | {"intentional": row["classification"] in intentional}
              for row in page_rows if row["words"] <= 2]
for path, rows in ((args.low, low_rows), (args.blank, blank_rows)):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["page", "words", "utilization", "classification"])
        writer.writeheader(); writer.writerows(rows)

font_output = subprocess.run(["pdffonts", str(args.pdf)], check=True, text=True, capture_output=True).stdout
font_lines = font_output.splitlines()[2:]
type3 = sum("Type 3" in line for line in font_lines)
font_flags = [re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line) for line in font_lines]
not_embedded = sum(bool(match and match.group(1) == "no") for match in font_flags)
labels = {label: len(re.findall(re.escape(label), text)) for label in
          ("LEARNING OBJECTIVES", "THEOREM", "Proof", "INTUITION", "COMPLEXITY", "COMMON PITFALL")}
report = {
    "pdf": str(args.pdf), "pages": len(page_rows), "page_size_points": [612, 792],
    "physical_overflow_count": len(physical_overflow), "physical_overflow": physical_overflow,
    "text_area_overflow_count": len(text_area_overflow), "text_area_overflow": text_area_overflow,
    "blank_or_nearly_blank_pages": len(blank_rows), "unclassified_blank_pages": sum(not r["intentional"] for r in blank_rows),
    "low_utilization_pages": len(low_rows), "stranded_heading_candidates": stranded,
    "font_records": len(font_lines), "fonts_not_embedded": not_embedded, "type3_font_records": type3,
    "semantic_panel_label_counts": labels,
    "notes": "Low-use and blank-page CSVs are review queues. Part/title/opening pages are intentionally sparse. Type 3 records originate in legacy/vector figure artwork, not body typography.",
}
args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: report[k] for k in ("pages", "physical_overflow_count", "text_area_overflow_count", "blank_or_nearly_blank_pages", "unclassified_blank_pages", "low_utilization_pages", "fonts_not_embedded", "type3_font_records")}))
raise SystemExit(bool(physical_overflow or not_embedded or report["unclassified_blank_pages"] or stranded))
