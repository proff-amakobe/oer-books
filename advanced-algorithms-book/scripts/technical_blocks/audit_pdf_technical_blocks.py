#!/usr/bin/env python3
"""Machine-check PDF geometry, numbering, glyphs, and technical-page pathologies."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("pdf", type=Path)
parser.add_argument("--report", type=Path, default=Path("editorial/phase5-pdf-technical-qa.json"))
args = parser.parse_args()

with tempfile.TemporaryDirectory() as tmp:
    xml_path = Path(tmp) / "bbox.html"
    text_path = Path(tmp) / "book.txt"
    subprocess.run(["pdftotext", "-bbox-layout", str(args.pdf), str(xml_path)], check=True)
    subprocess.run(["pdftotext", "-layout", str(args.pdf), str(text_path)], check=True)
    root = ET.parse(xml_path).getroot()
    text = text_path.read_text(encoding="utf-8", errors="replace")

physical, wrong_boxes = [], []
page_word_counts = []
for page_no, page in enumerate(root.findall(".//{*}page"), 1):
    width, height = float(page.attrib["width"]), float(page.attrib["height"])
    if abs(width - 612) > .2 or abs(height - 792) > .2:
        wrong_boxes.append({"page": page_no, "width": width, "height": height})
    words = list(page.findall(".//{*}word")); page_word_counts.append(len(words))
    for word in words:
        box = {key: float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax")}
        if box["xMin"] < -.5 or box["yMin"] < -.5 or box["xMax"] > width + .5 or box["yMax"] > height + .5:
            physical.append({"page": page_no, "text": "".join(word.itertext()), **box})

pages = text.split("\f")
title_only = {"terminal": [], "code": []}
for page_no, page in enumerate(pages, 1):
    meaningful = [x.strip() for x in page.splitlines() if x.strip() and not re.fullmatch(r"\d+", x.strip())]
    joined = " ".join(meaningful)
    if re.fullmatch(r"(?:Terminal(?: -- continued)?\s*){1,2}", joined, re.I): title_only["terminal"].append(page_no)
    if re.fullmatch(r"(?:(?:Python|Java|JavaScript|C\+\+)(?: -- continued)?\s*){1,2}", joined, re.I): title_only["code"].append(page_no)

algorithm_numbers = re.findall(r"Algorithm\s+(\d+\.\d+)", text)
algorithm_counts = Counter(algorithm_numbers)
duplicates = sorted(number for number, count in algorithm_counts.items() if count > 1)
missing_glyph_tokens = len(re.findall(r"����|\[missing glyph\]|Missing character:", text, re.I))
low_density = [i + 1 for i, count in enumerate(page_word_counts) if count < 20]
technical_cues = re.compile(r"Algorithm\s+\d+\.\d+|Terminal(?: -- continued)?|^\s*(?:Python|Java|JavaScript|Output)\s*$", re.M)
low_density_technical = [i + 1 for i, page in enumerate(pages)
                         if i < len(page_word_counts) and page_word_counts[i] < 20 and technical_cues.search(page)]
report = {
    "pdf": str(args.pdf), "pages": len(page_word_counts), "wrong_page_boxes": wrong_boxes,
    "physical_overflow": physical, "terminal_title_only_pages": title_only["terminal"],
    "code_title_only_pages": title_only["code"], "numbered_algorithms": len(algorithm_numbers),
    "duplicate_algorithm_numbers": duplicates, "missing_glyph_tokens": missing_glyph_tokens,
    "low_density_pages_for_review": low_density, "low_density_technical_pages": low_density_technical,
    "notes": "Low-density pages include intentional front matter, part/chapter openers, and figure or section transitions; review is visual, not an automatic failure.",
}
args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
summary = {"pages": report["pages"], "wrong_page_boxes": len(wrong_boxes),
           "physical_overflow": len(physical), "terminal_title_only": len(title_only["terminal"]),
           "code_title_only": len(title_only["code"]), "numbered_algorithms": len(algorithm_numbers),
           "duplicate_algorithms": len(duplicates), "missing_glyph_tokens": missing_glyph_tokens,
           "low_density_review": len(low_density), "low_density_technical": len(low_density_technical)}
print(json.dumps(summary))
raise SystemExit(bool(wrong_boxes or physical or title_only["terminal"] or title_only["code"] or duplicates or missing_glyph_tokens or low_density_technical))
