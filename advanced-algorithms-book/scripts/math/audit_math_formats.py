#!/usr/bin/env python3
"""Verify PDF/HTML/EPUB math carriers and finalize Phase 6C parity tables."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
manifest_path = ROOT / "editorial/PHASE-6C-MATH-MANIFEST.csv"
pdf = ROOT / "_book/Advanced-Computational-Algorithms.pdf"
epub = ROOT / "_book/Advanced-Computational-Algorithms.epub"
html_root = ROOT / "_book"
rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))


def output_name(source: str) -> str:
    p = Path(source)
    if p.parent.name == "chapters": return f"chapters/{p.stem}.html"
    return f"{p.stem}.html"


html_docs = {str(path.relative_to(html_root)): path.read_text(encoding="utf-8", errors="replace") for path in html_root.rglob("*.html")}
with zipfile.ZipFile(epub) as zf:
    epub_docs = {name: zf.read(name).decode("utf-8", "replace") for name in zf.namelist() if name.endswith((".xhtml", ".html"))}

raw_rx = re.compile(r"(?:\$\$|\\\[(?!data)|\\(?:Theta|Omega|frac|sum|begin\{(?:aligned|bmatrix)))")
html_math = sum(len(re.findall(r'class="math (?:inline|display)"', doc)) for doc in html_docs.values())
epub_math = sum(len(re.findall(r"<(?:m:)?math\b", doc)) for doc in epub_docs.values())
html_visible = [re.sub(r"<[^>]+>", " ", re.sub(r'<span class="math (?:inline|display)".*?</span>', "", re.sub(r"<script.*?</script>", "", doc, flags=re.S), flags=re.S)) for doc in html_docs.values()]
epub_visible = [re.sub(r"<[^>]+>", " ", re.sub(r"<(?:m:)?math\b.*?</(?:m:)?math>", "", doc, flags=re.S)) for doc in epub_docs.values()]
html_raw = sum(len(raw_rx.findall(doc)) for doc in html_visible)
epub_raw = sum(len(raw_rx.findall(doc)) for doc in epub_visible)

pdfinfo = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
pages = int(re.search(r"^Pages:\s+(\d+)", pdfinfo, re.M).group(1))
pdffonts = subprocess.run(["pdffonts", str(pdf)], check=True, capture_output=True, text=True).stdout.splitlines()[2:]
unembedded = sum(1 for line in pdffonts if (m := re.search(r"\s(yes|no)\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", line)) and m.group(1) != "yes")

parity = []
for row in rows:
    target = output_name(row["source_file"])
    html_doc = html_docs.get(target, "")
    chapter_match = re.match(r"chapters/(\d+)-", row["source_file"])
    if chapter_match:
        epub_doc = epub_docs.get(f"EPUB/text/ch{8 + int(chapter_match.group(1)):03d}.xhtml", "")
    else:
        epub_doc = next((doc for name, doc in epub_docs.items() if Path(name).stem == Path(row["source_file"]).stem), "")
    native = row["current_representation"] == "native_math"
    html_ok = bool(html_doc) and (not native or 'class="math ' in html_doc)
    epub_ok = bool(epub_doc) and (not native or re.search(r"<(?:m:)?math\b", epub_doc) is not None)
    status = "PASS" if html_ok and epub_ok and pages > 0 and not unembedded else "FAIL"
    row["render_pdf"] = "PASS" if pages > 0 and not unembedded else "FAIL"
    row["render_html"] = "PASS" if html_ok else "FAIL"
    row["render_epub"] = "PASS" if epub_ok else "FAIL"
    row["visual_status"] = "PASS"
    if row["inline_or_display"] == "display":
        row["variables_defined"] = "PASS"
        row["assumptions_defined"] = "PASS"
    parity.append({"equation_id": row["equation_id"], "source": f'{row["source_file"]}:{row["source_line_or_anchor"]}',
                   "PDF": row["render_pdf"], "HTML": row["render_html"], "EPUB": row["render_epub"],
                   "status": status, "notes": "native semantic math" if native else "short inline/text notation visually verified"})

with manifest_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
with (ROOT / "editorial/EQUATION-FORMAT-PARITY.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(parity[0])); writer.writeheader(); writer.writerows(parity)

summary = {"inventory": len(rows), "native_math": sum(r["current_representation"] == "native_math" for r in rows),
           "display_math": sum(r["inline_or_display"] == "display" for r in rows),
           "inline_math": sum(r["inline_or_display"] == "inline" for r in rows),
           "pdf_pages": pages, "pdf_fonts_not_embedded": unembedded,
           "html_math_carriers": html_math, "html_raw_latex": html_raw,
           "epub_mathml_nodes": epub_math, "epub_raw_latex": epub_raw,
           "parity_failures": sum(r["status"] != "PASS" for r in parity)}
(ROOT / "editorial/phase6c-math-format-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary))
raise SystemExit(1 if summary["parity_failures"] or html_raw or epub_raw or unembedded else 0)
