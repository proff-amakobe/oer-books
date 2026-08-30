#!/usr/bin/env python3
"""Inventory semantic and suspect mathematical content in the canonical QMD source."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


MATH_CMD = re.compile(r"\\(?:Theta|Omega|alpha|beta|epsilon|delta|lambda|mu|pi|phi|sum|prod|frac|sqrt|le|ge|ne|approx|in|subset|cup|cap|infty|nabla|partial)\b")
SYMBOL = re.compile(r"[≤≥≠≈∼∝∈∉⊂⊆∪∩→←↔⇒⇔∞√ΣΠ∂∇ΘΩαβεδλμπφ₀₁₂₃₄₅₆₇₈₉ᵢⁿ²³]")
ASCII_MATH = re.compile(r"(?:\b[OTPWEHFXQKVGfc]\([^\n]{0,100}[=<>]|\b(?:Theta|Omega|Big-O)\b|(?:^|\s)(?:<=|>=|!=)(?:\s|$)|\b(?:log|exp|sqrt)\s*\(|\bfor all\b)")
TYPE_PATTERNS = [
    ("RECURRENCE", re.compile(r"\bT\s*\(\s*n\s*\)")),
    ("PROBABILITY", re.compile(r"(?:\\Pr|P\s*\(|expect|probab)", re.I)),
    ("SUMMATION", re.compile(r"(?:\\sum|Σ)")),
    ("MATRIX", re.compile(r"(?:bmatrix|matrix|det\b)", re.I)),
    ("VECTOR", re.compile(r"(?:\\vec|vector|dot product)", re.I)),
    ("SET_NOTATION", re.compile(r"[∈∉⊂⊆∪∩]|\\(?:in|notin|subset|cup|cap)\b")),
    ("FLOW_CONSTRAINT", re.compile(r"(?:flow|capacity|f\s*\(\s*[uv])", re.I)),
    ("APPROXIMATION_RATIO", re.compile(r"(?:approximation|OPT|ratio)", re.I)),
    ("NUMERICAL_FORMULA", re.compile(r"(?:FFT|Fourier|convolution|e\^|\\exp)", re.I)),
    ("INEQUALITY", re.compile(r"[≤≥<>]|\\(?:le|ge|ne)\b")),
    ("ASYMPTOTIC_BOUND", re.compile(r"(?:\\(?:Theta|Omega)|[ΘΩ]|\bO\s*\()")),
    ("GRAPH_NOTATION", re.compile(r"G\s*=\s*\(\s*V|E\s*\)")),
]


def classify(text: str) -> str:
    for label, pattern in TYPE_PATTERNS:
        if pattern.search(text):
            return label
    if re.search(r"(?:theorem|proof)", text, re.I):
        return "PROOF_EXPRESSION"
    if re.search(r"[=+*/^]|\\frac", text):
        return "ALGORITHM_FORMULA"
    return "COMPLEXITY_EXPRESSION" if re.search(r"complex", text, re.I) else "OTHER"


def chapter_for(path: Path) -> str:
    m = re.match(r"(\d+)-", path.name)
    return str(int(m.group(1))) if m else "front-matter"


def section_at(lines: list[str], line_no: int) -> str:
    for line in reversed(lines[:line_no]):
        if line.startswith("#"):
            return re.sub(r"\s*\{.*$", "", line.lstrip("# ")).strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("editorial/PHASE-6C-MATH-MANIFEST.csv"))
    args = ap.parse_args()
    # index.qmd contains an HTML-only decorative recurrence motif, not canonical
    # manuscript mathematics; the print preface and EPUB content begin after it.
    files = sorted(Path("chapters").glob("*.qmd")) + [Path(p) for p in ["title.qmd", "edition.qmd", "copyright.qmd", "dedication.qmd", "about-author.qmd", "about-gdsi.qmd", "how-to-use.qmd", "references.qmd"] if Path(p).exists()]
    rows, suspect_blocks = [], []
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        fence, fence_start, fence_info, fence_buf = False, 0, "", []
        display = False
        display_start = 0
        display_buf: list[str] = []
        for i, line in enumerate(lines, 1):
            if line.startswith("```"):
                if not fence:
                    fence, fence_start, fence_info, fence_buf = True, i, line, []
                else:
                    text = " ".join(fence_buf).strip()
                    if (SYMBOL.search(text) or ASCII_MATH.search(text)) and not re.search(r"\.(?:append|add)|\b(?:def|class|import|return|print|while|for|if)\b", text):
                        suspect_blocks.append((path, fence_start, fence_info, text))
                        rows.append((path, fence_start, text, "technical_block", "native_math", "display"))
                    fence = False
                continue
            if fence:
                fence_buf.append(line)
                continue
            if line.strip().startswith("$$"):
                if not display:
                    display, display_start, display_buf = True, i, [line.strip()[2:]]
                else:
                    display_buf.append(line.strip()[:-2] if line.strip().endswith("$$") else line)
                    text = " ".join(display_buf).strip()
                    rows.append((path, display_start, text, "native_math", "native_math", "display"))
                    display = False
                continue
            if display:
                display_buf.append(line)
                continue
            spans = list(re.finditer(r"(?<!\\)\$(?![\d\s$])(.+?)(?<!\\)\$", line))
            for span in spans:
                rows.append((path, i, span.group(1), "native_math", "native_math", "inline"))
            scrubbed = re.sub(r"(?<!\\)\$(?![\d\s$]).+?(?<!\\)\$", "", line)
            if (SYMBOL.search(scrubbed) or ASCII_MATH.search(scrubbed) or MATH_CMD.search(scrubbed)) and line.strip():
                rows.append((path, i, scrubbed.strip(), "text_or_ascii", "native_math", "inline"))

    seen = set()
    out = []
    for path, line, text, current, recommended, placement in rows:
        key = (str(path), line, text)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "equation_id": f"MATH-{len(out)+1:04d}", "chapter": chapter_for(path),
            "section": section_at(path.read_text(encoding="utf-8").splitlines(), line),
            "source_file": str(path), "source_line_or_anchor": str(line),
            "content_type": classify(text), "current_representation": current,
            "recommended_representation": recommended, "inline_or_display": placement,
            "numbered": "YES" if "#eq-" in text else "NO", "cross_referenced": "UNKNOWN",
            "variables_defined": "REVIEW", "assumptions_defined": "REVIEW",
            "render_pdf": "PENDING", "render_html": "PENDING", "render_epub": "PENDING",
            "visual_status": "PENDING", "notes": re.sub(r"\s+", " ", text)[:280],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader(); writer.writerows(out)
    print(f"math_inventory={len(out)} suspect_math_blocks={len(suspect_blocks)}")
    for p, line, info, text in suspect_blocks:
        print(f"SUSPECT {p}:{line} {info} :: {re.sub(r'\s+', ' ', text)[:180]}")


if __name__ == "__main__":
    main()
