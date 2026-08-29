#!/usr/bin/env python3
"""Audit rendered HTML semantics, copy controls, IDs, and responsive containment."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HTML = sorted((ROOT / "_book/chapters").glob("*.html"))
CSS = (ROOT / "styles/technical-blocks.css").read_text(encoding="utf-8")
valid = {"program-code", "algorithm", "terminal", "program-output", "configuration",
         "data-example", "text-diagram", "inline-example", "technical-other"}
counts: Counter[str] = Counter(); ids: list[str] = []; source_blocks = 0; copy_buttons = 0; invalid = []
for path in HTML:
    text = path.read_text(encoding="utf-8")
    source_blocks += len(re.findall(r'<div class="sourceCode', text))
    copy_buttons += len(re.findall(r'class="code-copy-button', text))
    for match in re.finditer(r'<div class="technical-block ([^"]+)"', text):
        classes = set(match.group(1).split())
        semantic = classes & valid
        if len(semantic) != 1:
            invalid.append(f"{path.name}: {sorted(classes)}")
            continue
        counts[next(iter(semantic))] += 1
    ids.extend(re.findall(r'id="(alg-\d+-\d+)"', text))
duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
checks = {
    "chapter_pages": len(HTML), "technical_blocks": sum(counts.values()), "semantic_counts": dict(counts),
    "invalid_semantic_wrappers": invalid, "source_blocks": source_blocks,
    "copy_buttons": copy_buttons, "copy_button_coverage": copy_buttons >= source_blocks,
    "duplicate_algorithm_ids": duplicates,
    "horizontal_scroll_rule": "overflow-x: auto" in CSS,
    "mobile_rule": "@media (max-width: 576px)" in CSS,
    "pseudocode_wrap_rule": ".technical-block.algorithm pre" in CSS and "white-space: pre-wrap" in CSS,
    "page_wide_fixed_widths": len(re.findall(r"(?<!max-)width:\s*\d{3,}px", CSS)),
}
(ROOT / "editorial/phase5-html-technical-qa.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
print(json.dumps({**checks, "semantic_counts": dict(counts)}))
raise SystemExit(bool(invalid or not checks["copy_button_coverage"] or duplicates or not checks["horizontal_scroll_rule"]
                      or not checks["mobile_rule"] or not checks["pseudocode_wrap_rule"]
                      or checks["page_wide_fixed_widths"]))
