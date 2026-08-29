#!/usr/bin/env python3
"""Classify, tag, inventory, and audit fenced technical blocks."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = sorted((ROOT / "chapters").glob("*.qmd"))
VALID = {
    "PROGRAM_CODE", "PSEUDOCODE", "TERMINAL_COMMAND", "PROGRAM_OUTPUT",
    "CONFIGURATION", "DATA", "TEXT_DIAGRAM", "INLINE_EXAMPLE", "OTHER",
}
CLASS = {
    "PROGRAM_CODE": "program-code", "PSEUDOCODE": "algorithm",
    "TERMINAL_COMMAND": "terminal", "PROGRAM_OUTPUT": "program-output",
    "CONFIGURATION": "configuration", "DATA": "data-example",
    "TEXT_DIAGRAM": "text-diagram", "INLINE_EXAMPLE": "inline-example",
    "OTHER": "technical-other",
}
PROGRAM = {"python", "java", "javascript", "js", "c", "cpp", "c++", "r", "rust", "go"}
CONFIG = {"yaml", "yml", "json", "toml", "ini", "xml", "markdown", "md", "bibtex"}
SHELL = {"bash", "sh", "shell", "zsh", "console", "terminal"}


def headings(lines: list[str], stop: int) -> tuple[str, str]:
    chapter = section = ""
    for line in lines[:stop]:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            title = re.sub(r"\s+\{.*\}\s*$", "", m.group(2))
            if len(m.group(1)) == 1:
                chapter = title
            else:
                section = title
    return chapter, section


def parse_open(tail: str) -> tuple[str, list[str]]:
    tail = tail.strip()
    if tail.startswith("{") and tail.endswith("}"):
        tokens = tail[1:-1].split()
        language = next((x[1:] for x in tokens if x.startswith(".") and x[1:] not in CLASS.values()), "")
        return language.lower(), tokens
    return tail.lower(), ([f".{tail.lower()}"] if tail else [])


def classify(language: str, content: str, context: str, tokens: list[str]) -> str:
    low, ctx = content.lower(), context.lower()
    rows = [x for x in content.splitlines() if x.strip()]
    if language in PROGRAM:
        return "PROGRAM_CODE"
    if language in SHELL or any(re.match(r"^\s*\$\s+\S", x) for x in rows):
        return "TERMINAL_COMMAND"
    if language in CONFIG:
        return "CONFIGURATION"
    if language in {"csv", "tsv"}:
        return "DATA"
    if re.search(r"(?:expected |sample |program )?output\s*:?\s*$", ctx[-500:]) or "output" in ctx[-100:]:
        if not re.search(r"\b(def|class|function|for|while|return|if)\b", low):
            return "PROGRAM_OUTPUT"
    if any(c in content for c in "┌┐└┘├┤┬┴│─┼") or re.search(r"(?:^|\n)\s*[|+\\/].{2,}[|+\\/]", content):
        return "TEXT_DIAGRAM"
    pseudo_signals = len(re.findall(r"(?mi)^\s*(algorithm|input|output|for each|for |while |if |else|return|repeat|procedure|function)\b", content))
    if "pseudocode" in ctx[-600:] or "algorithm" in ctx[-180:] or "←" in content or pseudo_signals >= 2:
        return "PSEUDOCODE"
    if re.search(r"(?m)^\s*(pip|python|pytest|npm|git|quarto|make|docker)\s+", content):
        return "TERMINAL_COMMAND"
    if re.search(r"(?m)^\s*[\[{].*[\]}],?\s*$", content) and len(rows) <= 12:
        return "DATA"
    if len(rows) <= 3 and all(len(x) < 70 for x in rows):
        return "INLINE_EXAMPLE"
    return "OTHER"


def scan(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    blocks, i, ordinal = [], 0, 0
    while i < len(lines):
        opener = re.match(r"^(\s{0,3})(`{3,})([^`]*)$", lines[i].rstrip("\n"))
        if not opener:
            i += 1
            continue
        start = i
        indent, fence = opener.group(1), opener.group(2)
        language, tokens = parse_open(opener.group(3))
        i += 1
        body = []
        close = re.compile(r"^\s{0,3}`{" + str(len(fence)) + r",}\s*$")
        while i < len(lines) and not close.match(lines[i]):
            body.append(lines[i].rstrip("\n")); i += 1
        if i >= len(lines):
            raise ValueError(f"Unclosed fence: {path}:{start + 1}")
        ordinal += 1
        chapter_title, section = headings(lines, start)
        context = "".join(x for x in lines[max(0, start - 12):start] if not x.startswith("```"))
        semantic = classify(language, "\n".join(body), context, tokens)
        blocks.append({"start": start, "end": i, "indent": indent, "fence": fence, "language": language, "tokens": tokens,
                       "body": body, "semantic": semantic, "section": section,
                       "chapter_title": chapter_title, "ordinal": ordinal})
        i += 1
    return lines, blocks


def tag_file(path: Path, lines: list[str], blocks: list[dict]) -> int:
    changed = 0
    chapter_match = re.match(r"(\d+)-", path.name)
    chapter_attr = "chapter=" + str(int(chapter_match.group(1))) if chapter_match else None
    for block in reversed(blocks):
        cls = "." + CLASS[block["semantic"]]
        tokens = [x for x in block["tokens"]
                  if x not in {"." + c for c in CLASS.values()} and not x.startswith("chapter=")]
        if cls not in tokens:
            tokens.append(cls)
        language = block["language"]
        if language and not any(x == "." + language for x in tokens):
            tokens.insert(0, "." + language)
        if chapter_attr:
            tokens.append(chapter_attr)
        new = block["indent"] + block["fence"] + "{" + " ".join(tokens) + "}\n"
        if lines[block["start"]] != new:
            lines[block["start"]] = new; changed += 1
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write semantic classes into source fences")
    args = parser.parse_args()
    rows, errors, changed = [], [], 0
    for path in CHAPTERS:
        lines, blocks = scan(path)
        if args.apply:
            changed += tag_file(path, lines, blocks)
        chapter = re.search(r"/(\d+)-", "/" + str(path.relative_to(ROOT)))
        chapter_no = int(chapter.group(1)) if chapter else ""
        for b in blocks:
            content = "\n".join(b["body"])
            lengths = [len(x) for x in b["body"]]
            block_id = f"ch{chapter_no:02d}-b{b['ordinal']:03d}" if isinstance(chapter_no, int) else f"b{b['ordinal']:03d}"
            long = sum(n > 100 for n in lengths)
            semantic = b["semantic"]
            if semantic not in VALID or not content.strip(): errors.append(f"{block_id}: invalid or empty")
            rows.append({
                "block_id": block_id, "chapter": chapter_no, "section": b["section"],
                "source_file": str(path.relative_to(ROOT)), "current_language": b["language"] or "UNLABELED",
                "semantic_type": semantic, "executable": "YES" if semantic == "PROGRAM_CODE" else "NO",
                "complete_or_fragment": "FRAGMENT" if len(b["body"]) < 8 else "COMPLETE_OR_SUBSTANTIAL",
                "title_needed": "NO",
                "caption_needed": "NO", "line_numbers": "NO",
                "print_style": CLASS[semantic], "html_style": CLASS[semantic], "epub_style": CLASS[semantic],
                "overflow_risk": "HIGH" if long else "LOW", "special_handling": f"{long} lines >100 chars" if long else "NONE",
                "status": "CLASSIFIED", "notes": f"lines {b['start'] + 1}-{b['end'] + 1}; {len(b['body'])} content lines",
            })
    out = ROOT / "editorial/PHASE-5-TECHNICAL-BLOCK-MANIFEST.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    inventory_fields = ["chapter", "section", "source_file", "language", "block_type",
                        "fenced_correctly", "syntax_highlightable", "line_length_issue",
                        "execution_possible", "expected_output_present", "formatting_problem",
                        "recommended_style", "notes", "phase5_block_id", "phase5_status"]
    with (ROOT / "editorial/code-inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields); writer.writeheader()
        for row in rows:
            writer.writerow({
                "chapter": row["chapter"], "section": row["section"], "source_file": row["source_file"],
                "language": row["current_language"], "block_type": row["semantic_type"],
                "fenced_correctly": "YES", "syntax_highlightable": "YES" if row["current_language"] != "UNLABELED" else "NO",
                "line_length_issue": "YES" if row["overflow_risk"] == "HIGH" else "NO",
                "execution_possible": row["executable"], "expected_output_present": "YES" if row["semantic_type"] == "PROGRAM_OUTPUT" else "NO",
                "formatting_problem": "NONE" if row["overflow_risk"] == "LOW" else row["special_handling"],
                "recommended_style": row["print_style"], "notes": row["notes"],
                "phase5_block_id": row["block_id"], "phase5_status": row["status"],
            })
    counts = Counter(r["semantic_type"] for r in rows)
    print(f"technical_blocks={len(rows)} tagged_openers_changed={changed}")
    for key in sorted(VALID): print(f"{key}={counts[key]}")
    print(f"overlong_lines={sum(int(r['special_handling'].split()[0]) for r in rows if r['overflow_risk'] == 'HIGH')}")
    print(f"errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
