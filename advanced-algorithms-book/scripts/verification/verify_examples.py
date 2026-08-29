#!/usr/bin/env python3
"""Conservatively classify and verify fenced examples in the manuscript.

This script never executes a block containing top-level calls, loops, context
managers, exception handlers, or imports from modules with system/network
capabilities.  Execution happens in a temporary directory with a timeout.
"""

from __future__ import annotations

import ast
import csv
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = sorted((ROOT / "chapters").glob("*.qmd"))
OUTPUT = ROOT / "editorial" / "code-verification-results.csv"
FENCE = re.compile(r"^\s{0,3}(`{3,})\s*(?:\{\s*\.?([A-Za-z0-9_+.-]*)|([A-Za-z0-9_+.-]*))")
CHAPTER = re.compile(r"/(\d{2})-")
SAFE_IMPORTS = {
    "bisect", "collections", "dataclasses", "functools", "heapq", "itertools",
    "math", "operator", "random", "statistics", "string", "typing",
}
PROGRAM_LANGUAGES = {"python", "java", "javascript", "c", "cpp", "c++", "bash", "sh", "yaml"}


@dataclass
class Block:
    chapter: int
    section: str
    source: Path
    language: str
    start: int
    code: str


def blocks() -> list[Block]:
    found: list[Block] = []
    for source in CHAPTERS:
        match = CHAPTER.search(source.as_posix())
        chapter = int(match.group(1)) if match else 0
        section = ""
        lines = source.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("#"):
                section = re.sub(r"^#+\s*", "", lines[i]).strip()
            fence = FENCE.match(lines[i])
            if not fence:
                i += 1
                continue
            fence_width = len(fence.group(1))
            language = (fence.group(2) or fence.group(3) or "").lower()
            start = i + 1
            i += 1
            body: list[str] = []
            while i < len(lines) and not re.match(r"^\s{0,3}`{" + str(fence_width) + r",}\s*$", lines[i]):
                body.append(lines[i])
                i += 1
            found.append(Block(chapter, section, source, language, start, "\n".join(body) + "\n"))
            i += 1
    return found


def inert_python(tree: ast.Module) -> tuple[bool, str]:
    """Return whether executing a module can only define inert objects."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            if any(name not in SAFE_IMPORTS for name in names):
                return False, "non-baseline or system-capable import"
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or all(not isinstance(child, ast.Call) for child in ast.walk(value)):
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        return False, f"top-level {type(node).__name__} requires manual review"
    return True, "definition-only block"


def verify(block: Block, ordinal: int) -> dict[str, str | int]:
    language = block.language or "unlabeled"
    base = {
        "chapter": block.chapter,
        "section": block.section,
        "language": language,
        "example_id": f"ch{block.chapter:02d}-b{ordinal:03d}-L{block.start}",
        "classification": "ILLUSTRATIVE ONLY",
        "execution_status": "NOT TESTED",
        "expected": "",
        "actual": "",
        "error": "",
        "correctness": "MANUAL REVIEW",
        "action_taken": "none",
        "notes": f"{block.source.relative_to(ROOT)}:{block.start}",
    }
    corrected = {
        (10, "The Naive Approach"): "aligned empty-pattern semantics with KMP",
        (11, "Strassen's Algorithm: O(n^2.807)"): "preserved original dimensions when padding odd matrices",
        (13, "**Smoothed Analysis**"): "replaced built-in sort benchmark with explicit quicksort",
    }.get((block.chapter, block.section))
    if corrected:
        base["action_taken"] = corrected
    if language != "python":
        if language in PROGRAM_LANGUAGES:
            base["classification"] = "REQUIRES TOOLCHAIN" if language not in {"bash", "sh", "yaml"} else "MANUAL REVIEW"
        elif re.search(r"\b(def|class|return|import|print)\b", block.code):
            base["classification"] = "PARTIAL / SNIPPET"
        else:
            base["classification"] = "PSEUDOCODE"
        return base
    try:
        tree = ast.parse(block.code, filename=str(block.source))
    except SyntaxError as exc:
        base.update(classification="PARTIAL / SNIPPET", execution_status="COMPILE FAIL", error=f"{exc.msg} (line {exc.lineno})", correctness="NOT TESTED")
        return base
    safe, reason = inert_python(tree)
    if not safe:
        base.update(classification="MANUAL REVIEW", execution_status="COMPILE PASS", correctness="NOT TESTED", notes=f"{base['notes']}; {reason}")
        return base
    with tempfile.TemporaryDirectory(prefix="book-example-") as temp_dir:
        path = Path(temp_dir) / "example.py"
        path.write_text(block.code, encoding="utf-8")
        try:
            run = subprocess.run(
                [sys.executable, "-I", str(path)], cwd=temp_dir, text=True,
                capture_output=True, timeout=5, check=False,
            )
        except subprocess.TimeoutExpired:
            base.update(classification="MANUAL REVIEW", execution_status="TIMEOUT", error="5 second timeout", correctness="NOT TESTED")
            return base
    if run.returncode == 0:
        base.update(classification="PASS", execution_status="PASS", actual=(run.stdout + run.stderr).strip(), correctness="SYNTAX/DEFINITION PASS", notes=f"{base['notes']}; {reason}")
    else:
        base.update(classification="REQUIRES DEPENDENCY" if "ModuleNotFoundError" in run.stderr else "FAIL", execution_status="FAIL", error=run.stderr.strip(), correctness="FAIL")
    return base


def main() -> int:
    rows: list[dict[str, str | int]] = []
    per_chapter: dict[int, int] = {}
    for block in blocks():
        per_chapter[block.chapter] = per_chapter.get(block.chapter, 0) + 1
        rows.append(verify(block, per_chapter[block.chapter]))
    fields = ["chapter", "section", "language", "example_id", "classification", "execution_status", "expected", "actual", "error", "correctness", "action_taken", "notes"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    print(f"blocks={len(rows)}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
