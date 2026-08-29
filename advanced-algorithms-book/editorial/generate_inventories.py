#!/usr/bin/env python3
"""Generate evidence inventories for the Second Edition audit; does not edit source."""
from pathlib import Path
import csv, re

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = sorted((ROOT / "chapters").glob("*.qmd"))


def chapter_number(path):
    return int(path.name[:2])


def headings(lines):
    current = ""
    out = {}
    fenced = False
    for number, line in enumerate(lines, 1):
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and re.match(r"^#{1,6}\s+", line):
            current = re.sub(r"^#{1,6}\s+", "", line).strip()
        out[number] = current
    return out


def blocks(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    hs = headings(lines)
    open_block = None
    body = []
    for number, line in enumerate(lines, 1):
        if line.startswith("```"):
            if open_block is None:
                open_block = (number, line[3:].strip().split()[0] if line[3:].strip() else "")
                body = []
            else:
                start, language = open_block
                yield start, number, language, "\n".join(body), hs.get(start, "")
                open_block, body = None, []
        elif open_block is not None:
            body.append(line)


def block_type(language, body):
    lang = language.lower()
    if lang in {"python", "py"}: return "PYTHON"
    if lang == "java": return "JAVA"
    if lang in {"javascript", "js"}: return "JAVASCRIPT"
    if lang in {"c", "cpp", "c++"}: return "C/C++"
    if lang in {"bash", "sh", "shell", "zsh"}: return "SHELL"
    if lang in {"yaml", "yml", "json", "toml", "markdown"}: return "CONFIGURATION"
    if is_diagram(body): return "ASCII DIAGRAM"
    if re.search(r"(^|\n)\s*(Input|Output|Algorithm|for each|while|if)\b", body, re.I): return "PSEUDOCODE"
    if re.search(r"(^|\n)(\$|>>>|In \[\d+\]:|Output:)", body): return "TERMINAL OUTPUT"
    if re.search(r"\b(def|class|import|return|print)\b", body): return "PYTHON"
    if re.search(r"[∑∀∈≤≥]|T\(n\)\s*=|O\([^)]*\)", body): return "MATHEMATICS DISGUISED AS CODE"
    return "UNKNOWN"


def is_diagram(body):
    art = sum(body.count(x) for x in ("-->", "->", "←", "→", "│", "├", "└", "─", "┌", "┐", "▼", "▲"))
    tree = bool(re.search(r"\n\s*/\s+\\|\n\s*/\\", body))
    network = body.count("[") >= 3 and ("-->" in body or "--[" in body)
    return art >= 2 or tree or network


with (ROOT / "editorial/code-inventory.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["chapter", "section", "source_file", "language", "block_type", "fenced_correctly",
              "syntax_highlightable", "line_length_issue", "execution_possible", "expected_output_present",
              "formatting_problem", "recommended_style", "notes"]
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for path in CHAPTERS:
        for start, end, language, body, section in blocks(path):
            kind = block_type(language, body)
            maxlen = max((len(x) for x in body.splitlines()), default=0)
            runnable = "YES" if kind in {"PYTHON", "JAVA", "JAVASCRIPT", "C/C++", "SHELL"} else "NO"
            output = "YES" if re.search(r"(^|\n)(Output:|>>>|\$ )", body) else "NO"
            problem = []
            if not language: problem.append("missing language tag")
            if maxlen > 76: problem.append(f"max line {maxlen} chars")
            style = "PROGRAM CODE" if runnable == "YES" and kind != "SHELL" else "TERMINAL" if kind in {"SHELL", "TERMINAL OUTPUT"} else "ALGORITHM" if kind == "PSEUDOCODE" else "TEXT/FIGURE REVIEW"
            writer.writerow(dict(chapter=chapter_number(path), section=section, source_file=path.relative_to(ROOT),
                language=language or "UNLABELED", block_type=kind, fenced_correctly="YES",
                syntax_highlightable="YES" if language else "NO", line_length_issue="YES" if maxlen > 76 else "NO",
                execution_possible=runnable, expected_output_present=output,
                formatting_problem="; ".join(problem) or "NONE OBSERVED", recommended_style=style,
                notes=f"lines {start}-{end}; execution status requires isolated test"))

with (ROOT / "editorial/figure-inventory.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["chapter", "section", "source_file", "line_or_heading", "current_representation", "figure_type",
              "pedagogical_purpose", "recommended_action", "recommended_format", "complexity", "priority", "notes"]
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for path in CHAPTERS:
        for start, end, language, body, section in blocks(path):
            if not is_diagram(body) or language: continue
            sample = body.lower()
            if "├" in body or "└" in body: figtype, purpose, action, priority = "directory tree", "Show project/package organization", "KEEP AS TEXT", "LOW"
            elif "flow" in sample or "capacity" in sample or "--[" in body: figtype, purpose, action, priority = "flow network", "Explain capacity, residual flow, or cut structure", "REPLACE WITH SVG", "CRITICAL"
            elif "fib(" in body or "level" in sample or "work" in sample: figtype, purpose, action, priority = "recursion tree", "Explain recursive decomposition and work by level", "REPLACE WITH SVG", "HIGH"
            elif "segment tree" in sample or "[0-" in body: figtype, purpose, action, priority = "tree structure", "Explain range decomposition", "REPLACE WITH SVG", "HIGH"
            elif "/" in body and "\\" in body: figtype, purpose, action, priority = "tree structure", "Explain hierarchy or balance", "REPLACE WITH SVG", "HIGH"
            else: figtype, purpose, action, priority = "pseudo-visual block", "Support spatial explanation", "REVIEW MANUALLY", "MEDIUM"
            writer.writerow(dict(chapter=chapter_number(path), section=section, source_file=path.relative_to(ROOT),
                line_or_heading=f"lines {start}-{end}", current_representation="unlabeled fenced monospaced block",
                figure_type=figtype, pedagogical_purpose=purpose, recommended_action=action,
                recommended_format="SVG" if action == "REPLACE WITH SVG" else "TEXT", complexity="MEDIUM", priority=priority,
                notes=body.splitlines()[0][:120] if body.splitlines() else "empty"))

elements = {
    "clear chapter title": r"^#{1,2}\s+(?:Chapter\s+\d+|\d+\.\d+\s+Introduction)",
    "chapter subtitle": r"^##\s+(?!Section|\d+\.)",
    "learning objectives": r"learn(?:ing|'ll)|objectives",
    "introduction": r"^#{2,4}.*(?:Introduction|Welcome)",
    "conceptual explanation": r"concept|foundation|idea|intuition|understanding",
    "worked example": r"worked|step.by.step|example",
    "pseudocode": r"pseudocode|algorithm",
    "correctness argument": r"correctness|proof|invariant",
    "complexity analysis": r"complexity|analysis",
    "executable implementation": r"```(?:python|java|javascript|bash)",
    "real-world application": r"real.world|application|case stud",
    "summary": r"^#{2,4}.*summary|key takeaways",
    "key terms": r"key terms|glossary",
    "review questions": r"review questions|conceptual understanding|understanding$",
    "exercises": r"^#{2,4}.*exercises|practice problems",
    "project/activity": r"project|activity",
    "references": r"references|further reading|recommended reading",
}
with (ROOT / "editorial/chapter-structure-audit.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["chapter"] + list(elements)
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
    for path in CHAPTERS:
        text = path.read_text(encoding="utf-8")
        row = {"chapter": chapter_number(path)}
        for name, pattern in elements.items(): row[name] = "YES" if re.search(pattern, text, re.I | re.M) else "NO"
        writer.writerow(row)
