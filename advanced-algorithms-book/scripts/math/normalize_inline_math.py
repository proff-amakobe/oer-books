#!/usr/bin/env python3
"""Mechanically promote asymptotic expressions in prose to native inline math."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def normalize(expr: str) -> str:
    return (expr.replace("²", "^2").replace("³", "^3").replace("ⁿ", "^n")
            .replace("₀", "_0").replace("₁", "_1").replace("₂", "_2")
            .replace("√", r"\sqrt ").replace("log₂", r"\log_2"))


def promote(segment: str) -> str:
    out, i = [], 0
    while i < len(segment):
        m = re.search(r"(?<![A-Za-z0-9_\\$])([OΘΩ])\(", segment[i:])
        if not m:
            out.append(segment[i:]); break
        start = i + m.start(); open_at = i + m.end() - 1
        out.append(segment[i:start])
        depth, j = 0, open_at
        while j < len(segment):
            if segment[j] == "(": depth += 1
            elif segment[j] == ")":
                depth -= 1
                if depth == 0: break
            j += 1
        if depth:
            out.append(segment[start:]); break
        expr = normalize(segment[start:j + 1]).replace("Θ", r"\Theta").replace("Ω", r"\Omega")
        out.append(f"${expr}$"); i = j + 1
    return "".join(out)


def transform(line: str) -> str:
    # Backtick code spans and existing math spans are protected by delimiters.
    parts = re.split(r"(`[^`]*`|(?<!\\)\$(?!\$).*?(?<!\\)\$)", line)
    return "".join(part if idx % 2 else promote(part) for idx, part in enumerate(parts))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); args = ap.parse_args()
    changed_lines = changed_files = 0
    for path in sorted(Path("chapters").glob("*.qmd")):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        output, fenced, display = [], False, False
        for line in lines:
            if line.startswith("```"):
                fenced = not fenced; output.append(line); continue
            if line.strip().startswith("$$"):
                display = not display
                output.append(line)
                continue
            # A display is already a native math environment. Remove any inline
            # delimiters introduced by an earlier normalization pass.
            if display:
                new = line.replace("$", "")
            else:
                new = line if fenced or line.lstrip().startswith("<") else transform(line)
            changed_lines += new != line; output.append(new)
        if output != lines:
            changed_files += 1
            if args.apply: path.write_text("".join(output), encoding="utf-8")
    print(f"files_changed={changed_files} lines_changed={changed_lines} applied={args.apply}")


if __name__ == "__main__": main()
