#!/usr/bin/env python3
"""
Audit script: scans the book manuscript for .ipynb references and verifies
that every referenced notebook actually exists in the notebooks/ directory.

Usage:
    python3 scripts/audit_notebook_references.py [manuscript_dir] [notebooks_dir]

Exits with status 0 if every referenced notebook exists, 1 otherwise.
Rerun this after any manuscript edit to catch newly introduced dangling
references before they reach a reader.
"""
import os
import re
import sys

NOTEBOOK_PATTERN = re.compile(r'\b([A-Za-z0-9_\-]+\.ipynb)\b')


def find_manuscript_files(manuscript_dir):
    exts = {".md", ".docx", ".txt"}
    files = []
    for root, _, filenames in os.walk(manuscript_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    return sorted(files)


def extract_text(path):
    """Best-effort text extraction. For .docx, uses pandoc if available;
    falls back to skipping the file with a warning if no converter is found."""
    if path.lower().endswith(".docx"):
        try:
            import subprocess
            result = subprocess.run(
                ["pandoc", "-t", "plain", path], capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        print(f"  (skipped text extraction for {path}: pandoc unavailable or failed)")
        return ""
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def scan_references(manuscript_dir):
    references = {}  # notebook filename -> list of (source file, context snippet)
    for path in find_manuscript_files(manuscript_dir):
        text = extract_text(path)
        for m in NOTEBOOK_PATTERN.finditer(text):
            nb_name = m.group(1)
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 20)
            snippet = text[start:end].replace("\n", " ")
            references.setdefault(nb_name, []).append((path, snippet))
    return references


def main():
    manuscript_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    notebooks_dir = sys.argv[2] if len(sys.argv) > 2 else "notebooks"

    references = scan_references(manuscript_dir)
    existing = set(os.listdir(notebooks_dir)) if os.path.isdir(notebooks_dir) else set()

    missing = {nb: ctx for nb, ctx in references.items() if nb not in existing}
    found = {nb: ctx for nb, ctx in references.items() if nb in existing}
    orphaned = existing - set(references.keys())

    print(f"Manuscript directory: {os.path.abspath(manuscript_dir)}")
    print(f"Notebooks directory:  {os.path.abspath(notebooks_dir)}")
    print(f"Unique notebook filenames referenced: {len(references)}")
    print(f"  Found on disk:   {len(found)}")
    print(f"  MISSING:         {len(missing)}")
    print(f"Orphaned notebooks (exist but never referenced): {len(orphaned)}")
    print()

    if missing:
        print("=" * 60)
        print("MISSING NOTEBOOKS -- referenced but not found:")
        print("=" * 60)
        for nb, contexts in sorted(missing.items()):
            print(f"\n{nb}")
            for path, snippet in contexts[:3]:
                print(f"    in {path}: ...{snippet}...")

    if orphaned:
        print("\n" + "=" * 60)
        print("ORPHANED NOTEBOOKS -- exist but not referenced anywhere:")
        print("=" * 60)
        for nb in sorted(orphaned):
            print(f"  {nb}")

    if missing:
        print(f"\nFAILED: {len(missing)} referenced notebook(s) are missing.")
        sys.exit(1)
    else:
        print("\nPASSED: every referenced notebook exists.")
        sys.exit(0)


if __name__ == "__main__":
    main()
