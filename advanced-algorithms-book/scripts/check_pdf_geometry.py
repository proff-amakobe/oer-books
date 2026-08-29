#!/usr/bin/env python3
"""Fail unless the production PDF MediaBox is 8.5 x 11 inches."""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "_book" / "Advanced-Computational-Algorithms.pdf"
EXPECTED = (612.0, 792.0)

result = subprocess.run(["pdfinfo", str(PDF)], check=True, text=True, capture_output=True)
match = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+) pts", result.stdout, re.M)
if not match:
    raise SystemExit("FAIL: pdfinfo did not report a page size")
actual = tuple(float(value) for value in match.groups())
if any(abs(a - e) > 0.25 for a, e in zip(actual, EXPECTED)):
    raise SystemExit(f"FAIL: expected {EXPECTED[0]} x {EXPECTED[1]} pt, found {actual[0]} x {actual[1]} pt")
print(f"PASS: PDF geometry is {actual[0]} x {actual[1]} pt (8.5 x 11 in)")
