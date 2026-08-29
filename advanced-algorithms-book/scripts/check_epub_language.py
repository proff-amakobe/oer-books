#!/usr/bin/env python3
"""Inspect the EPUB container and require English package metadata."""

from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EPUB = ROOT / "_book" / "Advanced-Computational-Algorithms.epub"
CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
DC_NS = {"dc": "http://purl.org/dc/elements/1.1/"}

with zipfile.ZipFile(EPUB) as archive:
    container = ET.fromstring(archive.read("META-INF/container.xml"))
    rootfile = container.find(".//c:rootfile", CONTAINER_NS)
    if rootfile is None:
        raise SystemExit("FAIL: EPUB container has no rootfile")
    package = ET.fromstring(archive.read(rootfile.attrib["full-path"]))

language = package.find(".//dc:language", DC_NS)
actual = language.text.strip() if language is not None and language.text else ""
if actual not in {"en", "en-US"}:
    raise SystemExit(f"FAIL: expected en or en-US, found {actual!r}")
print(f"PASS: EPUB dc:language is {actual}")
