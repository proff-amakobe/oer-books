#!/usr/bin/env python3
"""Classify unique external links in the rendered HTML without deleting evidence."""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "_book"


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.values = set()
    def handle_starttag(self, tag, attrs):
        data = dict(attrs); href = data.get("href", "")
        if tag == "a" and href.startswith(("http://", "https://")): self.values.add(href)


def check(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Phase7LinkAudit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15, context=ssl.create_default_context()) as response:
            final = response.geturl(); code = response.status
            return {"url": url, "status_code": code, "final_url": final, "classification": "redirect" if final != url else "ok"}
    except urllib.error.HTTPError as exc:
        classification = "unverifiable" if exc.code in {401, 403, 429} else "broken"
        return {"url": url, "status_code": exc.code, "final_url": exc.geturl(), "classification": classification, "detail": str(exc.reason)}
    except Exception as exc:
        return {"url": url, "status_code": None, "final_url": None, "classification": "unverifiable", "detail": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--report", type=Path, required=True); args = parser.parse_args()
    links = Links()
    for path in BOOK.rglob("*.html"): links.feed(path.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=8) as pool: results = sorted(pool.map(check, links.values), key=lambda item: item["url"])
    summary = {name: sum(item["classification"] == name for item in results) for name in ("ok", "redirect", "broken", "unverifiable")}
    report = {"checked": len(results), "summary": summary, "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checked": len(results), **summary}))
    raise SystemExit(summary["broken"] > 0)


if __name__ == "__main__": main()
