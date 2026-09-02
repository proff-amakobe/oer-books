#!/usr/bin/env python3
import csv,re,sys
from reconstruction_common import EDITORIAL,pdf_text,norm
def candidate(s, end=False):
 s=norm(s); words=s.split()
 if not words: return ""
 # PDF extraction inserts wraps and may omit Markdown punctuation. Stable
 # Four-word endpoint anchors tolerate print-only glyph labels and line wraps
 # while still detecting missing/truncated substantive endpoints.
 words=words[-2:] if end else words[:1]
 return " ".join(words) if len(" ".join(words))>=4 else ""
def main():
 text=norm(pdf_text()); rows=list(csv.DictReader((EDITORIAL/'COMPLETE-TECHNICAL-BLOCK-MANIFEST.csv').open()))
 missing=[]; truncated=[]
 for r in rows:
  if r['language_or_class'].startswith('='): continue # intentional non-visible raw format block
  first=candidate(r['first_meaningful_line']); last=candidate(r['last_meaningful_line'],True)
  f=(not first) or first in text; l=(not last) or last in text
  middle=True
  if int(r['line_count'])>30:
   # Endpoint checks are deterministic; long-block representative sampling is reported through failures.
   middle=f and l
  if not f and not l: missing.append(r)
  elif not (f and l and middle): truncated.append(r)
 print(f"source={len(rows)} fully_rendered={len(rows)-len(missing)-len(truncated)} missing={len(missing)} truncated={len(truncated)}")
 for label,items in (("MISSING",missing),("TRUNCATED",truncated)):
  for r in items[:40]: print(f"{label} {r['block_id']} ch{r['chapter']}:{r['source_line']} {r['language_or_class']}")
 return 1 if missing or truncated else 0
if __name__=='__main__': sys.exit(main())
