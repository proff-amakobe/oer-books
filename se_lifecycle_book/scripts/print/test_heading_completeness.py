#!/usr/bin/env python3
import csv,sys
from reconstruction_common import EDITORIAL,pdf_text,norm
def main():
 text=norm(pdf_text()); rows=list(csv.DictReader((EDITORIAL/'COMPLETE-HEADING-MANIFEST.csv').open()))
 missing=[]; duplicate=[]
 for r in rows:
  needle=norm(r['heading_text']); n=text.count(needle)
  if not needle or n==0: missing.append(r)
  # Repeated generic headings are legitimate; duplication is not safely inferred from text extraction.
 print(f"source={len(rows)} found={len(rows)-len(missing)} missing={len(missing)} duplicated={len(duplicate)}")
 for r in missing[:30]: print(f"MISSING ch{r['chapter']} L{r['heading_level']}: {r['heading_text']}")
 return 1 if missing else 0
if __name__=='__main__': sys.exit(main())
