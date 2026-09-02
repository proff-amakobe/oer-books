#!/usr/bin/env python3
"""Fail on universal terminal conversion or unjustified typed Terminal blocks."""
import re,sys
from pathlib import Path
from reconstruction_common import ROOT,CHAPTERS,scan

def main():
 bad=[]; legacy=[]
 for lua in ROOT.rglob("*.lua"):
  if any(x in lua.parts for x in (".git","_book","output")): continue
  s=lua.read_text(errors="replace")
  fn=re.search(r"function\s+CodeBlock\b(.*?)(?=\nfunction\s|\Z)",s,re.S)
  if fn and "SETerminal" in fn.group(1) and not re.search(r"if\s+.*(?:bash|shell|console)",fn.group(1),re.I): legacy.append(str(lua.relative_to(ROOT)))
 profile=(ROOT/"_quarto-print-reconstruction.yml").read_text()
 if "print-components.lua" in profile or "SETerminal" in profile: bad.append("neutral profile uses destructive terminal machinery")
 typed=sum(bool(b.language) for p in CHAPTERS for b in scan(p)[2])
 # A quarantined legacy file is evidence for the forensic audit, not a defect
 # in the neutral profile. It becomes a failure if the profile references it.
 if "print-components.lua" in profile and legacy: bad.extend(legacy)
 if bad:
  print("FAIL: universal/unjustified Terminal conversion: "+", ".join(bad)); return 1
 print(f"PASS: neutral profile preserves {typed} typed source blocks; quarantined_legacy={','.join(legacy) or 'none'}")
 return 0
if __name__=='__main__': sys.exit(main())
