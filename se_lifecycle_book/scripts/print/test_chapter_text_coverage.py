#!/usr/bin/env python3
import re,sys
from reconstruction_common import CHAPTERS,pdf_text,plain,norm
def main():
 text=set(norm(pdf_text()).split()); bad=[]
 for n,p in enumerate(CHAPTERS,1):
  src=p.read_text(); src=re.sub(r"```.*?```"," ",src,flags=re.S); src=re.sub(r":::.*?::: *"," ",src,flags=re.S)
  words=[x for x in norm(plain(src)).split() if len(x)>2]; coverage=100*sum(w in text for w in words)/max(1,len(words))
  print(f"chapter={n:02d} coverage={coverage:.2f}%")
  if coverage<98: bad.append((n,coverage))
 print(f"manual_review_flags={len(bad)}")
 # Coverage is deliberately a flag, not an automatic content-loss verdict.
 return 0
if __name__=='__main__': sys.exit(main())
