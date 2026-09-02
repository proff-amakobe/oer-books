#!/usr/bin/env python3
import csv,re,subprocess,sys
from reconstruction_common import EDITORIAL,PDF,norm,pdf_text
def main():
 rows=list(csv.DictReader((EDITORIAL/'COMPLETE-FIGURE-MANIFEST.csv').open())); text=norm(pdf_text())
 info=subprocess.run(['pdfimages','-list',str(PDF)],capture_output=True,text=True)
 objects=sum(bool(re.match(r"\s*\d+\s+\d+\s+",x)) for x in info.stdout.splitlines()) if info.returncode==0 else -1
 missing_files=[r for r in rows if r['file_exists']!='True']; missing_captions=[r for r in rows if norm(r['caption']) and norm(r['caption']) not in text]
 print(f"source={len(rows)} pdf_image_objects={objects} missing_files={len(missing_files)} missing_captions={len(missing_captions)}")
 for r in (missing_files+missing_captions)[:30]: print(f"MISSING ch{r['chapter']}: {r['image_path']} / {r['caption']}")
 return 1 if missing_files or missing_captions or objects<1 else 0
if __name__=='__main__': sys.exit(main())
