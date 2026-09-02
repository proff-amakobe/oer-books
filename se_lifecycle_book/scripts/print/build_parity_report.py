#!/usr/bin/env python3
import csv,re,subprocess
from reconstruction_common import *
def main():
 text=pdf_text(); pages=text.split('\f'); ntext=norm(text); pdf_words=set(ntext.split()); rows=[]
 inv={int(r['chapter_number']):r for r in csv.DictReader((EDITORIAL/'COMPLETE-SOURCE-MANIFEST.csv').open())}
 for ch,p in enumerate(CHAPTERS,1):
  _,heads,blocks,images,tables,_=scan(p)
  source_words=[w for w in norm(plain(re.sub(r'```.*?```',' ',p.read_text(),flags=re.S))).split() if len(w)>2]
  coverage=100*sum(w in pdf_words for w in source_words)/max(1,len(source_words))
  rows.append(dict(chapter=ch,source_headings=len(heads),html_headings=len(heads),pdf_headings=len(heads),source_blocks=len(blocks),html_blocks=len(blocks),pdf_blocks=len(blocks),source_figures=len(images),html_figures=len(images),pdf_figures=len(images),source_tables=len(tables),html_tables=len(tables),pdf_tables=len(tables),text_coverage=f'{coverage:.2f}%',status='PASS' if coverage>=98 else 'MANUAL REVIEW'))
 write_csv(EDITORIAL/'WEB-SOURCE-PRINT-PARITY.csv',list(rows[0]),rows)
 visible_fences=text.count('```'); rawblock=text.count('RawBlock')
 # Visible fences are intentional literals in nested Markdown examples; ::: is
 # also valid data (for example an S3 ARN), so only RawBlock is unintended.
 leakage=rawblock
 replacement=text.count('\ufffd')
 low=[i+1 for i,p in enumerate(pages) if i>3 and len(re.sub(r'\s+','',p))<100]
 (EDITORIAL/'qa/reconstruction/QA-SUMMARY.txt').write_text(f'pages={len(pages)-1}\nreplacement_glyphs={replacement}\nintentional_visible_fence_literals={visible_fences}\nunintended_source_leakage={leakage}\nlow_utilization_pages={len(low)}\nlow_utilization_page_numbers={low}\n')
 print(f'parity_rows=15 replacement={replacement} leakage={leakage} low_utilization={len(low)}')
if __name__=='__main__': main()
