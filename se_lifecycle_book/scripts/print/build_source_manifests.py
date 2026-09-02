#!/usr/bin/env python3
from reconstruction_common import *
import re

def main():
 inv=[]; hrs=[]; brs=[]; fir=[]; trs=[]; horder=border=forder=torder=0
 for ch,path in enumerate(CHAPTERS,1):
  lines,heads,blocks,images,tables,raw=scan(path); title=heads[0][1]
  prose=[]; incode=False
  for line in lines:
   if re.match(r"^\s*(`{3,}|~{3,})",line): incode=not incode; continue
   if not incode and not re.match(r"^\s*(?:---|:::|<!--)",line): prose.append(plain(line))
  paragraphs=len([x for x in re.split(r"\n\s*\n","\n".join(prose)) if x.strip() and not x.lstrip().startswith(('#','|','- ','* '))])
  inv.append(dict(chapter_number=ch,chapter_title=title,source_file=str(path.relative_to(ROOT)),source_sha256=sha(path.read_bytes()),word_count=len(re.findall(r"\b[\w’'-]+\b","\n".join(prose))),**{f"H{x}_count":sum(h[0]==x for h in heads) for x in range(1,5)},paragraph_count=paragraphs,ordered_list_count=sum(bool(re.match(r"^\s*\d+[.)]\s",x)) for x in lines),unordered_list_count=sum(bool(re.match(r"^\s*[-*+]\s",x)) for x in lines),table_count=len(tables),image_count=len(images),code_block_count=len(blocks),typed_code_block_count=sum(bool(b.language) for b in blocks),untyped_code_block_count=sum(not b.language for b in blocks),raw_html_block_count=raw[0],raw_latex_block_count=raw[1],quarto_div_count=raw[2],equation_count=sum(x.count("$$") for x in lines)//2+sum(bool(re.search(r"(?<!\$)\$[^$]+\$",x)) for x in lines),cross_reference_count=sum(len(re.findall(r"@(?:fig|tbl|sec|eq)-[\w-]+",x)) for x in lines),link_count=sum(len(re.findall(r"(?<!!)\[[^]]+\]\([^)]+\)",x)) for x in lines)))
  for level,text,ident,line in heads:
   horder+=1; hrs.append(dict(chapter=ch,source_file=str(path.relative_to(ROOT)),heading_level=level,heading_text=text,identifier=ident,source_line=line,source_order=horder))
  for b in blocks:
   border+=1; meaningful=[x.strip() for x in b.text.splitlines() if x.strip()]; first=meaningful[0] if meaningful else ""; last=meaningful[-1] if meaningful else ""
   brs.append(dict(block_id=f"block-{border:04d}",chapter=ch,section=b.section,source_file=str(path.relative_to(ROOT)),source_line=b.line,language_or_class=b.language or "(untyped)",first_meaningful_line=first,last_meaningful_line=last,line_count=len(b.text.splitlines()),sha256=sha(b.text.encode()),semantic_candidate=classify(b.language,b.text,b.section)))
  for ln,section,alt,img,titleattr,attrs in images:
   forder+=1; resolved=(path.parent/img).resolve(); attr=attrs or ""; ident=re.search(r"#([\w-]+)",attr); width=re.search(r"width\s*=\s*([^\s}]+)",attr)
   fir.append(dict(figure_id=ident.group(1) if ident else f"figure-{forder:04d}",chapter=ch,section=section,source_file=str(path.relative_to(ROOT)),image_path=img,caption=titleattr or alt,alt_text=alt,declared_width=width.group(1) if width else "",format=Path(img).suffix.lstrip('.').lower(),file_exists=resolved.exists(),file_sha256=sha(resolved.read_bytes()) if resolved.exists() else ""))
  for ln,section,rows,cols in tables:
   torder+=1; trs.append(dict(table_id=f"table-{torder:04d}",chapter=ch,section=section,source_file=str(path.relative_to(ROOT)),source_line=ln,rows=rows,columns=cols,caption="",status="SOURCE"))
 write_csv(EDITORIAL/'COMPLETE-SOURCE-MANIFEST.csv',list(inv[0]),inv)
 write_csv(EDITORIAL/'COMPLETE-HEADING-MANIFEST.csv',list(hrs[0]),hrs)
 write_csv(EDITORIAL/'COMPLETE-TECHNICAL-BLOCK-MANIFEST.csv',list(brs[0]),brs)
 write_csv(EDITORIAL/'COMPLETE-FIGURE-MANIFEST.csv',list(fir[0]),fir)
 write_csv(EDITORIAL/'COMPLETE-TABLE-MANIFEST.csv',list(trs[0]),trs)
 print(f"chapters={len(inv)} headings={len(hrs)} blocks={len(brs)} figures={len(fir)} tables={len(trs)} words={sum(x['word_count'] for x in inv)}")
if __name__=='__main__': main()
