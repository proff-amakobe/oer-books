"""Report-only inventory of the locked manuscript and supplied images."""
from pathlib import Path
import csv,hashlib,re,struct,xml.etree.ElementTree as ET,json
from lock_original import verify
R=Path(__file__).resolve().parents[1];E=R/'editorial';verify()
def write(name,rows,fields):
 with (E/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
refs=[];chapters=[];queue=[]
for p in sorted((R/'original').glob('*.qmd')):
 text=p.read_text();chapter=p.name;section='';headings=[]
 for n,line in enumerate(text.splitlines(),1):
  if re.match(r'^#{1,6} ',line):section=line.lstrip('# ').strip();headings.append(section)
  for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)|\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',line):
   path=match[2] or match[3];source=R/'original/images'/path.split('images/',1)[-1]
   refs.append(dict(chapter=chapter,section=section,source_line=n,referenced_path=path,source=str(source.relative_to(R)),caption=match[1] or '',exists=source.is_file(),status='RESOLVED_BY_BUILD_ALIAS' if source.is_file() else 'MISSING'))
  category=None
  if re.search(r'GPT-4|Claude 3|4K tokens|100K\+|\$0\.03|\$47,000',line):category='FRESHNESS'
  elif re.search(r'Role assignment activates|genuinely novel|reasoning improves accuracy|Diagnostic accuracy|matched expert physician',line,re.I):category='TECHNICAL'
  elif re.search(r'\*\*(?:Accuracy|Diagnostic accuracy)\*\*.*\d|Improved by 34%',line):category='CITATION'
  if category and len(line)<1600:
   queue.append(dict(chapter=chapter,section=section,source_line=n,category=category,observation=line.strip(),severity='REVIEW',recommended_author_action='Author may wish to review the scope, date, and supporting source for this statement. No change has been applied.'))
 goals=[h for h in headings if h in ["What You'll Learn",'Learning Outcomes','Learning Objectives']]
 if len(goals)>1:queue.append(dict(chapter=chapter,section='Chapter opening',source_line='',category='CONSISTENCY',observation='Multiple learning-goal sections: '+', '.join(goals),severity='REVIEW',recommended_author_action='Author may wish to review whether the repeated goal sections are intentional. Preserved unchanged.'))
 title=next(h for h in headings if h.startswith('Chapter '));metadata=re.search(r'^title:\s*"([^"]+)"',text,re.M)
 if metadata and metadata[1]!=title:queue.append(dict(chapter=chapter,section='Chapter title',source_line=2,category='CONSISTENCY',observation=f'YAML title: {metadata[1]}; body heading: {title}',severity='REVIEW',recommended_author_action='Author may wish to review the differing titles. Both source strings are preserved; no heading is rewritten.'))
 chapters.append(dict(order=len(chapters)+1,filename=p.name,relative_path=str(p.relative_to(R)),body_title=title,metadata_title=metadata[1] if metadata else '',byte_size=p.stat().st_size))
images=[]
for p in sorted((R/'original/images').rglob('*')):
 if p.suffix.lower() not in ['.svg','.png','.jpg','.jpeg','.gif','.webp']:continue
 b=p.read_bytes();dims=''
 if b.startswith(b'\x89PNG'):dims='%s × %s'%struct.unpack('>II',b[16:24])
 elif b.startswith(b'\xff\xd8'):
  import pymupdf
  pix=pymupdf.Pixmap(b);dims=f"{pix.width} × {pix.height}"
 elif p.suffix=='.svg':
  e=ET.fromstring(b);dims=e.get('width','')+' × '+e.get('height','')+'; viewBox '+e.get('viewBox','')
 found=[x for x in refs if x['source']==str(p.relative_to(R))]
 images.append(dict(filename=p.name,relative_path=str(p.relative_to(R)),format='JPEG (filename .png)' if b.startswith(b'\xff\xd8') else p.suffix[1:].upper(),pixel_dimensions=dims,file_size=len(b),sha256=hashlib.sha256(b).hexdigest(),referenced_by_chapter='; '.join(x['chapter'] for x in found),referenced_in_source='; '.join(str(x['source_line'])+': '+x['referenced_path'] for x in found),exists='YES',notes='Used through byte-identical build resource alias' if found else 'Supplied asset not referenced by the original chapters; retained, not inserted.'))
for image in images:
 if image['format'].startswith('JPEG'):
  queue.append(dict(chapter=image['referenced_by_chapter'] or 'Unreferenced asset',section='Image asset',source_line=image['referenced_in_source'],category='IMAGE',observation=image['relative_path']+' has a .png filename but JPEG data.',severity='REVIEW',recommended_author_action='Author may wish to review the filename in a future approved revision. Original bytes are unchanged; the build uses a byte-identical .jpg alias.'))
write('ORIGINAL-CHAPTER-INVENTORY.csv',chapters,list(chapters[0]))
write('ORIGINAL-IMAGE-INVENTORY.csv',images,list(images[0]))
write('ORIGINAL-IMAGE-REFERENCES.csv',refs,list(refs[0]))
write('MISSING-ORIGINAL-IMAGES.csv',[{k:x[k] for k in ['chapter','section','referenced_path','caption','status']} for x in refs if not x['exists']],['chapter','section','referenced_path','caption','status'])
write('AUTHOR-REVIEW-QUEUE.csv',queue,['chapter','section','source_line','category','observation','severity','recommended_author_action'])
print(json.dumps(dict(chapters=len(chapters),images=len(images),image_references=len(refs),missing_images=sum(not x['exists'] for x in refs),author_review_candidates=len(queue)),indent=2))
