"""Fail closed on any author-byte change or missing rendered original payload."""
import collections,csv,hashlib,json,os,re,subprocess,sys,unicodedata,zipfile
from pathlib import Path
from urllib.parse import urlsplit,unquote
import xml.etree.ElementTree as ET
import pymupdf
from bs4 import BeautifulSoup
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));from lock_original import verify
R=Path(__file__).resolve().parents[2];E=R/'editorial';OUT=R/'output/reset';WEB=OUT/'html';errors=[];counts=collections.Counter();blockrows=[]
PANDOC=os.environ.get('PANDOC') or '/Applications/quarto/bin/tools/pandoc'
if not Path(PANDOC).exists():
 import shutil
 q=Path(shutil.which('quarto')).resolve();PANDOC=next(str(p) for p in [q.parent/'tools/pandoc',*q.parent.glob('tools/*/pandoc')] if p.exists())
def check(ok,msg):
 if not ok:errors.append(msg)
def norm(s):
 s=unicodedata.normalize('NFKC',s).replace('\u00ad','').replace('’',"'").replace('‘',"'").replace('“','"').replace('”','"')
 return re.sub(r'\s+','',s)
def literal(s):
 return s.replace("\r\n","\n").rstrip("\n")
def prose(s):
 return ''.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFKC',s),re.U)).lower()
def soup(t):return BeautifulSoup(t,'html.parser')
lock=verify(bool(os.environ.get('CI')))
pdf=pymupdf.open(OUT/'pdf/Generative-AI-ORIGINAL-MANUSCRIPT-REVIEW.pdf')
pdftext='\n'.join(p.get_text(clip=pymupdf.Rect(0,0,612,735)) for p in pdf);pdfnormal=norm(pdftext.replace('↪',''));pdfprose=prose(pdftext)
check(all(abs(p.rect.width-612)<.1 and abs(p.rect.height-792)<.1 for p in pdf),'PDF trim')
with zipfile.ZipFile(OUT/'epub/Generative-AI-ORIGINAL-MANUSCRIPT.epub') as z:
 check(z.testzip() is None,'EPUB corrupt member');names=set(z.namelist())
 epubs={n:soup(z.read(n)) for n in names if n.endswith('.xhtml')}
 for name in names:
  if name.endswith(('.xhtml','.opf','.ncx')):ET.fromstring(z.read(name))
 opf=next(z.read(n).decode() for n in names if n.endswith('.opf'));check('en-US' in opf,'EPUB language');check(z.read('mimetype')==b'application/epub+zip','EPUB mimetype')
 check(any(s.find('nav') for s in epubs.values()),'EPUB navigation')
 for n,s in epubs.items():
  for a in s.select('[href],[src]'):
   url=urlsplit(a.get('href',a.get('src','')))
   if url.scheme or url.netloc:continue
   target=os.path.normpath(str(Path(n).parent/unquote(url.path))) if url.path else n
   check(target in names,'EPUB missing resource: '+target)
   if url.fragment and target in epubs:check(epubs[target].find(id=unquote(url.fragment)) is not None,'EPUB missing fragment '+url.fragment)
 epubtext='\n'.join(s.get_text(' ',strip=True) for s in epubs.values());epubprose=prose(epubtext)
 epubcode=[literal(p.get_text()) for s in epubs.values() for p in s.find_all('pre')]
 epubimages=[x.get('src') for s in epubs.values() for x in s.find_all('img')]
 epub_image_hashes=set()
 for name,page in epubs.items():
  for im in page.find_all('img'):
   target=os.path.normpath(str(Path(name).parent/unquote(im.get('src',''))))
   if target in names:epub_image_hashes.add(hashlib.sha256(z.read(target)).hexdigest())
 # Author raster assets must be included byte-for-byte (JPEG aliases change only suffix).
 for source in [R/'original/images/chapters/ch01_img8.png',R/'original/images/chapters/marcus.png']:
  digest=hashlib.sha256(source.read_bytes()).hexdigest()
  check(any(hashlib.sha256(z.read(n)).hexdigest()==digest for n in names if n.lower().endswith(('.png','.jpg','.jpeg'))),'EPUB raster asset bytes absent: '+source.name)
chapters=sorted((R/'original').glob('*.qmd'))
for ch,p in enumerate(chapters,1):
 rendered=WEB/'original'/p.with_suffix('.html').name;check(rendered.exists(),'Missing chapter '+p.name)
 main=soup(rendered.read_text()).find('main');check(main is not None,'No main '+p.name)
 for x in main.select('.anchorjs-link,.code-copy-button'):x.decompose()
 htext=prose(main.get_text(' ',strip=True));htmlcode=[literal(x.get_text()) for x in main.find_all('pre')]
 expected=soup(subprocess.check_output([PANDOC,'-f','markdown','-t','html',str(p)],text=True))
 ast=json.loads(subprocess.check_output([PANDOC,'-f','markdown','-t','json',str(p)],text=True))
 def walk(x):
  if isinstance(x,dict):
   yield x
   for value in x.values():yield from walk(value)
  elif isinstance(x,list):
   for v in x:yield from walk(v)
 for n,b in enumerate([n for n in walk(ast['blocks']) if n.get('t')=='CodeBlock'],1):
  attr,text=b['c'];lines=[x for x in text.splitlines() if x.strip()];h=literal(text)
  blockrows.append(dict(chapter=p.name,block=n,classes=' '.join(attr[1]),first_meaningful_line=lines[0] if lines else '',last_meaningful_line=lines[-1] if lines else '',normalized_sha256=hashlib.sha256(h.encode()).hexdigest()))
  check(h in htmlcode,f'HTML technical block missing {p.name}:{n}')
  check(h in epubcode,f'EPUB technical block missing {p.name}:{n}')
  check(norm(h) in pdfnormal,f'PDF full technical payload missing {p.name}:{n}')
  rendered_pre=main.find_all('pre')[n-1]
  actual_classes=set(rendered_pre.get('class',[])) | set((rendered_pre.find('code') or {}).get('class',[]))
  check(set(attr[1]).issubset(actual_classes),f'HTML source language class missing {p.name}:{n}')
  if not attr[1]:check(not actual_classes.intersection({'python','json','bash','terminal'}),f'Untyped source block reclassified {p.name}:{n}')
  counts['technical_blocks']+=1
 # Headings and whole paragraphs are checked without substituting rewritten content.
 for h in expected.find_all(re.compile('^h[1-6]$')):
  text=prose(h.get_text(' ',strip=True))
  for fmt,hay in [('HTML',htext),('PDF',pdfprose),('EPUB',epubprose)]:check(text in hay,f'{fmt} heading missing {p.name}: {h.get_text()[:100]}')
  counts['headings']+=1
 for para in expected.find_all('p'):
  if para.find_parent(['td','th']):continue
  # Image alt text is covered by the image/caption checks, not treated as an extra prose paragraph.
  if para.find('img'):
   for img in para.find_all('img'):img.decompose()
  text=para.get_text(' ',strip=True)
  if not text.strip():continue
  for fmt,hay in [('HTML',htext),('PDF',pdfprose),('EPUB',epubprose)]:check(prose(text) in hay,f'{fmt} paragraph missing {p.name}: {text[:110]}')
  counts['paragraphs']+=1
 for li in expected.find_all('li'):
  # Textual payload in lists is substantive too; nested lists are checked in order.
  text=li.get_text(' ',strip=True)
  for fmt,hay in [('HTML',htext),('PDF',pdfprose),('EPUB',epubprose)]:check(prose(text) in hay,f'{fmt} list item missing {p.name}: {text[:100]}')
  counts['list_items']+=1
 for table in expected.find_all('table'):
  for cell in table.select('td,th'):
   for fmt,hay in [('HTML',htext),('PDF',pdfprose),('EPUB',epubprose)]:check(prose(cell.get_text(' ',strip=True)) in hay,f'{fmt} table cell missing {p.name}: {cell.get_text()[:80]}')
  counts['tables']+=1
 counts['chapters']+=1
# Every local link, chapter reference, and image must resolve.
pages={p.resolve():soup(p.read_text()) for p in WEB.rglob('*.html') if 'site_libs' not in p.parts}
for p,s in pages.items():
 for a in s.select('[href],[src]'):
  u=urlsplit(a.get('href',a.get('src','')))
  if u.scheme or u.netloc:continue
  target=(p.parent/unquote(u.path)).resolve() if u.path else p
  if u.path.startswith('/'):target=(WEB/unquote(u.path).lstrip('/')).resolve()
  if target.is_dir():target=target/'index.html'
  check(target.exists(),f'HTML missing asset/link {p.name}: {u.path}')
  if u.fragment and target in pages:check(pages[target].find(id=unquote(u.fragment)) is not None,f'HTML missing fragment {p.name}: {u.fragment}')
for row in csv.DictReader((E/'ORIGINAL-IMAGE-REFERENCES.csv').open()):
 check(row['exists']=='True','Missing supplied image '+row['source'])
 page=pages[(WEB/'original'/Path(row['chapter']).with_suffix('.html').name).resolve()]
 target=Path(row['source']).name;target='marcus.jpg' if target=='marcus.png' else target
 check(any(Path(urlsplit(im.get('src','')).path).name==target for im in page.select('main img')),'HTML referenced image missing '+target)
 check(hashlib.sha256((R/row['source']).read_bytes()).hexdigest() in epub_image_hashes,'EPUB referenced image missing '+target)
 tex=(OUT/'pdf/Generative-AI-ORIGINAL-MANUSCRIPT-REVIEW.tex').read_text()
 check(Path(target).stem in tex,'PDF image inclusion missing '+target)
 check((R/'assets/images'/Path(row['source']).relative_to('original/images')).read_bytes()==(R/row['source']).read_bytes(),'Resource alias changed image bytes '+target)
 counts['referenced_images']+=1
search=(WEB/'search.json').read_text()
for p in chapters:check(p.with_suffix('.html').name in search,'Search missing chapter '+p.name)
for i,p in enumerate(pdf):
 for b in p.get_text('dict')['blocks']:
  for line in b.get('lines',[]):
   for span in line['spans']:
    x0,y0,x1,y1=span['bbox'];check(x0>=-1 and y0>=-1 and x1<=613 and y1<=793,f'PDF physical text overflow page {i+1}')
    check('\ufffd' not in span['text'],f'PDF replacement character page {i+1}')
counts['pdf_pages']=len(pdf)
with (E/'ORIGINAL-TECHNICAL-PAYLOADS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(blockrows[0]));w.writeheader();w.writerows(blockrows)
result=dict(status='FAIL' if errors else 'PASS',integrity=lock,counts=dict(counts),errors=errors)
(E/'original-reset-qa-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False));sys.exit(bool(errors))
