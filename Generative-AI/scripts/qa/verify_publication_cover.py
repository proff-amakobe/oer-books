#!/usr/bin/env python3
"""Verify publication cover packaging and front-matter fidelity, independently of originals."""
from pathlib import Path
import hashlib,json,re,unicodedata,zipfile,xml.etree.ElementTree as ET
import pymupdf
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[2];O=R/'output/reset';cover=R/'assets/cover/generative-ai-cover.png'
def norm(s):return ''.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFKC',s).lower(),re.U))
pix=pymupdf.Pixmap(cover);assert (pix.width,pix.height)==(2550,3300) and not pix.alpha
pdf=pymupdf.open(O/'pdf/Generative-AI-ORIGINAL-MANUSCRIPT-REVIEW.pdf');assert all(p.rect.width==612 and p.rect.height==792 for p in pdf)
assert not pdf[0].get_text().strip(),'Cover must be unnumbered, without figure caption'
ims=pdf[0].get_images();assert len(ims)==1 and ims[0][2:4]==(2550,3300)
embedded=pymupdf.Pixmap(pdf,ims[0][0]);assert embedded.samples==pix.samples,'PDF cover pixels changed'
pdftext=norm(' '.join(p.get_text() for p in pdf))
with zipfile.ZipFile(O/'epub/Generative-AI-ORIGINAL-MANUSCRIPT.epub') as z:
 assert z.testzip() is None
 opfname=next(n for n in z.namelist() if n.endswith('.opf'));opf=ET.fromstring(z.read(opfname));ns={'o':'http://www.idpf.org/2007/opf'}
 items=opf.findall('o:manifest/o:item',ns);covers=[x for x in items if 'cover-image' in x.get('properties','').split()];assert len(covers)==1
 target=str(Path(opfname).parent/covers[0].get('href'));assert z.read(target)==cover.read_bytes()
 meta=opf.find('o:metadata/o:meta[@name="cover"]',ns)
 if meta is not None:assert meta.get('content')==covers[0].get('id')
 pages=[BeautifulSoup(z.read(n),'html.parser') for n in z.namelist() if n.endswith('.xhtml')]
 epubtext=norm(' '.join(p.get_text(' ',strip=True) for p in pages))
 assert any(p.find('nav') for p in pages)
 assert any((im.get('src') or im.get('xlink:href') or im.get('href','')).endswith(Path(target).name) for p in pages for im in p.find_all(['img','image']))
for name in ['about-author','acknowledgments','preface']:
 source=(R/(name+'.qmd')).read_text();html=BeautifulSoup((O/'html'/(name+'.html')).read_text(),'html.parser');htext=norm(html.find('main').get_text(' ',strip=True))
 paragraphs=[p for p in source.split('\n\n') if p.strip() and not p.startswith('#')]
 for p in paragraphs:
  expected=norm(p)
  for kind,text in [('HTML',htext),('PDF',pdftext),('EPUB',epubtext)]:assert expected in text,(name,kind,p[:50])
result={'status':'PASS','cover_dimensions':[pix.width,pix.height],'cover_sha256':hashlib.sha256(cover.read_bytes()).hexdigest(),'pdf_pages':len(pdf),'pdf_cover':'exact pixels, unnumbered','epub_cover':'recognized, exact PNG bytes','frontmatter_paragraphs':'PASS'}
(O/'qa/w2').mkdir(parents=True,exist_ok=True)
(O/'qa/w2/publication-cover.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
