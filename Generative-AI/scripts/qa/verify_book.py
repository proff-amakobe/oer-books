#!/usr/bin/env python3
"""Fail closed on missing chapter content, broken links, or edition leakage."""
import collections, hashlib, json, os, re, shutil, subprocess, sys, unicodedata, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit
from bs4 import BeautifulSoup
from pypdf import PdfReader
import pymupdf
from audit_manuscript import parse
ROOT=Path(__file__).resolve().parents[2]
ED=ROOT/'editorial'; WEB=ROOT/'output/html'
PANDOC=os.environ.get('PANDOC') or shutil.which('pandoc')
if not PANDOC:
    quarto=Path(shutil.which('quarto')).resolve()
    for path in [quarto.parent/'tools/pandoc',Path('/Applications/quarto/bin/tools/pandoc'), *sorted(quarto.parent.glob('tools/*/pandoc'))]:
        if path.exists():PANDOC=str(path);break
assert PANDOC, 'Install Pandoc or set PANDOC to the Quarto bundled executable'
errors=[];stats=collections.Counter(); details=[]
def check(ok,message):
    if not ok: errors.append(message)
def norm(s):
    s=unicodedata.normalize('NFKC',s).replace('\u00ad','').replace('↪','')
    return re.sub(r'\s+','',s).replace('’',"'").replace('‘',"'").replace('“','"').replace('”','"').replace('–','-').replace('—','-')
def words(s): return ''.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFKC',s),re.U)).lower()
def soup(text):return BeautifulSoup(text,'html.parser')
def pdf_symbols(t):
    for a,b in {'👍':'[thumbs up]','👎':'[thumbs down]','→':'->','↑':'[up]','↓':'[down]','├':'|','└':'+','─':'-'}.items():t=t.replace(a,b)
    return t

baseline={Path(k).name:v for k,v in json.loads((ED/'phase-0-review-baseline.json').read_text()).items()}
authorized=json.loads((ED/'phase-1-content-changes.json').read_text())
chapters=sorted((ROOT/'chapters').glob('[0-9][0-9]-*.qmd'))
check(len(chapters)==15,'Expected exactly 15 source chapters')
epub_path=ROOT/'output/epub/Generative-AI.epub'
with zipfile.ZipFile(epub_path) as z:
    check(z.testzip() is None,'EPUB corrupt ZIP member')
    for name in z.namelist():
        if name.endswith(('.xhtml','.opf','.ncx')): ET.fromstring(z.read(name))
    check(z.read('mimetype')==b'application/epub+zip','EPUB mimetype incorrect')
    epubs={name:soup(z.read(name).decode()) for name in z.namelist() if name.endswith('.xhtml')}
    opf=next(z.read(name).decode() for name in z.namelist() if name.endswith('.opf'))
    check('en-US' in opf,'EPUB language missing')
    check(any(s.select('nav') for s in epubs.values()),'EPUB navigation missing')
    epub_all='\n'.join(s.get_text(' ',strip=True) for s in epubs.values())
    # Check archive-local links and fragment targets.
    names=set(z.namelist())
    for name,s in epubs.items():
        for a in s.select('[href], [src]'):
            href=a.get('href',a.get('src','')); u=urlsplit(href)
            if u.scheme or u.netloc:continue
            target=os.path.normpath(str(Path(name).parent/unquote(u.path))) if u.path else name
            check(target in names,f'EPUB missing target {name}: {href}')
            if u.fragment and target in epubs:
                check(epubs[target].find(id=unquote(u.fragment)) is not None,f'EPUB broken fragment {name}: {href}')
reader=PdfReader(ROOT/'output/pdf/Generative-AI-PHASE-1-REVIEW.pdf')
check(all(abs(float(p.mediabox.width)-612)<.1 and abs(float(p.mediabox.height)-792)<.1 for p in reader.pages),'PDF not 612 x 792 pt on every page')
doc=pymupdf.open(ROOT/'output/pdf/Generative-AI-PHASE-1-REVIEW.pdf')
pdftext='\n'.join(p.get_text(clip=pymupdf.Rect(0,0,612,735),sort=False) for p in doc);pdfnorm=norm(pdftext);pdfwords=words(pdftext)
stats['pdf_pages']=len(reader.pages)
for forbidden in ['Read Online','Download PDF','Download EPUB','View Source on GitHub','What the book covers','Five Parts, fifteen chapters']:
    check(forbidden not in pdftext,'Web landing leaked into PDF: '+forbidden)
    check(forbidden not in epub_all,'Web landing leaked into EPUB: '+forbidden)
# Every heading, table cell, code payload, and substantive paragraph is checked.
for path in chapters:
    ch=int(path.name[:2]);source=path.read_text(); html_path=WEB/'chapters'/path.with_suffix('.html').name
    check(html_path.exists(),f'HTML chapter missing {ch}')
    page=soup(html_path.read_text());main=page.select_one('main');check(main is not None,f'No HTML main {ch}')
    for tag in main.select('.anchorjs-link, .code-copy-button'):tag.decompose()
    expected=soup(subprocess.check_output([PANDOC,'-f','markdown','-t','html','--citeproc','--bibliography',str(ROOT/'references.bib'),'-M','nocite=@*',str(path)],text=True))
    for refsdiv in expected.select('#refs'): refsdiv.decompose()
    # Quarto disambiguates author names across the complete book; standalone
    # Pandoc disambiguates only this chapter. Verify citation keys and links,
    # then use the book's labels when comparing otherwise exact prose.
    for citation in expected.select('span.citation'):
        cited=citation.get('data-cites','')
        if cited.startswith('sec-'):
            citation.replace_with('@'+cited)
            continue
        actual=main.find('span',attrs={'data-cites':cited})
        check(actual is not None,f'HTML citation keys missing {ch}: {cited}')
        if actual is not None:citation.replace_with(actual.get_text(' ',strip=True))

    epub=next((s for s in epubs.values() if s.find(id=f'sec-ch{ch:02}')),None)
    check(epub is not None,f'EPUB chapter missing {ch}');epub=epub or soup('')
    h1=main.find('h1');number=h1.select_one('.chapter-number') if h1 else None
    check(number is not None and number.get_text(strip=True)==str(ch),f'HTML chapter number wrong {ch}')
    check(len(main.find_all('h1'))==1,f'HTML multiple H1s in {ch}')
    htext=words(main.get_text(' ',strip=True));etext=words(epub.get_text(' ',strip=True))
    for h in expected.find_all(re.compile('^h[1-6]$')):
        text=h.get_text(' ',strip=True); n=words(text)
        for fmt,hay in [('HTML',htext),('EPUB',etext),('PDF',pdfwords)]:
            check(n in hay,f'{fmt} missing heading {ch}: {text}')
        stats['headings_checked']+=1
    # Independently compare line-based headings with Pandoc's semantic headings.
    def heading_words(text):
        text=re.sub(r'\s*\{[^}]*\}\s*$', '', text)
        text=re.sub(r'^Chapter \d+:\s*|^\d+\.\d+\s+', '', text)
        return words(text)
    original_heads,_,_=parse(baseline[path.name])
    source_heads,_,_=parse(source)
    check(len(source_heads)==len(expected.find_all(re.compile('^h[1-6]$'))),f'Unexpected Markdown heading interpretation {ch}')
    parsed_heads=[heading_words(h.get_text(' ',strip=True)) for h in expected.find_all(re.compile('^h[1-6]$'))]
    for _,title,_ in original_heads:
        check(heading_words(next((r['new'] for r in authorized['headings'] if r['chapter']==ch and r['old']==title),title)) in parsed_heads, f'Substantive source heading lost {ch}: {title}')
    # Payload fidelity against originals: one documented conversion from verbatim to native math.
    _,oldblocks,_=parse(baseline[path.name]);_,newblocks,_=parse(source)
    check(len(oldblocks)==len(newblocks),f'Source technical block count changed {ch}')
    for old,new in zip(oldblocks,newblocks):
        ident=re.search(r'#([\w-]+)',new['lang'])[1]
        allowance=next((r for r in authorized['technical_payloads'] if r['id']==ident),None)
        valid=old['text']==new['text']
        if allowance:
            valid=(hashlib.sha256(old['text'].encode()).hexdigest()==allowance['old_sha256'] and hashlib.sha256(new['text'].encode()).hexdigest()==allowance['new_sha256'])
        check(valid,f'Unreviewed source technical payload change {ident}')
    for pre in expected.find_all('pre'):
        payload=pre.get_text();ident=pre.get('id') or (pre.parent.get('id') if pre.parent else None);n=norm(payload)
        for fmt,rendered in [('HTML',main),('EPUB',epub)]:
            found=rendered.find(id=ident) if ident else None
            check(found is not None and n==norm(found.get_text()),f'{fmt} technical payload mismatch {ch}: {ident}')
        # Page wrapping is ignored; every original payload line must survive PDF extraction.
        for line in pdf_symbols(payload).splitlines():
            if not line.strip():continue
            check(norm(line) in pdfnorm,f'PDF missing technical line {ident}: {line[:90]}')
        stats['technical_blocks_checked']+=1
    for table in expected.find_all('table'):
        for cell in table.select('th,td'):
            n=words(cell.get_text(' ',strip=True))
            for fmt,hay in [('HTML',htext),('EPUB',etext),('PDF',pdfwords)]:
                check(n in hay,f'{fmt} table cell missing {ch}: {cell.get_text()[:70]}')
        stats['tables_checked']+=1
    # Compare complete prose paragraphs; resolve source xrefs through actual HTML link labels.
    refs={a.get('href','').split('#')[-1]:a.get_text(' ',strip=True) for a in main.select('a.quarto-xref')}
    for para in expected.find_all('p'):
        if para.find_parent(['td','th']):continue
        text=para.get_text(' ',strip=True)
        if len(text)<40 or '$$' in text:continue
        text=re.sub(r'@([\w-]+)',lambda m:refs.get(m[1],m[0]),text)
        n=words(text)
        for fmt,hay in [('HTML',htext),('EPUB',etext),('PDF',pdfwords)]:
            check(n in hay,f'{fmt} paragraph missing {ch}: {text[:100]}')
        stats['paragraphs_checked']+=1
    for line in source.splitlines():
        if '*Proposed Figure ' in line:
            text=re.search(r'\*Proposed Figure .*',line).group().strip('* \\');n=words(text)
            for fmt,hay in [('HTML',htext),('EPUB',etext),('PDF',pdfwords)]:check(n in hay,f'{fmt} figure description lost: {text}')
            stats['figure_descriptions_checked']+=1
    # Intentional Markdown inside pre is exempt; readers must see it literally.
    clean=soup(str(main))
    for pre in clean.select('pre,code,script,style'):pre.decompose()
    check(not re.search(r'```|(?:^|\s)#{1,6} [A-Z]|\{#sec-|(?m:^:::) |\\begin\{wrapfigure\}|(?m:^title:)',clean.get_text('\n')),f'HTML source leakage {ch}')
    epub_clean=soup(str(epub))
    for tag in epub_clean.select('pre,code,script,style,math'):tag.decompose()
    check(not re.search(r'```|(?:^|\s)#{1,6} [A-Z]|\{#sec-|(?m:^:::) |\\begin\{wrapfigure\}|(?m:^title:)',epub_clean.get_text('\n')),f'EPUB source leakage {ch}')
    stats['chapters_checked']+=1
# Bibliography and native math must not silently disappear in combined formats.
for key in re.findall(r'@\w+\{([^,]+)',(ROOT/'references.bib').read_text()):
    check(soup((WEB/'references.html').read_text()).find(id='ref-'+key) is not None,'HTML bibliography entry missing '+key)
    check(any(s.find(id='ref-'+key) for s in epubs.values()),'EPUB bibliography entry missing '+key)
for title in re.findall(r'title = \{\{(.+)\}\}',(ROOT/'references.bib').read_text()):
    check(words(title) in pdfwords,'PDF bibliography entry missing '+title)
mathpage=soup((WEB/'chapters/03-prompt-engineering.html').read_text())
check(len(mathpage.select('.math.display'))==2,'Expected two native HTML display expressions')
check(sum(len(s.find_all('math')) for s in epubs.values())>=2,'Native EPUB MathML missing')
check('240' in pdftext and '0.15' in pdftext and '36 + 23 = 59' in pdftext,'PDF worked calculation missing')
# Local links/assets including fragments across all rendered pages.
pages={p:soup(p.read_text()) for p in WEB.rglob('*.html') if 'site_libs' not in p.parts}
for p,s in pages.items():
    check(s.find('link',rel='canonical') is not None,f'Missing canonical URL {p.name}')
    for a in s.select('[href], [src]'):
        href=a.get('href',a.get('src',''));u=urlsplit(href)
        if u.scheme or u.netloc or href.startswith('data:'):continue
        if not u.path:target=p
        elif u.path.startswith('/'):target=WEB/unquote(u.path).lstrip('/')
        else:target=(p.parent/unquote(u.path)).resolve()
        if target.is_dir():target=target/'index.html'
        if not target.exists():stats['broken_local_links']+=1;errors.append(f'HTML missing target {p.name}: {href}');continue
        if u.fragment and target in pages and not pages[target].find(id=unquote(u.fragment)):
            stats['broken_local_links']+=1;errors.append(f'HTML missing fragment {p.name}: {href}')
check(len(pages[WEB/'index.html'].find_all('h1'))==1,'Landing page title duplicated')
for name in ['search.json','sitemap.xml','robots.txt','downloads/Generative-AI-PHASE-1-REVIEW.pdf','downloads/Generative-AI.epub']:
    check((WEB/name).exists(),'Missing web resource '+name)
search=(WEB/'search.json').read_text()
for p in chapters:check(p.with_suffix('.html').name in search,'Chapter absent from search '+p.name)
# High-confidence credential/PII patterns; never print potential credential values.
patterns=[r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}',r'\bAKIA[A-Z0-9]{16}\b',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\bgh[pousr]_[A-Za-z0-9]{30,}',r'https?://[^\s/@]+:[^\s/@]+@',r'\b\d{3}-\d{2}-\d{4}\b']
for p in chapters:
    for pattern in patterns:check(not re.search(pattern,p.read_text()),'Potential secret/PII requires review: '+p.name)
# Detect text outside physical PDF page boundaries; visually inspect representative pages separately.
for i,p in enumerate(doc):
    for block in p.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                check(x0>=-1 and y0>=-1 and x1<=613 and y1<=793,f'PDF text outside media box on page {i+1}')
                check('\ufffd' not in span['text'],f'PDF replacement glyph on page {i+1}')
for key in ['missing_headings','missing_technical_blocks','missing_tables','missing_figures','raw_source_leakage','broken_local_links','missing_rendered_assets','secrets_detected']:
    stats.setdefault(key,0)
result={'status':'PASS' if not errors else 'FAIL','counts':dict(stats),'errors':errors,'preexisting_missing_artwork':20,'figure_proposals_without_supplied_art':30,'limits':['Build completeness is distinct from scholarly review; see the Phase 1 report and adjudicated records.','EPUB ZIP/navigation/content validation is not a formal EPUBCheck certification.','Missing artwork is visibly disclosed, not reconstructed.']}
(ED/'qa-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'counts':result['counts'],'error_count':len(errors),'first_errors':errors[:30]},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
