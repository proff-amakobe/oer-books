#!/usr/bin/env python3
"""Validate Phase 1 review provenance, classifications, sources, and edition metadata."""
import csv, json, re, hashlib, collections, zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
R=Path(__file__).resolve().parents[2]; E=R/'editorial'; errors=[];counts={}
def check(ok,msg):
    if not ok:errors.append(msg)
def read(name):return list(csv.DictReader((E/name).open()))
allowed={'freshness-items':set('CURRENT_AND_DURABLE CURRENT_BUT_TIME_SENSITIVE OUTDATED UNVERIFIED ILLUSTRATIVE_ONLY REMOVE_OR_GENERALIZE NEEDS_AUTHOR_REVIEW'.split()),'citation-needed':set('CITATION_REQUIRED CITATION_HELPFUL COMMON_KNOWLEDGE ILLUSTRATIVE REMOVE_OR_GENERALIZE NEEDS_VERIFICATION'.split())}
for name,n in [('freshness-items',447),('citation-needed',510)]:
    old=read(name+'-phase-0.csv');new=read(name+'.csv');check(len(old)==len(new)==n,name+' count')
    for a,b in zip(old,new):
        for k,v in a.items():check(b.get('phase_0_'+k if k in ['status','verification'] else k)==v,f'{name} provenance changed: {b.get("review_id")} {k}')
        check(b['classification'] in allowed[name],name+' invalid classification')
        check(b['verification_date']=='2026-09-18',name+' undated decision')
        check(bool(b['notes']) and bool(b['correction']),name+' missing rationale')
        check(b['resolution']=='RESOLVED',name+' unresolved candidate')
    check(len({r['review_id'] for r in new})==n,name+' duplicate IDs')
    counts[name]=dict(collections.Counter(r['classification'] for r in new))
examples=read('technical-example-review.csv');check(len(examples)==88,'technical example count')
check(len({x['id'] for x in examples})==88,'duplicate example IDs')
check(all(x['classification'] in 'PSEUDOCODE STATIC_EXAMPLE RUNNABLE_OFFLINE RUNNABLE_WITH_DEPENDENCIES REQUIRES_API STRUCTURED_DATA PROMPT MODEL_RESPONSE PROGRAM_OUTPUT'.split() for x in examples),'invalid technical classification')
check(all(x['execution_status']=='NOT_EXECUTED' for x in examples if x['classification']=='PSEUDOCODE'),'pseudocode reported as executed')
check(all(x['execution_status']=='PASS' for x in examples if x['classification']=='RUNNABLE_OFFLINE'),'offline example failure')
readings=read('further-reading-audit.csv');check(len(readings)==50,'Further Reading inventory changed')
check(all(x['status'] in ['VERIFIED','CORRECTED','REMOVED','REPLACED'] and x['source'].startswith('https://') for x in readings),'unverified reading entry')
sources=json.loads((E/'phase-1-sources.json').read_text());bib=(R/'references.bib').read_text();bibkeys=set(re.findall(r'@\w+\{([^,]+)',bib));sourcekeys={s['bib_key']:s for s in sources.values()}
for k in bibkeys:check(k in sourcekeys and sourcekeys[k]['status']=='VERIFIED',f'Unverified bibliography: {k}')
for p in (R/'chapters').glob('*.qmd'):
    for k in re.findall(r'@([A-Za-z][\w-]+)',p.read_text()):check(k.startswith(('sec-','fig-','tbl-')) or k in bibkeys,'Unresolved source citation '+k)
subtitle='Foundations, Systems, Evaluation, and Responsible Deployment'
for name in ['_quarto.yml','copyright.qmd','README.md','styles/book-metadata.html','editorial/publication-metadata.json']:
    check(subtitle in (R/name).read_text(),'Subtitle absent: '+name)
landing=BeautifulSoup((R/'output/html/index.html').read_text(),'html.parser')
check(subtitle in landing.get_text(),'HTML subtitle missing')
ld=landing.find('script',type='application/ld+json');check(ld and subtitle in ld.get_text(),'JSON-LD subtitle missing')
og=landing.find('meta',property='og:title');check(og and subtitle in og.get('content',''),'OpenGraph subtitle missing')
with zipfile.ZipFile(R/'output/epub/Generative-AI.epub') as z:
    opf=next(z.read(n).decode() for n in z.namelist() if n.endswith('.opf'));check(subtitle in opf,'EPUB metadata subtitle missing')
pdf=PdfReader(R/'output/pdf/Generative-AI-PHASE-1B-REVIEW.pdf');check(subtitle.replace(' ','') in ''.join(p.extract_text() for p in pdf.pages[:3]).replace('\n','').replace(' ',''),'PDF title page subtitle missing')
check(all(abs(float(p.mediabox.width)-612)<.1 and abs(float(p.mediabox.height)-792)<.1 for p in pdf.pages),'PDF geometry')
counts.update(bibliography=len(bibkeys),further_reading=len(readings),technical_examples=dict(collections.Counter(x['classification'] for x in examples)),case_studies=len(read('case-study-audit.csv')),law_statements=len(read('law-policy-audit.csv')))
result={'status':'FAIL' if errors else 'PASS','verification_date':'2026-09-18','counts':counts,'errors':errors}
(E/'phase-1-qa-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(bool(errors))
