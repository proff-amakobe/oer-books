"""Regression checks for the controlled consistency pass; preserve Phase 1 evidence."""
from pathlib import Path
import collections,csv,hashlib,json,re,zipfile
import pymupdf
from bs4 import BeautifulSoup
from audit_manuscript import parse
R=Path(__file__).resolve().parents[2];E=R/'editorial';errors=[]
def check(ok,msg):
 if not ok:errors.append(msg)
base=json.loads((E/'phase-1b-source-baseline.json').read_text())
for name,digest in json.loads((E/'phase-1b-preserved-records.json').read_text()).items():
 check(hashlib.sha256((R/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,'Accepted Phase 1 record changed: '+name)
# Reconstruct every chapter from the accepted baseline and the ordered edit log.
# Only standardized missing-artwork notices and blank-line cleanup are implicit.
log=json.loads((E/'phase-1b-edit-log.json').read_text())
for name,original in base.items():
 ch=int(Path(name).name[:2]);expected=original
 for row in log:
  if row['chapter']==ch:
   old=row['conflicting_or_overstated_claim'];new=row['replacement']
   check(old in expected,'Edit log cannot replay: '+name+' '+old[:45])
   expected=expected.replace(old,new)
 expected=re.sub(r'\*\*Illustration unavailable\.\*\*[^\n]*?(?=\[Figure pending:|\n)', '',expected)
 def clean_blank(t):return re.sub(r'\n[ \t]*\n(?:[ \t]*\n)*','\n\n',t).strip()
 check(clean_blank(expected)==clean_blank((R/name).read_text()),'Undocumented source change: '+name)

chapters=sorted((R/'chapters').glob('*.qmd'));check(len(chapters)==15,'Chapter count')
counts=collections.Counter();markers=[]
for p in chapters:
 s=p.read_text();old=base[str(p.relative_to(R))];ch=int(p.name[:2])
 check(len(re.findall(r'^## Learning Objectives$',s,re.M))==1,'Objectives count: '+p.name)
 check(not re.search(r'^#{1,6} (?:What You.ll Learn|Learning Outcomes|Objectives)$',s,re.M),'Duplicate objectives: '+p.name)
 section=s.split('## Learning Objectives\n',1)[1].split('\n## ',1)[0]
 check(not re.search(r'(?:^[-\d. *]+)(?:Understand|Know|Appreciate|Grasp)\b',section,re.M),'Vague objective verb: '+p.name)
 oh,ob,_=parse(old);nh,nb,_=parse(s)
 check(len(ob)==len(nb),'Technical count changed: '+p.name)
 def reading(t):
  a=t.index('## Further Reading');b=t.find('\n## ',a+1);return t[a:b if b!=-1 else None]
 check(reading(old)==reading(s),'Further Reading changed: '+p.name)
 for a,b in zip(ob,nb):check(a['text']==b['text'] and a['lang']==b['lang'],'Technical identity/payload changed: '+p.name)
 counts['technical_fences']+=len(nb);counts['headings_before']+=len(oh);counts['headings_after']+=len(nh)
 counts['objective_sections_before']+=len(re.findall(r'^#{1,6} (?:What You.ll Learn|Learning Outcomes|Learning Objectives)$',old,re.M))
 counts['objective_sections_after']+=1
 markers+=re.findall(r'\[Figure pending: [^\n]+\]',s)
 check('Safety, Ethics, and Responsible AI' not in s,'Old Chapter 13 title')
 check(not re.search(r'Illustration unavailable|source artwork was not supplied|\*Proposed Figure',s),'Old figure wording')
 # Instructional requirements inside verbatim examples are deliberately excluded.
 prose=re.sub(r'```.*?```','',s,flags=re.S)
 for pattern in [r'genuinely novel outputs',r'higher = more creative',r'Role assignment activates',r'writing out steps catches errors',r'proven solutions to common problems',r'both powerful and secure',r'\*\*Reasoning improves accuracy\*\*',r'Short contexts \(4K tokens\)',r'jitteris']:
  check(not re.search(pattern,prose),'Overstatement regression: '+pattern)
 for title in ['Key Terminologies and Concepts','Discussion Questions','Further Reading','Chapter Wrap-Up','Project Milestone']:
  check(title in s,'Missing pedagogical section: '+p.name+' '+title)
check(counts['technical_fences']==87,'Expected 87 literal fences plus one native worked example')
check(sum(p.read_text().count('#tech-03-014') for p in chapters)==1,'Native worked example missing')
check(len(markers)==len(set(markers))==30,'Figure marker count/uniqueness')
figures=list(csv.DictReader((E/'figure-candidates.csv').open()))
check(len({x['candidate_id'] for x in figures})==len(figures),'Duplicate candidate IDs')
check(collections.Counter(x['placeholder'] for x in figures if x['placeholder'])==collections.Counter(markers),'Figure manifest mapping')
check(len({(x['chapter'],x['proposed_visual']) for x in figures})==len(figures),'Duplicate visual proposals')
for name in ['preface.qmd','index.qmd','copyright.qmd','about-author.qmd','references.qmd']:
 check(not re.search(r'Phase [01]|review baseline|structural review|review edition|await.*approval|pending verification|source artwork|TEMPORARY', (R/name).read_text(),re.I),'Process wording in '+name)
check((R/'chapters/14-law-policy.qmd').read_text()==base['chapters/14-law-policy.qmd'].replace('## Learning Outcomes','## Learning Objectives'),'Substantive law text changed')
check(len(re.findall(r'@\w+\{', (R/'references.bib').read_text()))==97,'Bibliography count')
# Verify the actual rendered TOC, including visible number-to-title gaps.
doc=pymupdf.open(R/'output/pdf/Generative-AI-PHASE-1B-REVIEW.pdf');toc='\n'.join(p.get_text() for p in list(doc)[1:8]);gaps={}
check(not re.search(r'1[0-5]\.\d{2}[A-Za-z]',toc),'Concatenated TOC number/title')
for p in list(doc)[1:8]:
 words=p.get_text('words')
 for i,w in enumerate(words[:-1]):
  if re.fullmatch(r'1[0-5]\.\d{2}',w[4]):
   nxt=words[i+1]
   if abs(w[1]-nxt[1])<2 and nxt[4][0].isalpha():gaps[w[4]]=round(nxt[0]-w[2],2)
for prefix in ['10.10','10.11','10.12','11.10','12.10','13.10','14.10','15.10']:
 check(prefix in gaps and gaps[prefix]>=4,'TOC gap missing or too narrow: '+prefix)
check(len(gaps)==28,'Not all double-digit TOC entries inspected')
check(all(g>=4 for g in gaps.values()),'TOC gap below four points')
check(all(abs(p.rect.width-612)<.1 and abs(p.rect.height-792)<.1 for p in doc),'PDF trim')
for p in doc:check(not re.search(r'jitteris',p.get_text()),'Rendered jitter spacing')
rendered=[('PDF','\n'.join(p.get_text() for p in doc))]
with zipfile.ZipFile(R/'output/epub/Generative-AI.epub') as z:
 rendered.append(('EPUB',' '.join(BeautifulSoup(z.read(n),'html.parser').get_text(' ',strip=True) for n in z.namelist() if n.endswith('.xhtml'))))
rendered.append(('HTML',' '.join(BeautifulSoup(p.read_text(),'html.parser').get_text(' ',strip=True) for p in (R/'output/html').rglob('*.html') if 'site_libs' not in p.parts)))
for fmt,s in rendered:
 check('Safety, Ethics, and Responsible AI' not in s,fmt+' old Chapter 13 title')
 check('Ethics and Responsible AI' in s,fmt+' missing Chapter 13 title')
 check('structural review baseline' not in s and 'awaits author approval' not in s,fmt+' stale review prose')
 check(not re.search(r'What You[’\']ll Learn|Learning Outcomes',s),fmt+' old objective heading')
audit=list(csv.DictReader((E/'PHASE-1B-CONSISTENCY-AUDIT.csv').open()))
check(all(x['status'] in ['FIXED','CONSISTENT'] for x in audit),'Unresolved consistency finding')
counts.update(figures=len(markers),figure_candidates=len(figures),pdf_pages=len(doc),technical_examples=88,toc_defects_before=len(json.loads((E/'phase-1b-toc-before.json').read_text())),toc_defects_after=0 if len(gaps)==28 and all(g>=4 for g in gaps.values()) else 1)
result=dict(status='FAIL' if errors else 'PASS',counts=dict(counts),findings=dict(collections.Counter(x['status'] for x in audit)),toc_gaps_pt=gaps,errors=errors)
(E/'phase-1b-qa-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(bool(errors))
