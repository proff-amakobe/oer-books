"""Validate guide content and rendered equivalence; run from Generative-AI/."""
import json,re,hashlib,subprocess
from pathlib import Path
from docx import Document
from pypdf import PdfReader
from PIL import Image,ImageChops
base=Path('course-materials/blackboard'); src=base/'source'; cur=json.loads((base/'qa/curriculum.json').read_text()); evidence=json.loads((base/'qa/source-evidence.json').read_text()); rows=[]
def norm(s):
 s=re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',s);s=re.sub(r'^\s*\d+\.\s*','',s,flags=re.M);return re.sub(r'\s+',' ',s.replace('\\$','$').replace('*','').replace('`','')).strip()
combined=PdfReader(base/'qa/render/combined/Generative_AI_Weeks01-16_Module_Guides.pdf'); offset=0; pixels=[]
for w in range(1,17):
 s=(src/f'week-{w:02}.md').read_text();meta=json.loads(re.search(r'<!-- metadata: (.+) -->',s)[1]); e=evidence[w-1]
 assert meta['co']==cur['week_emphasis'][w-1]
 assert meta['pslo']==sorted({p for c in meta['co'] for p in cur['mapping'][str(c)]})
 for c in meta['co']:assert cur['co'][str(c)] in s
 for line in s.splitlines():
  if re.match(r'\| \d+ \|',line):
   co=re.search(r'CO (\d+) / PSLOs ([\d, ]+)',line);assert co
   assert [int(x) for x in co[2].split(',')]==cur['mapping'][co[1]]
 if w<16:
  assert e['introduction'] in s and e['milestone'] in s
  original=Path(e['chapter_source']).read_bytes();assert hashlib.sha256(original).hexdigest()==e['source_sha256']
  assert e['introduction'] in original.decode() and e['milestone'] in original.decode()
 count=len((src/f'week-{w:02}-start-here.md').read_text().split());assert 200<=count<=350
 copy=(src/f'week-{w:02}-blackboard-copy.md').read_text();assert '[GUIDANCE]' not in copy
 rubrics=[]
 for table in re.findall(r'(?:^\|.*\n)+',s,re.M):
  lines=table.splitlines()
  if 'Full-credit evidence' in lines[0]:
   values=[int(x.strip().strip('|').split('|')[-1]) for x in lines[2:]];assert sum(values[:-1])==values[-1];rubrics.append(values[-1])
 assert rubrics==meta['points']
 d=Document(base/f'docx/Generative_AI_Week{w:02}_Module_Guide.docx')
 text='\n'.join(''.join(p._p.xpath('.//w:t/text()')) for p in d.paragraphs)
 if w<16:assert norm(e['introduction']) in norm(text) and norm(e['milestone']) in norm(text)
 pdf=PdfReader(base/f'qa/render/week-{w:02}/Generative_AI_Week{w:02}_Module_Guide.pdf');assert 8<=len(pdf.pages)<=12
 for n,p in enumerate(pdf.pages,1):
  t=p.extract_text();assert len(t)>250;assert f'Week {w} Module Guide' in t
  assert norm(t)==norm(combined.pages[offset+n-1].extract_text()),(w,n,'combined text differs')
  a=Image.open(base/f'qa/render/week-{w:02}/page-{n}.png').convert('RGB');b=Image.open(base/f'qa/render/combined/page-{offset+n}.png').convert('RGB')
  equal=a.size==b.size and ImageChops.difference(a,b).getbbox() is None
  pixels.append({'week':w,'page':n,'combined_page':offset+n,'pixel_equal':equal})
 rows.append({'week':w,'pages':len(pdf.pages),'start_here_words':count,'points':sum(rubrics),'source_fidelity':'PASS','co_pslo':'PASS','rubric_arithmetic':'PASS','docx':'PASS'})
 offset+=len(pdf.pages)
assert offset==len(combined.pages)
assert all(r['points']==50 for r in rows) and sum(r['points'] for r in rows)==800
(base/'qa/content-validation.json').write_text(json.dumps({'weeks':rows,'rubrics':48,'weekly_points':sum(r['points'] for r in rows),'residency_points':200,'total_points':sum(r['points'] for r in rows)+200,'combined_pages':offset,'combined_pixel_matches':sum(x['pixel_equal'] for x in pixels),'page_comparison':pixels},indent=2)+'\n')
print(json.dumps({'weeks':rows,'combined_pages':offset,'pixel_matches':sum(x['pixel_equal'] for x in pixels)},indent=2))
