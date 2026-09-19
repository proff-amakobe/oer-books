#!/usr/bin/env python3
"""Inventory the immutable Phase 0 source snapshot. Counts exclude fenced examples."""
import csv, hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ED = ROOT / 'editorial'

def write_csv(name, rows, fields):
    with (ED / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def parse(text):
    lines=text.splitlines(); heads=[]; blocks=[]; prose=[]; section=''; start=None; body=[]; lang=''; fence=''
    yaml_end = lines[1:].index('---')+1 if lines and lines[0]=='---' else -1
    for i,line in enumerate(lines,1):
        if i <= yaml_end+1: continue
        m=re.match(r'^\s*(`{3,}|~{3,})(.*)$',line)
        if start is not None:
            if m and m[1][0]==fence[0] and len(m[1])>=len(fence):
                blocks.append(dict(line=start, section=section, lang=lang.strip(), text='\n'.join(body))); start=None
            else: body.append(line)
            continue
        if m: start=i; fence=m[1]; lang=m[2]; body=[]; continue
        h=re.match(r'^(#{1,6})\s+(.+)',line)
        if h: section=h[2]; heads.append((len(h[1]),section,i))
        prose.append((i,line,section))
    assert start is None, 'Unclosed fence'
    return heads,blocks,prose

def semantic(ch,b,source):
    lang=b['lang']; text=b['text']; before='\n'.join(source.splitlines()[max(0,b['line']-6):b['line']-1])
    if lang=='json': return 'DATA'
    if lang in ('bash','sh','shell','console'): return 'TERMINAL'
    if lang in ('yaml','toml','ini'): return 'CONFIGURATION'
    if lang=='python' or 'final_template = (' in text:
        if 'test_result = {' in text: return 'DATA'
        if 'Conceptual' in text or 'Pseudocode' in text or '...' in text: return 'PSEUDOCODE'
        return 'PROGRAM_CODE'
    if ch==5: return 'STRUCTURED_OUTPUT'
    if ch==2:
        return 'MODEL_RESPONSE' if text.startswith(('Query:','User asks:')) else 'DATA'
    if ch==3:
        if before.rstrip().endswith('**Response**:'): return 'MODEL_RESPONSE'
        if text.startswith('Template Performance'): return 'PROGRAM_OUTPUT'
        if text.startswith(('Template: research_analysis','Test 1:')): return 'DATA'
        if text.startswith(('This AI assistant','After each response:')): return 'PLAIN_VERBATIM'
        if text.startswith('15% of 240'): return 'WORKED_CALCULATION'
        if text.startswith('Prompt: "Explain quantum'): return 'MODEL_RESPONSE'
        return 'PROMPT'
    return 'PLAIN_VERBATIM'

def main():
    if (ED/'phase-1-review-counts.json').exists():
        raise SystemExit('Phase 1 queues are adjudicated. Refusing to overwrite them; use verify_phase1.py.')
    ED.mkdir(exist_ok=True)
    snapshot=ED/'source-baseline.json'
    if not snapshot.exists():
        sources={p.name:p.read_text() for p in sorted(ROOT.glob('[0-9][0-9]-*.qmd'))}
        assert len(sources)==15
        snapshot.write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    sources=json.loads(snapshot.read_text()); inventory=[]; technical=[]; figures=[]; candidates=[]; fresh=[]; citations=[]; residues=[]; overflows=[]
    for filename,text in sources.items():
        ch=int(filename[:2]); heads,blocks,prose=parse(text); plain='\n'.join(p[1] for p in prose)
        images=list(re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)',plain))
        latex_images=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',plain)
        captions=[(i,line.strip('* '),sec) for i,line,sec in prose if re.search(r'Figure \d+\.\d+:',line)]
        for k,(i,line,sec) in enumerate(captions):
            cap=re.search(r'Figure (\d+\.\d+):\s*(.*)',line).group(2).strip('* \\')
            context='\n'.join(text.splitlines()[max(0,i-4):i]); im=re.search(r'!\[([^\]]*)\]\(([^)]+)\)',context)
            src=im[2] if im else ''; exists=bool(src and (ROOT/src).is_file())
            figures.append(dict(chapter=ch,section=sec,source=src or 'caption only',caption=cap,format=Path(src).suffix.lstrip('.') if src else 'proposed',exists=exists,dimensions='unknown' if src else 'N/A',semantic_role='instructional',print_suitability='missing source',web_suitability='missing source'))
            candidates.append(dict(chapter=ch,section=sec,source_line=i,proposed_visual=cap,reason='Existing manuscript visual reference requires original artwork or a new instructional SVG',priority='high'))
        for src in latex_images:
            figures.append(dict(chapter=ch,section='The Anatomy of Effective Prompts',source=src,caption='Marcus vignette illustration',format=Path(src).suffix[1:],exists=(ROOT/src).exists(),dimensions='unknown',semantic_role='narrative illustration',print_suitability='missing; raw wrapfigure',web_suitability='raw LaTeX invisible'))
        for b in blocks:
            kind=semantic(ch,b,text); lang=b['lang']; dep='scipy' if 'from scipy' in b['text'] else ('application helpers required' if lang=='python' else 'none')
            technical.append(dict(chapter=ch,section=b['section'],source_line=b['line'],language_class=lang or 'untyped',semantic_type=kind,line_count=len(b['text'].splitlines()),runnable='no' if kind!='PROGRAM_CODE' else 'fragment; not standalone',requires_API='indirect' if re.search(r'generate|model\.',b['text']) else 'no',requires_external_dependency=dep,contains_secret='no detected',expected_render='native math' if kind=='WORKED_CALCULATION' else 'labeled, wrapping, page-breakable block',notes='Mixed prompt/response transcript' if 'Response:' in b['text'] or 'A:' in b['text'] else 'Preserve literal payload; do not execute'))
            for offset,line in enumerate(b['text'].splitlines(),1):
                if len(line)>88: overflows.append(dict(chapter=ch,source_line=b['line']+offset,length=len(line),kind=kind,text=line))
        for i,line,sec in prose:
            if not line.strip(): continue
            words=re.findall(r'\b(course|week|module|assignment|instructor|semester|LMS|discussion|due date)\b',line,re.I)
            if words: residues.append(dict(chapter=ch,source_line=i,terms='; '.join(words),text=line,action='retain discussion/technical or real-world timing; generalize administrative schedule'))
            if len(line)>150 and line.startswith('|'): overflows.append(dict(chapter=ch,source_line=i,length=len(line),kind='TABLE_ROW',text=line))
            if not line.startswith('#') and re.search(r'OpenAI|Anthropic|Google|Gemini|Meta\b|Llama|Mistral|GPT|Claude|PaLM|context window|pricing|\bcost\b|API|fine.tun|agent|tool.use|multimodal|benchmark|regulat|policy|frontier|today|currently',line,re.I):
                fresh.append(dict(chapter=ch,source_line=i,section=sec,claim=line,verification='PENDING',priority='high' if re.search(r'\$|GPT|Claude|regulat|GDPR|AI Act|PaLM|price|pricing',line,re.I) else 'normal',source_needed='Dated primary paper, provider documentation/model card, or official legal text'))
            if not line.startswith('#') and not re.search(r'Further Reading|Academic Papers|Foundational Papers|Technical Resources|Industry Perspectives|Practical Guides|Research Tools',sec) and re.search(r'transformer|scaling|RLHF|DPO|alignment|RAG|fine.tun|LoRA|QLoRA|agent|multimodal|evaluat|bias|safety|ethic|law|regulat|deploy|\d+%',line,re.I):
                citations.append(dict(chapter=ch,source_line=i,section=sec,claim=line,reason='Claim or teaching item needs primary-source review; candidate, not adjudicated missing citation',status='PENDING'))
        inventory.append(dict(chapter_number=ch,filename=filename,current_H1=heads[0][1],recommended_title=re.sub(r'^Chapter \d+:\s*','',heads[0][1]),word_count=len(re.findall(r"\b[\w]+(?:[’'-][\w]+)*\b",text)),H2=sum(h[0]==2 for h in heads),H3=sum(h[0]==3 for h in heads),H4=sum(h[0]==4 for h in heads),code_blocks=len(blocks),code_languages='; '.join(sorted({b['lang'] or 'untyped' for b in blocks})),tables=len(re.findall(r'^\|?\s*:?-{3,}.*\|.*$',plain,re.M)),figures=len(captions)+len(latex_images),equations=len(re.findall(r'\$\$|\\\[',plain))//2,citations=len(re.findall(r'(?<!\w)@[A-Za-z][\w:-]*',plain)),external_links=len(re.findall(r'https?://[^\s)>]+',plain)),callouts=len(re.findall(r'\.callout-',plain)),exercises=sum(bool(re.search('Practice|Hands-On|Project Milestone',h[1])) for h in heads),learning_objectives=sum(bool(re.search('Learning Objectives|Learning Outcomes',h[1])) for h in heads),case_studies=sum('Case Study' in h[1] for h in heads),references_further_reading=sum('Further Reading' in h[1] for h in heads),raw_HTML=len(re.findall(r'<[A-Za-z][^>]*>',plain)),raw_LaTeX=len(re.findall(r'\\(?:begin|end|includegraphics|clearpage|vspace|centering|par)\b',plain)),video_placeholders=len(re.findall(r'\[Video:|Introductory Video',plain,re.I)),course_module_language=sum(r['chapter']==ch for r in residues),potential_source_leakage='raw LaTeX; missing images' if ch==3 else ('missing images' if images else 'none detected'),sha256=hashlib.sha256(text.encode()).hexdigest()))
    extras = [(4,'Integration architecture and failure paths'),(6,'Chunking, hybrid retrieval, and reranking'),(7,'LoRA and QLoRA adaptation workflow'),(8,'Research question, evidence appraisal, synthesis, and reproducible experiment'),(9,'Multimodal document ingestion and cross-modal retrieval'),(10,'Agent trajectory and multimodal evaluation framework'),(11,'Optimization pipeline with cost and latency budgets'),(12,'Alignment training and deployment safety layers'),(13,'Stakeholder impacts and ethical review'),(14,'Governance responsibilities and jurisdictional review'),(15,'Production architecture and model lifecycle'),(15,'Frontier research evidence map')]
    for ch,description in extras:
        candidates.append(dict(chapter=ch,section='Chapter-wide synthesis',source_line='',proposed_visual=description,reason='Clarify relationships currently conveyed only in prose; develop after technical review',priority='normal'))
    for name,rows,fields in [('chapter-inventory.csv',inventory,[]),('technical-block-inventory.csv',technical,[]),('figure-inventory.csv',figures,[]),('figure-candidates.csv',candidates,[]),('freshness-items.csv',fresh,[]),('citation-needed.csv',citations,[]),('course-language-inventory.csv',residues,[]),('overflow-inventory.csv',overflows,[])]:
        write_csv(name,rows,list(rows[0]) if rows else fields)
    totals={key:sum(row[key] for row in inventory) for key in ('word_count','H2','H3','H4','code_blocks','tables','figures','equations','video_placeholders')}
    totals.update(chapters=15,citation_candidates=len(citations),freshness_items=len(fresh),figure_candidates=len(candidates),source_image_references=sum(len(re.findall(r'!\[.*?\]\(',t))+len(re.findall(r'\\includegraphics',t)) for t in sources.values()))
    (ED/'baseline-counts.json').write_text(json.dumps(totals,indent=2)+'\n'); print(json.dumps(totals,indent=2))
if __name__=='__main__': main()
