#!/usr/bin/env python3
"""Bundle downloads and add a canonical URL to each generated HTML page."""
import re,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WEB=ROOT/'output/html'
if WEB.exists():
    for src in ['pdf/Generative-AI-PHASE-1-REVIEW.pdf','epub/Generative-AI.epub']:
        source=ROOT/'output'/src
        if source.exists():
            (WEB/'downloads').mkdir(exist_ok=True);shutil.copy2(source,WEB/'downloads'/source.name)
    for page in WEB.rglob('*.html'):
        if 'site_libs' in page.parts:continue
        text=page.read_text();text=re.sub(r'<link rel="canonical"[^>]*>\s*','',text)
        url='https://proff-amakobe.github.io/oer-books/Generative-AI/'+page.relative_to(WEB).as_posix()
        if page.name=='index.html':url=url[:-len('index.html')]
        text=text.replace('</head>',f'<link rel="canonical" href="{url}">\n</head>')
        page.write_text(text)
