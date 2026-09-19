#!/usr/bin/env python3
"""Render immutable originals; stage only byte-identical resource aliases."""
import argparse,os,shutil,subprocess,hashlib,urllib.request,csv
from pathlib import Path
from lock_original import verify
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--format',choices=['all','html','pdf','epub'],default='all');args=p.parse_args()
ci=bool(os.environ.get('CI'));print(verify(ci),flush=True)
# Source uses ../assets/images/, /assets/images/, and a raw-LaTeX assets path.
# This generated alias resolves all three without changing any original bytes.
assets=ROOT/'assets/images';assets.mkdir(parents=True,exist_ok=True)
for source in (ROOT/'original/images').rglob('*'):
 if source.is_file() and source.name!='.DS_Store':
  target=assets/source.relative_to(ROOT/'original/images');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  if source.suffix.lower()==".png" and source.read_bytes().startswith(b"\xff\xd8"):
   shutil.copyfile(source,target.with_suffix(".jpg"))
profiles=['html','print','epub'] if args.format=='all' else [{'pdf':'print'}.get(args.format,args.format)]
if 'print' in profiles:
 fonts=ROOT/'output/reset/fonts';fonts.mkdir(parents=True,exist_ok=True)
 symbols=fonts/'NotoSansSymbols2-Regular.ttf'
 if not symbols.exists():
  urllib.request.urlretrieve('https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssymbols2/NotoSansSymbols2-Regular.ttf',symbols)
 assert hashlib.sha256(symbols.read_bytes()).hexdigest()=='7d5fb73b7ca67a6798101741f5d280a3d016a56a197afcd4199dbb57b4b82a21','Unexpected font dependency bytes'
 candidates=[Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'),Path('/usr/local/Caskroom/miniconda/base/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf')]
 mono=next((p for p in candidates if p.exists()),None)
 if mono is None:raise RuntimeError('Install fonts-dejavu-core or provide DejaVuSansMono.ttf in the documented font path')
 shutil.copyfile(mono,fonts/mono.name)

try:
 for profile in profiles:
  subprocess.run(['quarto','render','--profile',profile],cwd=ROOT,check=True)
  if profile=='print':
   for name in ['Generative-AI-ORIGINAL-MANUSCRIPT-REVIEW.tex','index.tex']:
    if (ROOT/name).exists():shutil.move(ROOT/name,ROOT/'output/reset/pdf'/name)
 subprocess.run(['python3',str(ROOT/'scripts/assemble_web.py')],cwd=ROOT,check=True)
finally:
 # Quarto can leave generated HTML beside an input if rendering fails.
 # These paths are not in the locked source set; retain them outside original/.
 for source in (ROOT/"original").glob("*.qmd"):
  generated=source.with_suffix(".html")
  if generated.exists() and str(generated.relative_to(ROOT)) not in {r['relative_path'] for r in csv.DictReader((ROOT/'editorial/ORIGINAL-MANUSCRIPT-LOCK.csv').open())}:
   destination=ROOT/"output/reset/intermediates";destination.mkdir(parents=True,exist_ok=True)
   shutil.move(generated,destination/generated.name)
 print(verify(ci),flush=True)
