#!/usr/bin/env python3
"""Render the editions sequentially, then assemble the portable web download bundle."""
import argparse, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--format',choices=['all','html','pdf','epub'],default='all');args=p.parse_args()
profiles=['print','epub','html'] if args.format=='all' else [{'pdf':'print'}.get(args.format,args.format)]
for profile in profiles:
    subprocess.run(['quarto','render','--profile',profile],cwd=ROOT,check=True)
    if profile=='print' and (ROOT/'Generative-AI-REVIEW.tex').exists():
        shutil.move(ROOT/'Generative-AI-REVIEW.tex',ROOT/'output/pdf/Generative-AI-REVIEW.tex')
subprocess.run(['python3',str(ROOT/'scripts/assemble_web.py')],cwd=ROOT,check=True)
