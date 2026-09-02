#!/usr/bin/env python3
"""Render full-book and per-chapter JPEG contact sheets using Poppler + Swift/AppKit."""
import re,subprocess,tempfile,sys,shutil
from pathlib import Path
from reconstruction_common import ROOT,PDF,CHAPTERS,norm

OUT=ROOT/'editorial/qa/reconstruction'
SWIFT=r'''
import AppKit
import Foundation
let args=CommandLine.arguments
let output=args[1], cols=Int(args[2])!, rows=Int(args[3])!, files=Array(args.dropFirst(4))
let tw=220, th=285
let canvas=NSImage(size:NSSize(width:tw*cols,height:th*rows))
canvas.lockFocus()
NSColor.white.setFill(); NSRect(x:0,y:0,width:tw*cols,height:th*rows).fill()
for (i,f) in files.enumerated() {
 if let im=NSImage(contentsOfFile:f) {
  let x=(i % cols)*tw, y=(rows-1-i/cols)*th
  im.draw(in:NSRect(x:x,y:y,width:tw,height:th),from:NSRect.zero,operation:.copy,fraction:1)
 }
}
canvas.unlockFocus()
if let tiff=canvas.tiffRepresentation, let rep=NSBitmapImageRep(data:tiff), let jpg=rep.representation(using:.jpeg,properties:[.compressionFactor:0.72]) { try! jpg.write(to:URL(fileURLWithPath:output)) }
'''
def main():
 global OUT
 pdf=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else PDF
 OUT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else OUT
 OUT.mkdir(parents=True,exist_ok=True)
 pages=subprocess.run(['pdftotext','-layout',str(pdf),'-'],capture_output=True,text=True,check=True).stdout.split('\f')
 starts=[]; cursor=15
 for p in CHAPTERS:
  title=next(x[2:] for x in p.read_text().splitlines() if x.startswith('# '))
  needle=norm(title)
  found=next((i+1 for i,x in enumerate(pages[cursor:],cursor) if needle in norm(x)),None)
  if not found: raise SystemExit(f'chapter start not found: {title}')
  starts.append(found); cursor=found
 ends=[x-1 for x in starts[1:]]+[len(pages)-1]
 with tempfile.TemporaryDirectory(prefix='se-contact-') as td:
  td=Path(td); subprocess.run(['pdftoppm','-jpeg','-r','38',str(pdf),str(td/'p')],check=True,stdout=subprocess.DEVNULL)
  imgs=sorted(td.glob('p-*.jpg'),key=lambda p:int(p.stem.split('-')[-1]))
  sf=td/'contact.swift'; sf.write_text(SWIFT); binary=td/'contact'
  subprocess.run(['swiftc',str(sf),'-o',str(binary)],check=True)
  def mosaic(dest, subset,cols=5,rows=5):
   if dest.exists(): return
   subprocess.run([str(binary),str(dest),str(cols),str(rows),*[str(x) for x in subset]],check=True)
  for i in range(0,len(imgs),25): mosaic(OUT/f'book-{i//25+1:02d}.jpg',imgs[i:i+25])
  for ch,(a,b) in enumerate(zip(starts,ends),1):
   subset=imgs[a-1:b]; cols=5; rows=max(1,(len(subset)+cols-1)//cols); mosaic(OUT/f'chapter-{ch:02d}.jpg',subset,cols,rows)
  shutil.copyfile(OUT/'chapter-03.jpg',OUT/'chapter-03-uml.jpg')
  shutil.copyfile(OUT/'chapter-04.jpg',OUT/'chapter-04-design-patterns.jpg')
  glyph_pages=[93,190,246,302,392,473,544,644,657]
  leakage_pages=[314,637,638,639,640,689,690,691,692,693]
  # Old-page regression sheets remain useful locators after repagination; the
  # resolved checklist maps them to source anchors for final verification.
  mosaic(OUT/'glyph-regression-pages.jpg',[imgs[min(len(imgs)-1,max(0,n-1))] for n in glyph_pages],5,2)
  mosaic(OUT/'source-leakage-regression.jpg',[imgs[min(len(imgs)-1,max(0,n-1))] for n in leakage_pages],5,2)
 print(f'book_sheets={(len(pages)+24)//25} chapter_sheets=15 path={OUT}')
if __name__=='__main__': main()
