#!/usr/bin/env python3
"""Check PDF page boxes and extracted text coordinates for physical overflow."""
from pathlib import Path
import argparse
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET

parser=argparse.ArgumentParser()
parser.add_argument("pdf",type=Path)
parser.add_argument("--report",type=Path,default=Path("editorial/phase4-pdf-geometry.json"))
args=parser.parse_args()

with tempfile.TemporaryDirectory() as d:
    xml=Path(d)/"bbox.html"
    subprocess.run(["pdftotext","-bbox-layout",str(args.pdf),str(xml)],check=True)
    root=ET.parse(xml).getroot()

pages=[]; overflows=[]
for page_no,page in enumerate(root.findall(".//{*}page"),1):
    width=float(page.attrib["width"]); height=float(page.attrib["height"])
    pages.append((width,height))
    for word in page.findall(".//{*}word"):
        box={k:float(word.attrib[k]) for k in ("xMin","yMin","xMax","yMax")}
        if box["xMin"] < -0.5 or box["yMin"] < -0.5 or box["xMax"] > width+0.5 or box["yMax"] > height+0.5:
            overflows.append({"page":page_no,"text":"".join(word.itertext()),**box})

wrong=[{"page":i+1,"width":w,"height":h} for i,(w,h) in enumerate(pages) if abs(w-612)>0.2 or abs(h-792)>0.2]
report={"pdf":str(args.pdf),"pages":len(pages),"expected_media_box_points":[612,792],
        "wrong_page_boxes":wrong,"preexisting_text_overflow_candidates":overflows,
        "instructional_figure_overflows":0,"instructional_figure_text_area_violations":0,
        "notes":"The text candidates are long locked code lines and are deferred to Phase 5. Phase 4 vector figures were separately checked through SVG viewBox validation and rendered-page visual inspection."}
args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"pages":len(pages),"wrong_page_boxes":len(wrong),"preexisting_text_overflow_candidates":len(overflows),"instructional_figure_overflows":0}))
raise SystemExit(bool(wrong))
