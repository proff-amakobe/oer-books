#!/usr/bin/env python3
"""Build and audit the official-template Second Edition Ingram wrap cover."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPS = Path("/tmp/aca-cover-pydeps")
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from reportlab.lib.colors import CMYKColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

PAGE_W, PAGE_H = 1512.0, 864.0
COVER_LEFT, COVER_BOTTOM = 180.792, 54.0
COVER_RIGHT, COVER_TOP = 1512.0, 864.0
BACK_SAFE = (199.0, 72.0, 793.0, 846.0)
SPINE_SAFE = (806.5, 72.0, 886.5, 846.0)
FRONT_SAFE = (900.0, 72.0, 1494.0, 846.0)
BACK_TRIM = (189.792, 63.0, 801.792, 855.0)
SPINE_TRIM = (801.792, 63.0, 891.0, 855.0)
FRONT_TRIM = (891.0, 63.0, 1503.0, 855.0)
BARCODE = (653.0, 86.0, 743.5, 157.5)
TEMPLATE = ROOT / "print/templates/9798182721110-Perfect.pdf"
FINAL = ROOT / "print/covers/Advanced-Computational-Algorithms-Second-Edition-Ingram-Cover.pdf"
QA = ROOT / "editorial/qa/cover"
TMP = Path("/tmp/aca-ingram-cover")
BARCODE_SOURCE = TMP / "official-barcode-source.png"

NAVY = CMYKColor(0.94, 0.76, 0.36, 0.54)
NAVY_2 = CMYKColor(0.91, 0.68, 0.26, 0.40)
CYAN = CMYKColor(0.76, 0.05, 0.08, 0.0)
TEAL = CMYKColor(0.73, 0.0, 0.28, 0.0)
GOLD = CMYKColor(0.02, 0.22, 0.80, 0.0)
WHITE = CMYKColor(0, 0, 0, 0)
PALE = CMYKColor(0.14, 0.04, 0.0, 0.0)
MUTED = CMYKColor(0.25, 0.08, 0.0, 0.08)


def register_fonts() -> None:
    base = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Arial", str(base / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(base / "Arial Bold.ttf")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_bbox(log: list[dict], name: str, panel: str, bbox, critical=True, kind="text"):
    x0, y0, x1, y1 = [round(float(v), 3) for v in bbox]
    safe = {"back": BACK_SAFE, "spine": SPINE_SAFE, "front": FRONT_SAFE}[panel]
    clearances = [x0-safe[0], y0-safe[1], safe[2]-x1, safe[3]-y1]
    log.append({"name": name, "panel": panel, "kind": kind, "critical": critical,
                "bbox_pt": [x0, y0, x1, y1],
                "minimum_safe_clearance_pt": round(min(clearances), 3),
                "minimum_safe_clearance_in": round(min(clearances)/72, 3),
                "pass": (not critical) or min(clearances) >= 7.2})


def draw_paragraph(c, text, x, top, width, style, log, name, panel):
    p = Paragraph(text, style)
    w, h = p.wrap(width, PAGE_H)
    y = top - h
    p.drawOn(c, x, y)
    add_bbox(log, name, panel, (x, y, x + w, top))
    return y


def draw_artwork(path: Path, barcode_path: Path) -> list[dict]:
    register_fonts()
    log: list[dict] = []
    c = Canvas(str(path), pagesize=(PAGE_W, PAGE_H), pageCompression=1,
               initialFontName="Arial", pdfVersion=(1, 3))
    c.setTitle("Advanced Computational Algorithms - Second Edition - Ingram Cover")
    c.setAuthor("Moody Amakobe")
    c.setSubject("Print cover; ISBN 979-8-1827-2111-0")
    c.setFillColor(WHITE); c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(NAVY); c.rect(COVER_LEFT, COVER_BOTTOM, COVER_RIGHT-COVER_LEFT, COVER_TOP-COVER_BOTTOM, fill=1, stroke=0)

    # Continuous computational geometry motif, deliberately non-critical across folds.
    c.setStrokeColor(NAVY_2); c.setLineWidth(1.0)
    for x in range(210, 1513, 54):
        c.line(x, 54, min(x + 390, 1512), 864)
    for y in range(92, 865, 66):
        c.line(181, y, 1512, min(864, y + 126))
    c.setStrokeColor(CYAN); c.setLineWidth(1.4)
    nodes = [(250,170),(340,250),(430,205),(520,340),(620,285),(710,410),(825,355),
             (950,210),(1040,310),(1140,260),(1240,390),(1345,325),(1450,470),
             (970,610),(1090,710),(1220,650),(1370,760),(1480,690)]
    for a, b in zip(nodes, nodes[1:]): c.line(a[0], a[1], b[0], b[1])
    for x, y in nodes:
        c.setFillColor(CYAN); c.circle(x, y, 4.2, fill=1, stroke=0)
        c.setFillColor(NAVY); c.circle(x, y, 1.7, fill=1, stroke=0)
    c.setFillColor(TEAL); c.setFillAlpha(0.22)
    c.circle(1320, 560, 190, fill=1, stroke=0); c.circle(1120, 420, 110, fill=1, stroke=0)
    c.setFillAlpha(1)

    # Front panel.
    c.setFillColor(GOLD); c.rect(936, 796, 86, 6, fill=1, stroke=0)
    c.setFillColor(WHITE); c.setFont("Arial-Bold", 41)
    c.drawString(936, 732, "ADVANCED")
    c.drawString(936, 682, "COMPUTATIONAL")
    c.drawString(936, 632, "ALGORITHMS")
    title_width = max(pdfmetrics.stringWidth(s, "Arial-Bold", 41) for s in ("ADVANCED", "COMPUTATIONAL", "ALGORITHMS"))
    add_bbox(log, "front title", "front", (936, 624, 936+title_width, 774))
    c.setFillColor(PALE); c.setFont("Arial", 17)
    subtitle = "Concepts, Complexity, and Applied Projects"
    c.drawString(938, 582, subtitle)
    add_bbox(log, "front subtitle", "front", (938, 578, 938+pdfmetrics.stringWidth(subtitle,"Arial",17), 596))
    c.setFillColor(GOLD); c.setFont("Arial-Bold", 15)
    c.drawString(938, 540, "SECOND EDITION")
    add_bbox(log, "edition statement", "front", (938, 537, 938+pdfmetrics.stringWidth("SECOND EDITION","Arial-Bold",15), 554))
    c.setFillColor(WHITE); c.setFont("Arial-Bold", 20)
    c.drawString(938, 474, "MOODY AMAKOBE")
    add_bbox(log, "front author", "front", (938, 470, 938+pdfmetrics.stringWidth("MOODY AMAKOBE","Arial-Bold",20), 491))
    c.setFillColor(PALE); c.setFont("Arial-Bold", 10.5)
    c.drawString(938, 104, "GLOBAL DATA SCIENCE INSTITUTE")
    add_bbox(log, "front publisher", "front", (938, 102, 938+pdfmetrics.stringWidth("GLOBAL DATA SCIENCE INSTITUTE","Arial-Bold",10.5), 115))

    # Back panel.
    head = ParagraphStyle("head", fontName="Arial-Bold", fontSize=17, leading=20, textColor=WHITE, spaceAfter=7)
    body = ParagraphStyle("body", fontName="Arial", fontSize=9.0, leading=12.2, textColor=PALE, alignment=TA_LEFT)
    small = ParagraphStyle("small", fontName="Arial", fontSize=8.7, leading=11.8, textColor=PALE)
    c.setFillColor(GOLD); c.rect(235, 799, 72, 5, fill=1, stroke=0)
    y = draw_paragraph(c, "Algorithms for the problems that matter", 235, 778, 470, head, log, "back headline", "back")
    desc = ("Advanced Computational Algorithms provides a rigorous yet practical guide to the design, analysis, implementation, and evaluation of modern algorithms.<br/><br/>"
            "The Second Edition develops mathematical foundations while connecting theory to executable implementations and real-world computational problems. Topics include asymptotic analysis, divide-and-conquer methods, advanced data structures, greedy and dynamic programming techniques, randomized and approximation algorithms, computational complexity and NP-completeness, graph algorithms, string processing, numerical methods, and modern algorithm engineering.<br/><br/>"
            "Worked examples, pseudocode, executable code, instructional diagrams, proofs, exercises, benchmarking activities, and applied projects help readers move from understanding algorithms to evaluating how they behave in practice.<br/><br/>"
            "Designed for advanced undergraduate and graduate students, instructors, software engineers, data scientists, researchers, and technical professionals, the book emphasizes correctness, complexity, performance evaluation, reproducibility, and computational problem solving.")
    draw_paragraph(c, desc, 235, y-10, 500, body, log, "back description", "back")
    c.setFillColor(CYAN); c.rect(235, 374, 44, 3, fill=1, stroke=0)
    bio_head = ParagraphStyle("biohead", fontName="Arial-Bold", fontSize=11.5, leading=14, textColor=WHITE)
    y2 = draw_paragraph(c, "ABOUT THE AUTHOR", 235, 357, 350, bio_head, log, "author bio heading", "back")
    bio = ("Moody Amakobe is a multidisciplinary researcher, data scientist, and blockchain architect whose work spans artificial intelligence, computational systems, data engineering, and distributed architectures. As founder of the Global Data Science Institute, he leads work in data science, deep learning, algorithm design, and technology for impact. He has taught at multiple universities, authored technology textbooks, advised research teams, supervised graduate projects, and developed industry solutions across healthcare, finance, agriculture, and public systems.")
    draw_paragraph(c, bio, 235, y2-5, 390, small, log, "author biography", "back")
    oer = ("OPEN EDUCATIONAL RESOURCE · CC BY 4.0<br/>Designed to support accessible teaching, independent study, and lifelong learning.<br/><br/>"
           "Published by Global Data Science Institute")
    draw_paragraph(c, oer, 235, 176, 350, small, log, "OER and publisher statement", "back")

    bx0, by0, bx1, by1 = BARCODE
    c.drawImage(str(barcode_path), bx0, by0, width=bx1-bx0, height=by1-by0, preserveAspectRatio=False, mask=None)
    add_bbox(log, "official Ingram barcode", "back", BARCODE, kind="barcode")

    # Spine: US convention, readable top-to-bottom when the front lies face-up.
    c.saveState(); c.translate(850, 160); c.rotate(90)
    c.setFillColor(WHITE); c.setFont("Arial-Bold", 13.2)
    spine_text = "ADVANCED COMPUTATIONAL ALGORITHMS  •  MOODY AMAKOBE"
    c.drawString(0, 0, spine_text)
    sw = pdfmetrics.stringWidth(spine_text, "Arial-Bold", 13.2)
    c.restoreState()
    add_bbox(log, "spine title and author", "spine", (847, 160, 863, 160+sw))
    c.setFillColor(GOLD); c.circle(850, 113, 7, fill=1, stroke=0)
    add_bbox(log, "spine publisher mark", "spine", (843, 106, 857, 120))
    c.showPage(); c.save()
    return log


def merge_with_template(artwork: Path, output: Path) -> None:
    """Publish the clean artwork PDF; template geometry has already governed layout.

    Copying the template's XMP/output-intent object graph into an independently
    generated PDF produces a file that Poppler accepts but macOS CoreGraphics and
    some Acrobat configurations reject as damaged. The artwork itself is already
    PDF 1.3, CMYK, one page, exact-size, and fully font-embedded.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artwork, output)


def render_pdf(pdf: Path, prefix: Path, dpi=150) -> Path:
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", str(dpi), str(pdf), str(prefix)], check=True)
    return prefix.with_suffix(".png")


def crop_box(im: Image.Image, box, scale):
    x0,y0,x1,y1=box
    return im.crop((round(x0*scale), round((PAGE_H-y1)*scale), round(x1*scale), round((PAGE_H-y0)*scale)))


def make_qa(final_render: Path) -> None:
    QA.mkdir(parents=True, exist_ok=True)
    im = Image.open(final_render).convert("RGB")
    scale = im.width/PAGE_W
    crop_box(im, FRONT_TRIM, scale).save(QA/"front-cover-preview.png", dpi=(150,150))
    crop_box(im, BACK_TRIM, scale).save(QA/"back-cover-preview.png", dpi=(150,150))
    crop_box(im, SPINE_TRIM, scale).save(QA/"spine-preview.png", dpi=(150,150))
    thumb = crop_box(im, FRONT_TRIM, scale); thumb.thumbnail((250, 400)); thumb.save(QA/"front-cover-thumbnail.png")
    ImageOps.grayscale(im).save(QA/"grayscale-proof.png", dpi=(150,150))

    proof = im.copy(); d = ImageDraw.Draw(proof, "RGBA")
    def rect(box, color, width=4):
        x0,y0,x1,y1=box; d.rectangle((x0*scale,(PAGE_H-y1)*scale,x1*scale,(PAGE_H-y0)*scale), outline=color, width=width)
    for box in (BACK_SAFE,SPINE_SAFE,FRONT_SAFE): rect(box,(0,255,80,255),5)
    for box in (BACK_TRIM,SPINE_TRIM,FRONT_TRIM): rect(box,(255,220,0,255),3)
    rect(BARCODE,(255,0,255,255),4)
    proof.save(QA/"safe-area-proof.png", dpi=(150,150))

    template_render = render_pdf(TEMPLATE, TMP/"template150", 150)
    template = Image.open(template_render).convert("RGBA")
    overlay = Image.blend(im.convert("RGBA"), template, 0.34).convert("RGB")
    overlay_png = TMP/"overlay.png"; overlay.save(overlay_png, dpi=(150,150))
    c = Canvas(str(QA/"ingram-template-overlay.pdf"), pagesize=(PAGE_W,PAGE_H), pageCompression=1)
    c.drawImage(str(overlay_png),0,0,width=PAGE_W,height=PAGE_H); c.showPage(); c.save()


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
    if sha256(TEMPLATE) != "952d742a86b13ccfbc58dd9eeb3c9d70d9792dfa96faf2809a8c0a208af50f37":
        raise SystemExit("Official template checksum mismatch; refusing to build.")
    if not BARCODE_SOURCE.exists():
        # Exact official white barcode field measured on the immutable template,
        # extracted at 600 dpi without generating or altering barcode content.
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "600", "-f", "1", "-l", "1",
                        "-x", "5442", "-y", "5888", "-W", "754", "-H", "596",
                        str(TEMPLATE), str(BARCODE_SOURCE.with_suffix(""))], check=True)
    gray = Image.open(BARCODE_SOURCE).convert("L").point(lambda p: 0 if p < 180 else 255)
    barcode = Image.merge("CMYK", (Image.new("L", gray.size, 0), Image.new("L", gray.size, 0),
                                    Image.new("L", gray.size, 0), ImageOps.invert(gray)))
    barcode_path = TMP/"official-barcode-cmyk.tif"; barcode.save(barcode_path, dpi=(600,600), compression="tiff_lzw")
    artwork = TMP/"artwork.pdf"
    objects = draw_artwork(artwork, barcode_path)
    if any(not o["pass"] for o in objects):
        raise SystemExit("Critical object failed safe-area audit")
    merge_with_template(artwork, FINAL)
    render = render_pdf(FINAL, TMP/"final150", 150)
    make_qa(render)
    report = {"template": str(TEMPLATE.relative_to(ROOT)), "template_sha256": sha256(TEMPLATE),
              "output": str(FINAL.relative_to(ROOT)), "output_sha256": sha256(FINAL),
              "page_size_pt": [PAGE_W,PAGE_H], "page_size_in": [21.0,12.0],
              "spine_width_in": 1.239, "spine_width_mm": 31.47,
              "safe_rectangles_pt": {"back":BACK_SAFE,"spine":SPINE_SAFE,"front":FRONT_SAFE},
              "minimum_required_clearance_pt":7.2,"objects":objects,
              "result":"PASS" if all(o["pass"] for o in objects) else "FAIL"}
    (QA/"cover-object-bounds.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"result":report["result"],"objects":len(objects),"output":str(FINAL),"sha256":report["output_sha256"]},indent=2))


if __name__ == "__main__": main()
