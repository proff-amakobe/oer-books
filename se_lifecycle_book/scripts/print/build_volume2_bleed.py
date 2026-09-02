#!/usr/bin/env python3
"""Create the 409-page Volume II Ingram bleed PDF without scaling content."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, FloatObject, NameObject, RectangleObject
from reportlab.pdfgen import canvas

MEDIA_W, MEDIA_H = 621, 810
TRIM_W, TRIM_H, BLEED = 612, 792, 9
OPENERS = {14, 87, 164, 228, 287, 340, 373}
NAVY = (7 / 255, 17 / 255, 71 / 255)
DEEP_BLUE = (18 / 255, 58 / 255, 157 / 255)
PERIWINKLE = (137 / 255, 152 / 255, 255 / 255)
ACADEMIC_BLUE = (49 / 255, 85 / 255, 212 / 255)


def overlay(page_no: int) -> PdfReader | None:
    if page_no != 3 and page_no not in OPENERS:
        return None
    data = BytesIO()
    c = canvas.Canvas(data, pagesize=(MEDIA_W, MEDIA_H), bottomup=1)
    if page_no == 3:
        c.setFillColorRGB(*NAVY)
        c.rect(0, 801, 621, 9, stroke=0, fill=1)
        c.rect(612, 801 - 4.20 * 72, 9, 4.20 * 72, stroke=0, fill=1)
        c.setFillColorRGB(*ACADEMIC_BLUE)
        c.rect(612, 801 - 4.28 * 72, 9, 0.08 * 72, stroke=0, fill=1)
    else:
        trim_left = 0 if page_no % 2 else 9
        split = trim_left + 0.53 * 612
        c.setFillColorRGB(*NAVY)
        c.rect(0, 801, split, 9, stroke=0, fill=1)
        c.setFillColorRGB(*DEEP_BLUE)
        c.rect(split, 801, 621 - split, 9, stroke=0, fill=1)
        outside_x = 612 if page_no % 2 else 0
        c.setFillColorRGB(*(DEEP_BLUE if page_no % 2 else NAVY))
        c.rect(outside_x, 801 - 5.62 * 72, 9, 5.62 * 72, stroke=0, fill=1)
        c.setFillColorRGB(*PERIWINKLE)
        c.rect(outside_x, 801 - 5.68 * 72, 9, 0.06 * 72, stroke=0, fill=1)
    c.save()
    data.seek(0)
    return PdfReader(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = PdfReader(args.source)
    if len(reader.pages) != 409:
        raise SystemExit(f"source has {len(reader.pages)} pages; expected 409")
    source_contents = [
        page.get_contents().get_data() if page.get_contents() else b""
        for page in reader.pages
    ]
    for page_no, page in enumerate(reader.pages, start=1):
        tx = 0 if page_no % 2 else BLEED
        ty = BLEED
        # Retain the source page's original clipping boundary. This prevents
        # latent off-page TikZ paths from appearing in the bind-side canvas.
        raw = source_contents[page_no - 1]
        clipped = DecodedStreamObject()
        clipped.set_data(
            f"q {tx} {ty} {TRIM_W} {TRIM_H} re W n\n"
            f"q 1 0 0 1 {tx} {ty} cm\n".encode()
            + raw
            + b"\nQ\nQ\n"
        )
        page[NameObject("/Contents")] = clipped

        annotations = page.get("/Annots")
        if annotations:
            for annotation_ref in annotations.get_object():
                annotation = annotation_ref.get_object()
                rect = annotation.get("/Rect")
                if rect:
                    annotation[NameObject("/Rect")] = ArrayObject(
                        [FloatObject(float(rect[0]) + tx), FloatObject(float(rect[1]) + ty),
                         FloatObject(float(rect[2]) + tx), FloatObject(float(rect[3]) + ty)]
                    )

        page.mediabox = RectangleObject((0, 0, MEDIA_W, MEDIA_H))
        page.cropbox = RectangleObject((0, 0, MEDIA_W, MEDIA_H))
        page.bleedbox = RectangleObject((0, 0, MEDIA_W, MEDIA_H))
        page.trimbox = RectangleObject((tx, ty, tx + TRIM_W, ty + TRIM_H))
        art = overlay(page_no)
        if art:
            page.merge_page(art.pages[0], over=True)
        if not art and len(page.get_contents().get_data()) < len(raw):
            raise RuntimeError(f"page {page_no}: source content was not preserved")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter(clone_from=reader)
    for page_no, page in enumerate(writer.pages, start=1):
        tx = 0 if page_no % 2 else BLEED
        page.mediabox = RectangleObject((0, 0, MEDIA_W, MEDIA_H))
        page.cropbox = RectangleObject((0, 0, MEDIA_W, MEDIA_H))
        page.bleedbox = RectangleObject((0, 0, MEDIA_W, MEDIA_H))
        page.trimbox = RectangleObject((tx, BLEED, tx + TRIM_W, BLEED + TRIM_H))
    if reader.metadata:
        writer.add_metadata(reader.metadata)
    with args.output.open("wb") as handle:
        writer.write(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
