# Software Engineering Two-Volume Print-Lock Report

## Volume I

| Check | Result |
|---|---|
| Title | Software Engineering Foundations and Design |
| Page count before | 468 |
| Page count after | 450 |
| Chapters | 8 |
| Closing Note | REMOVED |
| Glossary | REMOVED |
| TOC and bookmarks | PASS |
| Figures | PASS - populated List of Figures; zero geometry violations |
| Tables | PASS - registered tables remain in content; empty List of Tables omitted |
| PDF | `output/volume1/Software-Engineering-Foundations-and-Design.pdf` |

## Volume II

| Check | Result |
|---|---|
| Title | Software Delivery, Operations, and Evolution |
| Page count before | 429 |
| Page count after | 409 |
| Chapters | 7 |
| Glossary | REMOVED |
| TOC and bookmarks | PASS |
| Figures | PASS - no registered figure list is fabricated; zero geometry violations |
| Tables | PASS - empty List of Tables omitted |
| PDF | `output/volume2/Software-Delivery-Operations-and-Evolution.pdf` |

## Volume II Ingram Bleed Correction

Original Ingram response: **BOOKBLOCK: INSUFFICIENT BLEED**

Root cause: the title-page field and seven chapter-opener vector backgrounds terminated at the 8.5 x 11 inch MediaBox/trim edge. The source PDF had no physical bleed canvas, TrimBox, or BleedBox.

Correction: the locked 409 pages are imposed without scaling onto mirrored 621 x 810 pt production pages. Chapter-opener vector colors continue through the 0.125 inch top and outside bleed; ordinary text, figures, code, headers, footers, and page numbers retain their original trim-relative positions.

| Check | Result |
|---|---|
| Source page count | 409 |
| Ingram production/template count | 410 |
| Final bleed geometry | 621 x 810 pt |
| Odd/even mirroring | PASS |
| Full-bleed backgrounds | PASS |
| Fonts | PASS - all embedded |
| Cover/spine | UNCHANGED - official 410-page template and 0.839 in spine retained |

## Complete Edition

| Check | Result |
|---|---|
| 15 chapters | PASS |
| Canonical glossary retained | PASS |
| HTML | PASS |
| PDF | PASS |
| EPUB | PASS - verified with a clean EPUB-only render |
| Web assets, sitemap, robots, and SEO configuration | PASS - unchanged |

## QA

| Check | Result |
|---|---|
| Chapter numbering | PASS - Volume I 1-8; Volume II 1-7 |
| Cross references | PASS - no Volume II references to local Chapters 9-15 |
| Figures | PASS - 0 physical and 0 text-area overflows in both PDFs |
| Terminal pagination | PASS - no terminal-only final pages, empty continuation boxes, or margin collisions found |
| Low-utilization pages | PASS - intentional front-matter versos/openers excluded; Volume II trailing reference orphan removed |
| Missing glyphs | PASS visually - box drawing transliterated to ASCII in print-only terminal blocks; canonical digital source unchanged. `pdftotext` still emits replacement codes for some embedded SVG/font extraction maps, but rendered-page inspection found no visible replacement glyphs or black boxes. |
| Fonts | PASS - fonts embedded; no encryption; imported SVGs account for limited Type 3 subsets in Volume I |
| Trim | PASS - 612 × 792 points |
| Metadata | PASS - title, subject, and author verified |

## Build Commands

```bash
quarto render --profile volume1
quarto render --profile volume2
quarto render --profile digital
quarto render --profile digital --to epub
quarto render --profile print --to pdf
```

`pdfinfo` and PyMuPDF independently confirmed the final standalone page counts.

## Publication Metadata

- Volume I final page count: **450**
- Volume II final page count: **409**
- Volume I Print ISBN: **979-8-2408-9097-0**
- Volume II Print ISBN: **979-8-2408-9370-4**
- Volume I Ebook ISBN: **TBD**
- Volume II Ebook ISBN: **TBD**

## ISBN Lock

| Check | Result |
|---|---|
| Volume I Print ISBN | 979-8-2408-9097-0 |
| Volume II Print ISBN | 979-8-2408-9370-4 |
| Ebook ISBNs | TBD |
| ISBN cross-contamination | PASS |
| Volume I page count after ISBN update | 450 |
| Volume II page count after ISBN update | 409 |
| Interior pagination unchanged | PASS |

## Ingram Preparation

- Volume I template: **PENDING ISBN + FINAL PAGE COUNT**
- Volume II template: **PENDING ISBN + FINAL PAGE COUNT**

## Next Step

1. Obtain or assign standalone ISBNs.
2. Enter metadata in IngramSpark.
3. Generate official Ingram templates using the final page counts.
4. Design coordinated Volume I and Volume II wraparound covers.
5. Determine print cost and retail pricing.
6. Perform final cover/interior preflight.
7. Upload.
