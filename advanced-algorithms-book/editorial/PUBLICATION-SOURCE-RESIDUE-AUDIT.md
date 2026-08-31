# Publication Source-Residue Audit

## Scope

This audit covers the complete canonical Quarto manuscript and every rendered page of the 614-page *Advanced Computational Algorithms* PDF. It checks for visible YAML/front matter, code-fence markers, fenced-Div markers, Pandoc attributes, raw HTML, Quarto cell directives, format-specific metadata, and web-only interaction language.

## Confirmed Defects and Repairs

| Location | Confirmed visible residue | Repair | Rendered verification |
|---|---|---|---|
| PDF page 7 / `index.qmd` | The landing page's HTML-specific YAML front matter was serialized as a literal technical block by Quarto's self-contained PDF book merge. | Added a LaTeX-only guard for the unique `aca-home` landing-page metadata signature. Canonical HTML metadata remains intact. | Page 7 begins directly with the Preface and Welcome content. |
| PDF page 117 / Chapter 2 | Raw HTML disclosure wording printed as “Solutions (click to reveal).” | Changed the portable disclosure summary to “Solutions.” | PDF shows a normal “Solutions” label; HTML retains a functioning disclosure element. |
| PDF page 130 / Chapter 2 | An accidental nested `````python`` marker appeared inside a Python program. | Removed the stray marker from the canonical code block. | The program renders continuously with no fence marker. |

## Whole-Book Source Audit

- Canonical QMD files inspected: **24/24**.
- Project/profile YAML inspected: **PASS**.
- Literal YAML/front matter visible in the final PDF: **0**.
- Pandoc/Quarto fenced-Div markers visible in the final PDF: **0**.
- Pandoc attributes visible in the final PDF: **0**.
- Raw HTML tags visible in the final PDF: **0**.
- Quarto execution directives visible in the final PDF: **0**.
- Web-only interaction language visible in the final PDF: **0**.
- Quarto render structure warnings: **0**.

The landing page's raw HTML is intentionally contained by `.web-only` and is absent from PDF/EPUB. The Chapter 2 `<details>` element is intentionally portable: HTML uses it as a disclosure control, while Pandoc renders its neutral “Solutions” summary in PDF/EPUB.

The final PDF still contains literal Markdown and BibTeX fence examples on pages 582–587 and the YAML line `format: json` on page 575. These are intentional, labeled teaching examples inside the documentation/configuration section—not publication residue—and were visually reviewed in context.

## Rendered PDF Verification

- Final page count: **614**.
- Page size: **US Letter (612 × 792 pt)**.
- Pages rasterized: **614/614**.
- All-page contact sheets reviewed: **25/25**.
- Corrected pages reviewed at full resolution: **pages 7, 117, 118, and 130**.
- Physical overflow: **0**.
- Text-area overflow: **0**.
- Blank or nearly blank pages: **0**.
- Unembedded fonts: **0**.
- Missing glyph tokens: **0**.
- Technical title-only pages: **0**.

The page-density audit flagged pages 8 and 612 for manual review. Page 8 is the intentional short Preface continuation; page 612 is the intentional closing perspective. Both pass visual review.

Evidence is stored under `editorial/qa/source-residue/`, including 25 all-page contact sheets and full-resolution images of the corrected pages.

## Cross-Format and Regression Results

- PDF/HTML/EPUB build: **PASS**.
- Mathematical inventory/parity: **1,018/1,018 PASS**.
- HTML math carriers: **684**.
- EPUB MathML nodes: **684**.
- Raw LaTeX in HTML/EPUB: **0**.
- Technical blocks: **431/431**.
- Terminal blocks: **6 intentional dark treatments preserved**.
- Instructional figures: **14/14**.
- Behavioral verification: **12/12**.
- Chapter numbering: **1–15 in HTML and EPUB**.
- Internal links: **17,718 checked / 0 broken**.

## Final Status

PUBLICATION SOURCE RESIDUE: **PASS — 0 confirmed visible defects remaining**

PHASE 6C / PHASE 7 ARTIFACT: **PRESERVED AT 614 PAGES**
