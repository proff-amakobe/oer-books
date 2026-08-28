# Software Engineering Two-Volume Architecture Report

## Series

Series: *The Complete Software Engineering Lifecycle*

## Volume I

- Title: *Software Engineering Foundations and Design*
- Series line: *The Complete Software Engineering Lifecycle — Volume I*
- Chapters: 1–8
- Final page count: **468**
- PDF: `output/volume1/Software-Engineering-Foundations-and-Design.pdf`
- EPUB: `output/volume1/Software-Engineering-Foundations-and-Design.epub`
- Print ISBN: TBD
- Ebook ISBN: TBD

## Volume II

- Title: *Software Delivery, Operations, and Evolution*
- Series line: *The Complete Software Engineering Lifecycle — Volume II*
- Chapters: 1–7
- Master source chapters: 9–15
- Final page count: **430**
- PDF: `output/volume2/Software-Delivery-Operations-and-Evolution.pdf`
- EPUB: `output/volume2/Software-Delivery-Operations-and-Evolution.epub`
- Print ISBN: TBD
- Ebook ISBN: TBD

## Architecture

The root `_quarto.yml` keeps `digital` as the default complete-OER profile and places `digital`, `print`, `volume1`, and `volume2` in one mutually exclusive profile group. `_quarto-volume1.yml` and `_quarto-volume2.yml` select canonical chapter paths directly; neither copies manuscript text. Isolated output directories prevent complete builds and volume builds from deleting one another.

Build commands:

```sh
quarto render
quarto render --profile print --to pdf
quarto render --profile volume1
quarto render --profile volume2
```

## Renumbering

Quarto assigns chapter numbers from the selected book order. Therefore Volume II maps master 9→1, 10→2, 11→3, 12→4, 13→5, 14→6, and 15→7 without changing source filenames or headings. Native section, TOC, bookmark, figure, and table counters inherit the local chapter counter. The print preamble supplies local chapter categories and opener descriptions.

## Cross-Reference Audit

- Hard-coded references found: 154 literal `Chapter N` occurrences (152 in the canonical glossary and two correct local references in Volume I prose).
- Converted dynamically: all glossary chapter annotations and index headings.
- Cross-volume references: 91 in Volume I and 53 in Volume II, explicitly labeled with the other volume.
- Remaining stale Volume II Chapter 9–15 references: **0**.

## Independence Audit

- Volume I: PASS.
- Volume II: PASS.

## Front Matter

- Volume I: half title, full title, copyright, About the Series, preface, usage guide, three-page TOC, three-page List of Figures.
- Volume II: half title, full title, copyright, About the Series, preface, usage guide, three-page TOC. It has no numbered figures or populated List of Tables, so empty lists are omitted.

## Glossary Strategy

The complete canonical glossary appears in both volumes. Render-time chapter mapping provides local and cross-volume labels from that one source.

## References Strategy

Both profiles retain canonical `references.bib` metadata. No sources were invented. Because the current manuscripts do not expose a reliable cited-source subset for each volume, no speculative bibliography filtering was added.

## Numbering QA

- Volume I TOC and openers: Chapters 1–8.
- Volume II TOC and openers: Chapters 1–7.
- Volume II first chapter: Continuous Integration and Continuous Deployment.
- Volume II last chapter: Final Project Integration and Course Synthesis.
- Volume I numbered figures run through Chapter 8; Volume II currently contains no captioned/numbered figures.
- Volume II currently contains no captioned/numbered tables; uncaptioned instructional tables remain intact.

## Print QA

- Page size: 612 × 792 points (US Letter) for both PDFs.
- Fonts: all embedded; no unembedded fonts reported by `pdffonts`.
- Encryption: none.
- Visual inspection: title, TOC, parts, representative chapter openers, and glossary pages PASS.
- Overflow: no visible page-edge overflow in representative renders. The repository PyMuPDF audit script was attempted, but its optional `pymupdf` dependency is not installed in this environment.
- Missing glyphs: none visible; build output reported none.

## Complete Edition Regression

- HTML: PASS; 15 chapters plus glossary present.
- Complete digital PDF: PASS; 909 pages during regression.
- EPUB: PASS; English metadata and complete title retained.
- Professional print PDF: PASS; 864 pages, Chapter 09 and Chapter 15 retained, page 864 blank.
- GitHub Pages URL: unchanged.
- Sitemap: unchanged structure and includes Chapters 9 and 15.
- `robots.txt`: present and unchanged.

## Publication Preparation

- Volume I page count: 468.
- Volume II page count: 430.
- ISBNs required: four if separate print and ebook ISBNs are assigned to both volumes.

## Next Steps

1. Author review of both PDFs.
2. Lock final page counts.
3. Obtain ISBNs.
4. Update metadata.
5. Generate Ingram templates.
6. Design coordinated Volume I and Volume II covers.
7. Determine pricing.
8. Perform final print-lock review.
