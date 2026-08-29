# Phase 6 — Print Typography and Page Architecture

## Outcome

Phase 6 establishes a print-only interior design system for *Advanced Computational Algorithms*. The canonical manuscript, chapter order, URL map, figure set, XeLaTeX engine, US Letter trim, First Edition metadata, publication year, and ISBN remain unchanged.

The production PDF is 653 pages at 612 × 792 points. The table of contents occupies six pages and resolves at section depth, keeping the navigation useful without allowing subsection detail to overwhelm it.

## Design rationale

- **Type system:** Latin Modern Roman is the primary reading face, with Latin Modern Sans for navigation and hierarchy and Latin Modern Mono for technical material. The body is 10.5 pt with 1.17 leading. Embedded subsets are used throughout.
- **Page architecture:** mirrored inner/outer margins support bound-book reading. Running heads identify the book on verso pages and the current chapter on recto pages; page numbers sit at the outer foot. Part dividers and chapter openers suppress running matter.
- **Hierarchy:** each part receives a numbered navy divider. Each chapter receives an explicit two-digit chapter label, quiet oversized numeral, strong title, and teal rule. Section headings use the same navy/ink system with protected following space.
- **Navigation:** the six-page TOC uses sans-serif part/chapter hierarchy and serif section entries. PDF TOC entries and the existing chapter sequence remain intact.
- **Semantic panels:** existing objective, theorem/proof, intuition, complexity, implementation-note, and warning language is recognized at render time and styled consistently. No manuscript wording is rewritten or reordered.
- **Technical continuity:** Phase 5 code, algorithm, output, configuration, and terminal semantics remain intact. Breakable technical panels reserve padding at page splits, preventing clipped final lines.
- **Tables and figures:** tables use compact book typography, improved row spacing, and consistent rules. Figures retain the Phase 4 vector assets and coordinated captions. The only raster object inside the PDF is Quarto's 88 × 31 px CC license badge; it is a small UI/legal mark, not instructional artwork. All instructional figures remain vector.

## Implemented files

- `print/preamble.tex` — print typography, page styles, opening architecture, TOC, captions, tables, and semantic panels.
- `filters/print-design.lua` — print-only part/chapter commands, semantic-panel recognition, and paragraph ink-state protection.
- `print/technical-blocks.tex` and `filters/technical-blocks.lua` — safe split padding and explicit post-terminal color reset.
- `_quarto.yml` and `_quarto-print.yml` — coordinated print geometry, leading, TOC depth, preamble, and filter wiring.
- `scripts/print/audit_phase6_pdf.py` — repeatable media-box, live-area, blank/low-use, font, panel, and stranded-heading audit.

## Quantitative QA

| Check | Result |
|---|---:|
| PDF pages | 653 |
| Trim | US Letter, 612 × 792 pt |
| TOC pages | 6 |
| Physical text overflow | 0 |
| Live-area text overflow | 0 |
| Blank/nearly blank pages | 0 |
| Low-utilization review pages | 7 |
| Unintentional low-utilization pages | 0 |
| Stranded numbered-heading candidates | 0 |
| Fonts not embedded | 0 |
| Type 3 font records | 25 |
| Numbered algorithms | 16 |
| Duplicate algorithm numbers | 0 |
| Technical title-only pages | 0 |
| Missing-glyph audit tokens | 0 |

The 25 Type 3 records originate in established SVG figure conversions; body, heading, navigation, and code typography are embedded CID/TrueType fonts. The seven low-use pages are five deliberate part dividers plus the dedication and How to Use transition pages. They are retained intentionally; there are no generated blank leaves.

Panel-label occurrence counts in the PDF are: 3 `LEARNING OBJECTIVES`, 2 `THEOREM`, 37 proof-label occurrences, 38 `INTUITION`, and 16 `COMPLEXITY`. No source block uses an explicit `COMMON PITFALL` lead label, so no content was invented to populate that style.

## Visual proof sample

Raster review covered the title page, first TOC page, all opening archetypes, representative body and technical pages, the Chapter 13 color-state regression page, and the final references page. The proof confirmed readable density, visible hierarchy, stable running matter, safe technical splits, and complete body ink after multipage terminal panels.

## Regression and build status

- Full `quarto render`: **PASS** (PDF, EPUB, HTML).
- Rendered HTML chapter sequence: **PASS**, Chapters 1–15.
- EPUB chapter sequence and `dc:language=en-US`: **PASS**.
- Phase 5 PDF technical regression: **PASS**.
- Phase 2 behavioral groups: **12/12 PASS**.
- The general example inventory still contains its documented locked-content dispositions (11 `FAIL`, 95 `MANUAL REVIEW`, 13 `PARTIAL / SNIPPET`, 2 `REQUIRES TOOLCHAIN`); Phase 6 did not change or reclassify them.

Generated QA artifacts:

- `editorial/phase6-print-qa.json`
- `editorial/phase6-pdf-technical-regression.json`
- `editorial/phase6-low-utilization-pages.csv`
- `editorial/phase6-blank-pages.csv`

## Production artifact

The stable print proof is `output/print/Advanced-Computational-Algorithms-Print.pdf`. This is an interior proof, not a printer-ready cover/interior distribution package. Final metadata, ISBN, cover, and distributor locks remain reserved for the publication-lock phase.
