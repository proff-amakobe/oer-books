# Advanced Computational Algorithms
## Phase 1 Structural Repair Report

### Baseline

- HTML: PASS
- PDF: PASS with maximum-rerun warning
- EPUB: PASS
- Baseline pages: 690
- Configured trim: 7 × 10 in
- Actual baseline trim: A4

### Chapter Numbering

- Before: Chapter 1 → 9; Chapter 15 → 23
- After: Chapter 1 → 1; Chapter 15 → 15
- Manual numeric headings before: 374
- Explicit Chapter/Section prefixes before: 72
- Remaining semantic heading-number defects: **0**

All 15 chapter source files retain their filenames. Manual `Chapter N`, `Section N.N`, and chapter-owned numeric heading prefixes were removed outside fenced content. Chapters 1, 2, and 5 had duplicate generic H1s removed; each chapter now has one semantic H1. Opener headings and their descendants are explicitly unnumbered so the first real section remains 1.1, 2.1, and so on.

### Front Matter

Before: eight pages were ordinary titled book chapters and consumed counters 1–8.

After: their explicit H1 headings are unnumbered; redundant page-level title YAML was removed. Preface, edition, copyright, dedication, author, institute, and usage pages remain accessible without consuming counters. The web-only manual publication page is unlisted in print; PDF/EPUB use the generated metadata title page.

Front matter consumes counters: **NO**. PDF front matter uses Roman numerals in the TOC pages; Arabic chapter pagination begins after the part opener. This is acceptable for the provisional proof.

### Print Geometry

- Old actual: A4 (595.28 × 841.89 pt)
- New actual: **612 × 792 pt**
- Expected: 8.5 × 11 in / 612 × 792 pt
- Page count: **716**
- Independent page count: 716 (`pypdf` temporary QA environment)
- Fonts: all reported fonts embedded and subset
- Encryption: none

The base PDF format and `_quarto-print.yml` both use explicit `paperwidth=8.5in` and `paperheight=11in`; HTML/EPUB receive no print geometry.

The automated MediaBox glyph scan found 24 extracted tokens extending 1.25–5.67 pt beyond the physical page boundary across 19 pages (61, 77, 78, 80–82, 92, 333, 347, 447, 454, 505, 543, 609, 611, 614, 619, 685, 707). These are code/data tokens, including long identifiers, format strings, and DNA strings. No instructional ASCII-diagram token was identified among the overflow set. The larger trim helps but does not eliminate the later code-formatting work; the 38 diagram candidates remain unchanged.

### HTML

- Numbering: PASS
- URLs preserved: PASS (15 original filenames)
- Download links: PASS (relative PDF and EPUB targets exist)
- Search: PASS (`search.json` and search UI present)
- Front matter: accessible and unnumbered
- Visual/static spot checks: Chapters 1, 4, 9, and 15 show matching page, breadcrumb, sidebar, and section numbers. The browser screenshot of Chapter 1 exposed an opener-descendant `1.0.1` issue; it was corrected and the final output contains no `1.0.1` residue.

### EPUB

- Numbering: PASS, chapter and section labels derived from semantic headings
- Language: **en-US**
- Navigation: PASS, Chapters 1–15 consecutive
- Print assumptions: none observed
- Stable date: 2025-01-01 (technical year anchor for the current First Edition transition)

Quarto 1.7 disables Pandoc's native numbering for book EPUB/PDF output. `filters/format-numbering.lua` supplies format labels from the same normalized heading tree while ignoring unnumbered front matter and part headings.

### Metadata

- `date: today`: REMOVED
- Stable date strategy: explicit `2025-01-01` year anchor, preventing rebuild drift; it is not a Second Edition publication claim
- Publisher centralized: PARTIAL/PASS architecture. Catalog metadata uses “Global Data Science Institute”; the already-used display imprint “Global Data Science Institute (GDSI Press)” is a single variable pending author catalog-identity confirmation.
- ISBN strategy: First Edition ISBN 979-8-2754-2277-1 preserved and explicitly labeled; future Second Edition ISBN remains TBD and was not fabricated.

ISBN occurrences are canonical config metadata and visible First Edition publication data in `title.qmd`; generated PDF/EPUB/HTML inherit current metadata. Audit reports retain historical references to the same ISBN.

### Links and Residue

- Broken `CONTRIBUTING.md`: REMOVED; replaced with accurate neutral wording because no contribution file exists.
- PDF download: PASS
- EPUB download: PASS
- `summary.qmd`: stale/unused and still excluded; placeholder text cannot enter the build.
- `.DS_Store`: ignored at root and recursively; existing tracked files were not deleted in this phase.
- Build outputs remain tracked for the least-disruptive deployment policy; broader cleanup is deferred.

### Build Warnings

Maximum-rerun warning: **RESOLVED**.

Root cause: the previous auxiliary/TOC/bookmark state was unstable around conflicting manual/generated numbering and multiple top-level headings. After normalization, every final PDF build converged in two XeLaTeX passes. No warning was suppressed.

The Chapter 14 Markdown examples contained nested triple fences that exposed code comments as document H1s. Four outer example fences were changed to four backticks, preserving their content while restoring the semantic tree.

### PDF and Visual QA

Rendered and inspected: metadata title page, TOC, Part I opener, Chapters 1, 4, 9, and 15 openers, and a representative monospaced/diagram page. Visible chapter labels, margins, headers/footers, and page numbering are intact. PDF bookmarks independently list front matter, Parts I–V, and Chapters 1–15 without the former 9–23 offset.

### QA Scripts

- Chapter/EPUB numbering: `scripts/audit_chapter_numbering.py`
- Heading normalization/idempotence: `scripts/normalize_heading_numbering.py`
- PDF geometry: `scripts/check_pdf_geometry.py`
- EPUB language: `scripts/check_epub_language.py`

### Deferred Work

- Figures: 38 candidates retained
- Code: 537 audited blocks retained; 24 overflow tokens documented for the code-system pass
- Citation issues: deferred
- Freshness: deferred
- Chapter 14 conceptual redesign: deferred
- Chapter 15 conceptual redesign: deferred
- Bibliography reconstruction: deferred

### Recommendation

The structural foundation is stable enough to proceed to **Phase 2 — technical correctness and citation preparation**. Publisher catalog wording should be confirmed before final publication metadata is locked.

