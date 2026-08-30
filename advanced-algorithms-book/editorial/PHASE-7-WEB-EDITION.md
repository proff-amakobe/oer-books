# Advanced Computational Algorithms
## Phase 7 — Web Edition Synchronization and UX Modernization

### Baseline

Current public URL:
<https://proff-amakobe.github.io/oer-books/advanced-algorithms-book/>

Print edition baseline: 608 pages; Phase 6C corrected edition: 614 pages
Edition: Second Edition
Year: 2026

The detailed architecture baseline is recorded in `editorial/PHASE-7-WEB-AUDIT.md`.

### Publication Metadata

Title: **PASS**
Subtitle: **PASS**
Author: **PASS**
Publisher: **PASS**
Edition: **PASS**
Year: **PASS**
First Edition residue: **0**
2025 current-edition residue: **0**
Old ISBN incorrectly exposed: **NO**

The unassigned Second Edition ISBN is omitted from visible web metadata, JSON-LD, and the homepage citation. Publication date is deterministic (`2026-01-01`).

### Web Structure

Chapter URLs preserved: **15/15**
Chapter numbering: **PASS (1–15)**
Front matter: **PASS, unnumbered**
References: **PASS, unnumbered and navigable**

The sidebar now distinguishes Front Matter and Parts I–V, uses a non-color-only active marker, and retains natural title wrapping. Chapter openers use a separate decorative numeral without changing the semantic H1.

### Landing Page

Second Edition identity: **PASS**
Format actions: **PASS**
Book structure: **PASS**
OER identity: **PASS**

The homepage uses a lightweight graph/recurrence motif drawn in CSS. It does not use the existing robot cover as hero imagery. The existing cover remains the EPUB and social-card artwork because it carries no stale edition/year.

### Technical Blocks

Expected canonical blocks: **512**
HTML semantic wrappers: **513** (one format-level wrapper distinction)
Program code: **PASS**
Pseudocode: **PASS**
Terminal: **PASS**
Output: **PASS**
Configuration/data: **PASS**
Copy controls: **PASS**

Eligible code controls have accessible names and success feedback. Long code scrolls inside its own surface, while the page itself remains fixed to the viewport.

### Figures

Expected: **14**
Rendered: **14**
Responsive: **PASS**
Alt text: **PASS (14/14 meaningful descriptions)**
Overflow: **0**

All instructional figures remain SVG. Light figure cards preserve technical readability across appearance settings.

### Tables

Responsive: **PASS**
Overflow: **0 page-level; wide tables scroll locally**

### Accessibility

Heading hierarchy: **PASS**
Keyboard navigation: **PASS**
Focus: **PASS**
Contrast: **PASS**
Alt text: **PASS**
Landmarks: **PASS**

Quarto supplies the skip link and navigation/main/footer landmarks. Phase 7 adds visible focus outlines, accessible copy-button labels, a non-color-only active navigation marker, reduced-motion handling, and responsive technical content. No information is hover-only.

### Responsive QA

375px: **PASS**
430px: **PASS**
768px: **PASS**
1024px: **PASS**
1440px: **PASS**
Global horizontal overflow: **0 across 35 representative checks**

Eleven device-emulated review screenshots and `responsive-qa.json` are stored in `editorial/qa/phase7/`. Browser QA reported zero console exceptions, failed requests, missing images, or landmark failures.

### Search

**PASS.** Search index results were confirmed for Master Theorem, QuickSort, Huffman, NP-complete, dynamic programming, segment tree, and reproducibility.

### Downloads

PDF: **PASS** — `Advanced-Computational-Algorithms.pdf`
EPUB: **PASS** — `Advanced-Computational-Algorithms.epub`
GitHub: **PASS** — <https://github.com/proff-amakobe/oer-books/tree/main/advanced-algorithms-book>

The site links to Quarto's public digital PDF, not `output/print/Advanced-Computational-Algorithms-Print.pdf`.

### SEO

Canonical URLs: **PASS (24/24 HTML pages)**
OpenGraph/Twitter cards: **PASS**
JSON-LD Book: **PASS; no ISBN/DOI/rating/price fabrication**
Sitemap: **PASS**
Robots: **PASS**

### Link Audit

Internal links checked: **17,718**
Broken internal: **0**
External links checked: **51**
External direct success: **31**
External redirects: **11**
External confirmed broken: **0**
External unverifiable: **9**

The nine unverifiable links are DOI endpoints returning HTTP 403 to automated requests; they were retained rather than misclassified as academically broken. Details are in `editorial/phase7-external-links.json`.

### Format Parity

Print: **PASS**
HTML: **PASS**
EPUB: **PASS**

See `editorial/SECOND-EDITION-FORMAT-PARITY.md`.

## Mathematical Rendering Parity

PDF: **PASS**

HTML: **PASS**

EPUB: **PASS**

Whole-book equation audit: **PASS — 1,018/1,018 inventoried expressions**

Hidden equation defects: **0/10 remaining**

Missing math glyphs: **0/23 remaining**

HTML supplies 684 semantic MathJax carriers, EPUB supplies 684 MathML nodes, and neither format exposes raw LaTeX. Responsive browser QA passes 27/27 focused equation checks. See `editorial/PHASE-6C-WHOLE-BOOK-MATH-AUDIT.md` and `editorial/EQUATION-FORMAT-PARITY.csv`.

### Print Regression

Page count before Phase 6C: **608**
Page count after Phase 6C: **614 (+6, documented equation-semantic correction)**
Trim: **612 × 792 pt**
Technical blocks: **431/431 (81 net math blocks reclassified)**
Figures: **14/14**
Overflow: **0**
Fonts not embedded: **0**

PRINT REGRESSION AFTER AUTHORIZED MATH CORRECTION: **PASS**

### Deployment

Commit: **`38b827f` — `Fix whole-book mathematical rendering in Advanced Algorithms`**
Push: **PASS — `origin/main`**
GitHub Pages: **PASS — Actions run `33285203326`**
Live site: **PASS — cache-bypassed PDF, EPUB, and representative HTML artifacts verified after deployment**

The live PDF is 614 US Letter pages and has zero unembedded fonts. The complete live EPUB contains 684 MathML nodes, declares `en-US`, and exposes zero raw LaTeX. Representative live HTML checks likewise expose zero raw LaTeX.

### Final Status

WEB SECOND EDITION: **DEPLOYED AND VERIFIED**
PRINT EDITION: **614 PAGES / PHASE 6C REGRESSION PASS**
READY FOR PHASE 8: **YES**
