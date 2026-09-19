# Generative AI
## Homepage and Preface Publication Pass

### Manuscript Integrity

Original files: **50** checked against the existing immutable lock before editing and after rendering. Modified: **0**. Hash changes: **0**. The 15 original QMDs, 32 images, PROJECT_ARC.md, and incidental metadata remain byte-identical. `chapters/` remains excluded.

### Homepage

| Check | Result |
|---|---|
| Hero | PASS |
| Author cover | PASS — exact source bytes via existing cover.jpg staging alias |
| Read Online | PASS |
| PDF | PASS — Download PDF (Review Edition) |
| EPUB | PASS |
| GitHub | PASS |
| Five-Part overview | PASS — 15 canonical chapter title links |
| Project section | PASS — original milestone evidence recorded separately |
| Audience | PASS |
| OER | PASS |
| Mobile | PASS — 1440, 1024, 768, 390 px, without horizontal overflow |
| Dark mode | PASS — all four widths |
| Accessibility | PASS — semantic headings, cover alt text, visible keyboard focus, wrapping links/buttons, link contrast at least 4.5:1 in both modes |
| SEO | PASS — canonical URL, OpenGraph, Twitter card, Book JSON-LD |

The cover remains the author's supplied image, including its embedded title. No artwork was modified or generated. Its JPEG bytes are served through the existing .jpg alias despite the original .png filename. Homepage styles are scoped to `.book-landing`; existing chapter typography is retained. The rich homepage is HTML-only. No cards, actions, or project promotional section appear in PDF/EPUB. HTML-only TOC settings do not suppress the book TOC.

Browser checks used headless Chrome after the in-app browser connection failed. Automated accessibility checks are targeted checks, not a full accessibility certification. GitHub's source URL returned HTTP 200. `scripts/qa/verify_homepage.py` supports local checks and the deployed `--base` URL. Screenshots and detailed results are under `output/reset/qa/w1/`.

### Preface

Purpose: **PASS**. Five-Part structure: **PASS**. Project philosophy: **PASS**. Audience: **PASS**. How-to-use: **PASS**. Fast-moving-field framing: **PASS**. OER framing: **PASS**.

Review/build-process language: **0**. Word count: **1,184**, counting headings and word tokens. The Preface occupies three pages in the local PDF (pages 10–12). All three pages were visually inspected. Copyright metadata remains consistent; acknowledgments and About the Author were not expanded.

### Builds

HTML: **PASS**. PDF: **PASS**. PDF pages: **214**. EPUB: **PASS**. PDF remains Letter, 612 × 792 pt. `pdfinfo`, `pdffonts`, and `qpdf --check` completed; fonts remain embedded and no PDF syntax/stream corruption was found. No print design changes were introduced.

### Fidelity

Headings missing: **0 / 470**. Paragraphs missing: **0 / 948**. List items missing: **0 / 781**. Technical blocks missing: **0 / 88**. Tables missing: **0 / 15**. Referenced chapter images missing: **0 / 20**. All 15 chapters pass `verify_original.py` in HTML, PDF, and EPUB.

### Deployment

Commit: the commit titled **Polish Generative AI homepage and preface** containing this report. Its exact hash and remote verification results will be recorded in the local report after deployment.

Push: **PENDING**. GitHub Actions: **PENDING**. Live homepage: **PENDING**. Local checks above are complete; remote success is not claimed before the workflow and live checks finish.

### Next Step

Recommend **About the Author / acknowledgments cleanup**, using author-provided biographical information, followed later by visual/figure review. Neither task has been started.
