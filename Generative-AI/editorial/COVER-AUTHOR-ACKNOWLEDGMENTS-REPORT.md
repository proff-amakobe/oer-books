# Generative AI
## Publication Cover and Front Matter Pass

### Manuscript Integrity

Original source: **Generative-AI/original/**. All **50 files** checked before and after work against the existing lock. Original files modified: **0**. Hash changes: **0**. The original cover, chapter figures, captions, QMDs, PROJECT_ARC.md, and incidental metadata are unchanged. The old rewritten chapters remain excluded.

### Cover

Original source cover: `original/images/cover.png`. Original source modified: **NO**.

Derived publication cover: `assets/cover/generative-ai-cover.png`. Editable companion: `assets/cover/generative-ai-cover.svg`.

Dimensions: **2550 × 3300**. Aspect ratio: **0.772727 (8.5:11)**. Opaque PNG rendered natively from vector composition; no source-image recompression or raster upscaling. The design uses system-compatible typography without committing font files.

Title: **Generative AI**. Subtitle: **Foundations, Systems, Evaluation, and Responsible Deployment**. Author: **Moody Amakobe**. Publisher: **Global Data Science Institute**.

Thumbnail readability: **PASS**, inspected at full size, 50%, 300 px, and 150 px. Title, subtitle, author, and publisher have clear hierarchy and high contrast. Text remains inside generous safe margins. The 150 px title is immediately readable; subtitle is legible at 300 px. No clipping, transparency, distortion, compression artifacts, or enlarged raster artifacts were observed.

Web: **PASS**. EPUB: **PASS**. PDF: **PASS**. OpenGraph: **PASS**. Twitter and Book JSON-LD also use the publication cover. The EPUB manifest identifies the PNG as `cover-image`, with exact source PNG bytes and a cover XHTML/SVG wrapper. The PDF embeds exact cover pixels on its first page, without page number, caption, or figure number.

See `COVER-REVIEW.md` for source-cover findings and author-photo discovery, and `COVER-GENERATION-PROMPT.md` for the built-in image-generation concept prompt and final SVG reproduction method. The generated concept was below the requested master resolution; the final vector composition supplies crisp typography and geometric artwork at the required size.

### About the Author

Words: **230**, counting heading and word tokens. Author-confirmed source used: **advanced-algorithms-book/about-author.qmd**. Unsupported facts added: **0**. No photo added. Existing publication use of an author photo is documented in `COVER-REVIEW.md`.

### Acknowledgments

Words: **265**, counting heading and word tokens. Unverified named individuals added: **0**. The text acknowledges research, open-source resources, educators/students, and GDSI without inventing personal contributors.

### Builds

HTML: **PASS**. PDF: **PASS**. PDF pages: **215**. EPUB: **PASS**.

All PDF pages are **612 × 792 pt**. The opening cover and both expanded front-matter pages were visually inspected. `pdfinfo`, `pdffonts`, and `qpdf --check` pass. EPUB ZIP/XML/navigation checks pass; this is not formal EPUB certification. `verify_publication_cover.py` verifies cover packaging and every paragraph of the biography, acknowledgments, and unchanged Preface in HTML/PDF/EPUB.

### Regression

15 chapters: **PASS**. Original manuscript integrity: **PASS**. Homepage: **PASS**. Preface: **PASS**, unchanged. Downloads: **PASS**, locally.

Original fidelity: 470 headings, 948 paragraphs, 781 list items, 88 technical blocks, 15 tables, and 20 referenced chapter images; **0 missing**. Homepage tests pass at 1440/1024/768/390 px in light and dark modes, including cover bytes, navigation, download links, focus, contrast, and metadata. Homepage prose and Preface source are unchanged; homepage edits are limited to cover path and descriptive cover metadata. Browser verification used headless Chrome after the previously documented in-app browser connection failure.

### Deployment

Pending commit, push, successful exact-commit workflow, and live verification. The local report will record the confirmed revision and results after deployment.

### Final Status

PUBLICATION COVER: **PASS**. ABOUT AUTHOR: **PASS**. ACKNOWLEDGMENTS: **PASS**.

READY FOR CHAPTER VISUAL REVIEW: **YES**.

READY FOR FINAL PRINT DESIGN: **NO**. READY FOR ISBN: **NO**.

Next recommended task: **Chapter image and visual-placement review**. That task has not been started. No Ingram wraparound cover, spine, back cover, barcode, or ISBN assignment was created.
