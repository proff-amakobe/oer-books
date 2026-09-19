# Generative AI
## Original Manuscript Reset

### Canonical Manuscript

Path: **Generative-AI/original/**. These files are the sole manuscript authority; no prose from the previous automated rewrite is used.

### Original Files

QMD files: **15**. Image files: **32**. Other source files: **1**, `PROJECT_ARC.md`. Two incidental `.DS_Store` files are included in the local 50-file lock but excluded from Git. Exact filenames, order, titles, sizes, and hashes are in the original manifests.

### Integrity

Original hashes recorded: **PASS**. Original files modified: **0**. Original files with changed hashes: **0**.

Prose rewritten: **NO**. Headings rewritten: **NO**. Code rewritten: **NO**. Prompts rewritten: **NO**. Tables rewritten: **NO**. Citations rewritten: **NO**. Images replaced: **NO**.

Direct rendering uses the original QMD paths. No normalized or rewritten chapter copies are created. The resource aliases resolve image paths outside the immutable directory. The original metadata/body title differences and additional Frontier heading remain preserved.

### Book Architecture

Chapter files: **15**. External grouping Parts: **5**.

HTML: **PASS**. PDF: **PASS**. Local PDF pages: **211**. PDF trim: **612 × 792 pt on every page**. EPUB: **PASS**.

Review artifact: `output/reset/pdf/Generative-AI-ORIGINAL-MANUSCRIPT-REVIEW.pdf`. HTML and EPUB are under `output/reset/html/` and `output/reset/epub/`. Earlier review artifacts remain separate. This is not a final print edition; page count was not optimized.

### Images

Original images discovered: **32**. Images used: **20**. Supplied but unreferenced: **12**. Referenced but missing: **0**.

Missing-image report: `editorial/MISSING-ORIGINAL-IMAGES.csv` (header only; no missing entries). All referenced HTML images load. EPUB includes and references their exact source bytes, with packaging filenames checked by hash. PDF image inclusions use only the supplied assets. Captions and source image bytes remain unchanged. No replacement or newly generated artwork was created.

### Fidelity

| Payload | Checked | Missing |
|---|---:|---:|
| Original headings | 470 | 0 |
| Paragraphs | 948 | 0 |
| List items | 781 | 0 |
| Technical blocks | 88 | 0 |
| Tables | 15 | 0 |
| Referenced images | 20 | 0 |

`verify_original.py` compares each original source against HTML, PDF, and EPUB. HTML/EPUB technical checks retain spaces and indentation; PDF checks ignore only presentation wrapping and its continuation marker. Typed blocks retain their source language classes; untyped blocks are not classified as Terminal or another invented type.

PDF QA: `pdfinfo`, `pdffonts`, and `qpdf --check` pass. All 298 listed local font instances are embedded; no syntax/stream corruption or physical text overflow was detected. Representative authored image, wrapfigure, code/Unicode, and text pages were rendered and visually inspected. Browser checks cover all 15 chapters, 20 loaded images, 88 technical blocks, search, Read Online navigation, and mobile width. EPUB ZIP/XML, language, navigation, asset links, images, table cells, and technical content pass; no formal EPUB certification is claimed.

### Author Review Queue

Technical candidates: **6**. Freshness candidates: **32**. Citation candidates: **0** (no dedicated citation audit). Consistency candidates: **3**. Image-format candidates: **2**. Total: **43**.

Applied to manuscript: **0**. These are review suggestions, not automatic corrections or a new comprehensive scholarly audit.

### Previous Rewritten Manuscript

Status: **EXCLUDED / NOT ACTIVE MANUSCRIPT**. It remains in `chapters/` for history and is not part of the active Quarto configuration. Prior phase reports are marked **SUPERSEDED BY ORIGINAL MANUSCRIPT RESET**.

### Final Status

ORIGINAL MANUSCRIPT PRESERVED: **PASS**.

BOOK BUILDS FROM ORIGINAL: **PASS**.

READY FOR AUTHOR CONTENT REVIEW: **YES**.

READY FOR AUTOMATIC REWRITING: **NO**.

READY FOR FIGURE CREATION: **NO**.

READY FOR FINAL PRINT DESIGN: **NO**.

Commit/push delivery also requires successful GitHub Actions for the exact pushed revision and verification of the live original edition and its downloads. Remote pagination may vary with the CI toolchain; local counts above identify the artifact actually inspected here.
