# The Complete Software Engineering Lifecycle
## Complete Print Reconstruction Audit

### Branch

Main SHA: `2520b018c47fe36a3e2cd6a300f512187af7ff03`

Reconstruction branch: `se-print-reconstruction`

Reconstruction starting SHA: `2520b018c47fe36a3e2cd6a300f512187af7ff03`

### Canonical Source

Chapters: 15

Glossary: YES

HTML: PASS (complete digital profile rendered after reconstruction changes)

### Source Inventory

Headings: 881

Technical blocks: 696

Figures: 75

Tables: 44

Words: 79,936

Counts are parsed from the canonical QMD files; manifest SHA-256 values make the inventory reproducible. Raw heading-like lines embedded within source technical regions remain represented because the manifest is an editorial source-order inventory.

### Neutral PDF PDF

Path: `output/reconstruction/The-Complete-Software-Engineering-Lifecycle-NEUTRAL-REVIEW.pdf`

Pages: 715

Trim: 612 × 792 pt

Build: PASS

### Heading Completeness

Source: 881

PDF text matches: 881

Missing: 0

Duplicates: 0 detected

### Technical Blocks

Source: 696

Fully rendered by endpoint test: 694

Missing: 0

Truncated: 2

Mangling: 0 automatically established

Blocking detail: `block-0665` and `block-0672` in Chapter 13 are malformed/untyped source regions whose long lines exceed ordinary verbatim width. The start anchors render, but the end anchors do not survive PDF extraction. They require human/source-structure review; the neutral baseline deliberately does not conceal them with `SETerminal` or a semantic rewrite.

### Figures

Source: 75

Rendered captions: 75

PDF image objects: 203 (SVG/PDF internals can create multiple objects per figure)

Missing source files/captions: 0

### Tables

Source: 44

Rendered: 44 by source/PDF anchor parity

Missing: 0 detected

### Chapter Parity

Chapters 1–15: PASS for headings, figures, tables, and ≥99.7% normalized lexical coverage. Detailed counts are in `editorial/WEB-SOURCE-PRINT-PARITY.csv`.

### Chapter 8 Regression

JavaScript: PASS — standard syntax-highlighted code, no Terminal chrome

Python: PASS — standard syntax-highlighted code, no Terminal chrome

ASCII diagrams: PASS — plain monospaced blocks, no Terminal chrome

Testing pyramid: PASS — present on the chapter contact sheet

### Chapter 9/10 Regression

YAML: PASS

SQL: PASS

JavaScript: PASS

Shell: PASS as ordinary code

Configuration: PASS

ASCII diagrams: PASS as ordinary monospaced content

### Volume I Missing-Section Root Cause

The current Volume I PDF contains all tested raw H1–H4 source strings after accounting for the deliberate `volume-crossrefs.lua` rewrite of “Your Semester Project” to “Your Project” and PDF hyphenation of “Techniques.” The volume chapter selector and `volume-content.lua` do not delete sections. The apparent loss is caused by semantic/presentation collapse: the universal handler turns large technical and untyped regions into dark terminal panels, erasing heading hierarchy inside any region Pandoc parses as a CodeBlock. The Learning Objectives H2 is also deliberately removed and replaced by a custom environment. There is no evidence that the split deleted canonical chapter sections.

### Volume II Terminal Root Cause

`print-components.lua` unconditionally returns raw `SETerminal` LaTeX for every CodeBlock. `print/preamble.tex` defines the large dark window. This architecture predates the split and directly causes the Volume II symptom.

### Glyph QA

FAIL / MANUAL REVIEW REQUIRED

Replacement characters in extracted text: 1,233. Contact-sheet inspection shows readable code and diagrams, but Menlo's PDF text mapping does not round-trip numerous technical glyphs. Unicode semantics were not rewritten; this remains a blocking accessibility/extraction issue.

### Source Leakage

Unintended leakage: 0 detected. Twelve visible triple-backtick strings are intentional content inside examples; one `:::` occurrence is part of an AWS S3 ARN. No `RawBlock` marker was found.

### Low-Utilization Pages

2 automatically flagged: page 22 (front matter/TOC transition) and the terminal form-feed artifact after the last page. No pattern of dozens of custom-filter blank pages was found.

### Human Review Artifact

Neutral PDF: `output/reconstruction/The-Complete-Software-Engineering-Lifecycle-NEUTRAL-REVIEW.pdf`

Contact sheets: `editorial/qa/reconstruction/` (29 whole-book sheets and 15 chapter sheets)

### Final Status

READY FOR HUMAN CONTENT-COMPLETENESS REVIEW: NO — two Chapter 13 block truncation flags and the extraction glyph audit remain blocking.

READY FOR PRINT REDESIGN: NO

READY FOR VOLUME SPLIT: NO
