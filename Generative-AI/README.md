# Generative AI

Moody Amakobe · Global Data Science Institute · First Open Edition · 2026 · CC BY 4.0

Subtitle: **Foundations, Systems, Evaluation, and Responsible Deployment**.

**The sole canonical manuscript is `original/`. Its files are immutable author-supplied content. Do not edit, normalize, correct, rename, or merge rewritten material into them.** Suggestions belong in `editorial/AUTHOR-REVIEW-QUEUE.csv` and require author approval before any future content change.

The supplied directory contains 15 chapter QMDs, 32 images, and `PROJECT_ARC.md`, plus two incidental `.DS_Store` files covered by the local integrity lock but excluded from Git. Exact order and titles are recorded in `editorial/ORIGINAL-CHAPTER-INVENTORY.csv`. The project arc is a production reference, not an additional chapter.

`chapters/` contains the previous automated rewrite and is **NOT ACTIVE MANUSCRIPT**. Prior Phase 0, 1, and 1B reports are superseded historical records. They do not authorize manuscript changes.

## Build

Use Quarto, Python 3.10+, and TinyTeX/TeX Live. Python QA dependencies are in `scripts/qa/requirements.txt`. PDF dependencies include XeLaTeX, `tex-gyre`, `fvextra`, `newunicodechar`, and `wrapfig`. On Linux install `librsvg2-bin`, `fonts-dejavu-core`, `poppler-utils`, and `qpdf`.

```sh
python3 scripts/lock_original.py
python3 scripts/build.py
python3 scripts/qa/verify_original.py
```

Use a Python environment containing the QA dependencies for the verification command. Build profiles sequentially; `scripts/build.py --format html`, `--format pdf`, or `--format epub` selects one edition. The script stages image resources, verifies source hashes, renders, assembles downloads, and verifies hashes again. Never execute the chapter examples as part of publication.

The original image references resolve through generated, byte-identical aliases under ignored `assets/images/`. Two supplied `.png` filenames contain JPEG data; additional `.jpg` aliases preserve those exact bytes. A narrowly scoped Lua bridge renders the authored raw-LaTeX Marcus image in HTML/EPUB and selects the JPEG alias in PDF. It has no code-block or heading transformation.

PDF fonts are generated dependencies outside the manuscript and are not committed. The build uses DejaVu Sans Mono from the standard Linux font path or the documented local Matplotlib installation, plus a SHA-256-checked Noto Sans Symbols 2 font downloaded to `output/reset/fonts/` for the original thumbs-up/down characters. Ordinary verbatim blocks retain their authored content, type, and normal font size.

Outputs:

- `output/reset/html/`: web edition, navigation, search, and downloads.
- `output/reset/pdf/Generative-AI-ORIGINAL-MANUSCRIPT-REVIEW.pdf`: neutral US Letter review PDF.
- `output/reset/epub/Generative-AI-ORIGINAL-MANUSCRIPT.epub`: EPUB from the same sources.

Earlier outputs remain outside `output/reset/`. No final print design, new illustration, or ISBN assignment is part of this reset.

## Integrity and fidelity

`editorial/ORIGINAL-MANUSCRIPT-LOCK.csv` locks raw bytes for all 50 supplied files. `.gitattributes` disables Git newline conversion for `original/**`. Local verification checks all 50 files; CI checks the 48 publishable files, excluding only the two uncommitted `.DS_Store` files. An unexpected change, addition, or loss fails verification.

`verify_original.py` checks original headings, paragraphs, list items, technical payloads and language classes, table cells, and referenced images against HTML/PDF/EPUB. EPUB image names may change, so embedded bytes are checked by hash. PDF comparisons ignore line wrapping and its generated continuation marker; HTML/EPUB code comparison preserves whitespace and indentation. See the integrity, infrastructure, and reset reports in `editorial/`.

Automatic section numbering is disabled. Authored headings, repeated material, manual figure captions, metadata titles, and Chapter 15’s additional top-level Frontier heading remain unchanged. Quarto may show a metadata title as well as a repeated body heading; this is retained for author review rather than removing an authored heading.

The GitHub workflow builds this edition and publishes `output/reset/html/` at https://proff-amakobe.github.io/oer-books/Generative-AI/. Publication requires a successful workflow and live-site verification.
