# Generative AI

**Moody Amakobe · Global Data Science Institute · First Open Edition · 2026**

Approved subtitle: **Foundations, Systems, Evaluation, and Responsible Deployment**. This is a Phase 1 technical and scholarly review edition, not final print production. Print and ebook ISBNs are TBD.

Fifteen chapters in five Parts cover foundations and language models; prompt engineering; APIs, agents, RAG, and fine-tuning; research methods, multimodal AI, evaluation, and optimization; safety, ethics, and law; deployment and the frontier. A sustained research-assistant design portfolio connects the chapters. The canonical manuscript is exclusively `chapters/` in this project; the separate old book is not a source dependency.

## Requirements

Quarto 1.7.31 or later, Python 3.10+, and TinyTeX/TeX Live with XeLaTeX for PDF. Install the TeX packages `tex-gyre`, `fvextra` through `tlmgr install tex-gyre fvextra newunicodechar needspace` if needed. Fonts come from TeX, are referenced by filename for portability, and are not committed. Examples are not executed, so no API account or chapter-specific Python packages are required.

```sh
# From Generative-AI/
python3 scripts/build.py
# Or build an individual edition:
python3 scripts/build.py --format html
python3 scripts/build.py --format pdf
python3 scripts/build.py --format epub
```

Equivalent direct render commands are `quarto render --profile html`, `quarto render --profile print`, and `quarto render --profile epub`. Run `python3 scripts/assemble_web.py` afterward to assemble downloads and canonical URLs. A standalone HTML build has working download links only after PDF and EPUB have been built and assembled. Use `scripts/build.py` for a complete publication bundle. Render profiles sequentially.

Outputs:

- `output/html/index.html`: responsive web edition, search, navigation, and downloads.
- `output/pdf/Generative-AI-PHASE-1-REVIEW.pdf`: US Letter review PDF, 612 × 792 pt.
- `output/epub/Generative-AI.epub`: portable ebook with semantic prompts and code.

## Verification

```sh
python3 -m venv .venv
.venv/bin/pip install -r scripts/qa/requirements.txt
.venv/bin/python scripts/qa/verify_book.py
.venv/bin/python scripts/qa/verify_phase1.py
```

QA uses the Pandoc executable bundled with Quarto (or `PANDOC`/`pandoc` on PATH), and Poppler's `pdftotext`/`pdfinfo` for independent PDF inspection when available. Current results are documented in `editorial/PHASE-1-TECHNICAL-AND-SCHOLARLY-REVIEW.md`. Original queue lines point to the immutable Phase 0 source snapshot; added columns identify current locations. Do not regenerate the adjudicated queues. `technical-example-review.csv` distinguishes offline execution from static review. The audit distinguishes existing missing artwork from content lost in conversion. Twenty image files were absent from the supplied manuscript; review editions explicitly identify their absence and retain all visual descriptions. No replacement artwork has been imported from the old book.

## Repository layout

- `_quarto.yml`: shared metadata and canonical order; `_quarto-{html,print,epub}.yml`: edition profiles.
- `chapters/`: fifteen canonical chapters; root QMDs: front matter and references.
- `styles/`: lightweight HTML, print, and EPUB styles.
- `scripts/`: build assembly, semantic-block filter, and reproducible QA.
- `editorial/`: audits, inventories, subtitle options, preserved source snapshot, and the non-reader-facing project plan.
- `output/`: ignored generated editions. No empty asset or glossary directories.

The monorepo's established `.github/workflows/publish.yml` builds the new project and copies its HTML/download bundle to `public/Generative-AI/` alongside existing books. The intended URL is https://proff-amakobe.github.io/oer-books/Generative-AI/; a successful local build does not by itself establish a live deployment.

License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). © 2026 Moody Amakobe. See `copyright.qmd`.
