# Phase 6B — Print Rescue and Render-Completeness Audit

## Outcome

Phase 6B rescues the Phase 6 interior after manual review demonstrated that box geometry and text-bounding checks had missed obvious pagination failures. The final interior is a 608-page US Letter PDF. Acceptance is based on a 110 DPI raster of every page, 25 contact sheets, targeted 100% and grayscale inspection, and source-to-PDF completeness manifests.

## Root cause and architectural repair

The severe gaps were caused by placing long Pandoc verbatim payloads inside decorative `tcolorbox` environments. The outer box and inner verbatim material did not share a reliable page-breaking model. This produced empty panel shells, title/language-only pages, and code that appeared one or more pages later.

The repair separates technical-block behavior by length:

- blocks of 24 lines or fewer retain the compact Phase 5 semantic panel;
- long non-terminal blocks use Pandoc's native breakable listing flow with a compact semantic header;
- terminal blocks retain the dark terminal treatment and compact continuation labels;
- no long block reserves space for its entire payload;
- a conditional part break starts a divider on a clean page without generating a second blank page;
- the Preface objective panel is kept together;
- empty print-only `web-only` containers are removed rather than emitted as decorative shells.

This reduced the PDF from 653 pages to 608 while retaining canonical content, aside from the explicitly documented residue repairs.

## Full visual sweep

All 608 pages were rendered to PNG at 110 DPI. Contact sheets contain 25 pages each (the last contains eight) and are stored under `output/print/phase6b-contact-sheets/`.

The sweep explicitly covered every page, including all pages cited in the rejection prompt. It confirmed:

- no empty code/configuration-panel pages;
- no language-tag-only or continuation-title-only pages;
- no stranded subsection pages caused by technical-block reservation;
- long code begins naturally and continues without truncation;
- all 15 chapter openers use the same number/title architecture;
- the Preface Learning Objectives panel is complete on one page;
- all five part dividers are deliberate standalone pages;
- code remains light and terminal sessions remain dark;
- representative program and terminal pages remain readable in grayscale.

Grayscale samples are under `output/print/phase6b-grayscale-proof/`.

## Meaningful-utilization review

The new visual scanner excludes the running-head band, footer/folio band, trim margins, isolated rules, and empty panel borders. It uses visible body-ink row occupancy rather than PDF word counts.

Twelve pages fall below the review thresholds. Every one was manually reviewed and classified:

- title page: 1;
- half-title: 9;
- edition notice: 10;
- dedication: 12;
- How to Use/front-matter closing page: 16;
- part openers: 17, 144, 298, 354, 528;
- natural chapter-closing pages before forced chapter openings: 261 and 432.

Open or unexplained low-utilization pages: **0**. Unintentional empty content pages: **0**. Stranded content pages: **0**.

The row-level disposition is recorded in `editorial/phase6b-low-utilization-pages.csv`, with a contact-sheet reference for every flagged page.

## Render completeness

- Canonical technical blocks: **512/512 PASS**.
- Blocks with first and last source signatures found directly: **486**.
- Extraction-limited mathematical/diagram blocks confirmed through section anchors and the full raster sweep: **24**.
- Blocks manually confirmed by source line, PDF extraction, and page raster: **2**.
- Missing, partial, truncated, misplaced, or duplicated blocks: **0**.
- Phase 4 SVG figures: **14/14 PASS** with vector source, rendered figure, and visible caption.

See `editorial/PHASE-6B-TECHNICAL-BLOCK-MANIFEST.csv` and `editorial/PHASE-6B-FIGURE-MANIFEST.csv`.

## Mechanical QA

| Check | Result |
|---|---:|
| Physical pages | 608 |
| MediaBox | 612 × 792 pt |
| Physical overflow | 0 |
| Live-area text overflow | 0 |
| Wrong page boxes | 0 |
| Technical title-only pages | 0 |
| Code title-only pages | 0 |
| Duplicate numbered algorithms | 0 |
| Missing-glyph audit tokens | 0 |
| Fonts not embedded | 0 |
| Expected figures present | 14/14 |
| Expected technical blocks present | 512/512 |

The 25 Type 3 font records remain confined to established vector-figure conversions; body, heading, code, and navigation fonts are embedded CID/TrueType fonts. The earlier 72 ppi remote CC badge has been removed, leaving instructional graphics as vector artwork.

## Publication identity

- Edition: Second Edition
- Publication year: 2026
- Copyright: © 2026 Moody Amakobe
- Second Edition ISBN: TBD
- Historical First Edition ISBN: not reused as current metadata

## Acceptance artifact

The final stable proof is `output/print/Advanced-Computational-Algorithms-Print.pdf`. Phase 6 should not be treated as locked until this Phase 6B proof and its contact sheets receive manual acceptance.
