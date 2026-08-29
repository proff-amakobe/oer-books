# Advanced Computational Algorithms
## Phase 4 Instructional Figures and Visual Pedagogy

### Executive Summary

ASCII/text visual candidates before: **38**

Final instructional figures created: **14 SVGs**

Legacy candidate rows replaced: **15** (two pairs consolidate into one figure each)

Reclassified as tables: **3**

Reclassified as algorithms: **7**

Kept as text: **12**

Removed as redundant: **1**

Diagram-like unlabeled text blocks remaining: **22**. These are intentionally retained directory trees, procedural traces, formulas, compact comparisons, or deferred semantic conversions; they are not unresolved broken-image replacements.

### Builds

HTML: **PASS**

PDF: **PASS**

EPUB: **PASS**

### Pagination

Before: **720 pages**

After: **720 pages**

### Figure System

Palette: deep navy, cobalt blue, electric teal, restrained amber, white, light gray, and charcoal.

Typography: system Arial/Helvetica sans serif; monospace only for strings and coefficient vectors. No font files are embedded in the SVG sources.

Node conventions: pale/navy default nodes; solid navy source nodes; solid teal sink nodes; teal active states; amber selected states; heavier strokes reinforce highlighting.

Edge conventions: navy directed edges, teal heavy selected paths, dashed residual/backward edges, and inline capacity or flow/capacity labels.

Highlight conventions: color is reinforced by stroke weight, fill value, dashes, labels, and source/sink state. Representative grayscale proofs retained path, cut, query, and selected-node distinctions.

### Chapter Inventory

- Chapter 1: created 0
- Chapter 2: created 4
- Chapter 3: created 1
- Chapter 4: created 0
- Chapter 5: created 1
- Chapter 6: created 0
- Chapter 7: created 1
- Chapter 8: created 0
- Chapter 9: created 4
- Chapter 10: created 1
- Chapter 11: created 1
- Chapter 12: created 1
- Chapters 13–15: created 0

### Priority Figures

- Find-max, merge-sort decomposition, merge-sort work, and quicksort partition-balance trees
- Binary heap to array-index mapping
- Naive versus memoized Fibonacci dependencies
- Mathematically conservative P/NP/PSPACE/EXP containment
- Capacity network, residual edge, max-flow/min-cut, and bipartite-matching flow
- Suffix-array ordering for `banana$`
- FFT convolution pipeline
- Segment-tree range-query decomposition

### Network Flow

Status: **PASS**. Four coordinated SVGs use capacity-only and flow/capacity notation, distinct residual/backward edges, source/sink states, selected paths, and a labeled minimum cut. Values match the surrounding Chapter 9 examples.

### Data Structures

Status: **PASS**. Heap indices and values match the manuscript. The segment tree preserves array values, interval sums, query decomposition, and result 24.

### Dynamic Programming

Status: **PASS**. The Fibonacci comparison makes repeated subproblems and memoized unique states spatially explicit. Knapsack recurrences and backtracking remain procedural text pending a native-table treatment.

### String Algorithms

Status: **PASS**. The suffix figure preserves all suffixes and the verified suffix array `[6, 5, 3, 1, 0, 4, 2]`.

### Accessibility

Alt text: **PASS — 0 missing**

Color independence: **PASS**

Grayscale: **PASS**

SVG title/description elements: **PASS — 14 of 14**

### Print QA

Physical instructional-figure overflow: **0**

Instructional-figure text-area violations: **0**

Clipped labels: **0**

MediaBox violations: **0**

Representative odd and even pages were rasterized and inspected. The geometry audit also records 23 pre-existing long code-line candidates; these belong to the locked code system and are deferred to Phase 5.

### Web QA

Responsive: **PASS**

Horizontal figure overflow: **0**

All generated figures use Quarto's responsive `img-fluid` class, percentage widths no greater than 88%, and intrinsic SVG viewBoxes. A localhost crawl loaded every referenced SVG successfully. The in-app preview connection and macOS headless screenshot surface were unavailable/unreliable, so responsive verification used generated markup/CSS, local HTTP asset requests, and direct SVG raster proofs.

### EPUB QA

Missing assets: **0**

Clipped visuals: **0 observed**

The EPUB contains 14 SVG assets and 14 figure references with no missing `alt` attributes; language remains `en-US` and navigation remains intact.

### Figure Numbering

Duplicates: **0**

Missing captions: **0**

Generic captions: **0**

HTML and PDF use chapter-based automatic numbering (for example, Figures 2.1–2.4 and 9.1–9.4). No List of Figures was enabled because the current print system does not require one.

### Structural Regression

Chapter numbering: **PASS — 1 through 15**

Front matter and References unnumbered: **PASS**

Trim: **PASS — 612 × 792 pt**

EPUB language: **PASS — en-US**

URLs: **PASS — public chapter filenames unchanged**

Phase 2 behavioral verification: **PASS — 12 of 12 groups**

### Deferred Visual Work

Code blocks: **537 before / 520 after**. Seventeen fenced instructional ASCII/pipeline blocks were removed during semantic replacement; executable code behavior was not redesigned.

Code redesign: **PHASE 5**

Chapter opener redesign: **later phase**

Cover: **later phase**

Optional high-value opportunities are recorded in `optional-figure-opportunities.md`.

### Recommendation

The visual manuscript is ready for **Phase 5 — Code, Pseudocode, and Terminal Design**. Phase 5 should address the 23 locked long-code-line overflow candidates without changing the instructional SVG system.
