# Advanced Computational Algorithms
## Second Edition Audit

### Executive Summary

Overall condition: a substantial, energetic 15-chapter manuscript with broad practical coverage and successful three-format generation, but not yet production-safe. The most serious defects are structural numbering, an ineffective trim setting, untrusted citations, unstable PDF convergence, inconsistent code semantics, and fast-aging claims.

Strengths: clear enthusiasm, extensive examples, 279 explicitly tagged Python blocks, projects and applications, broad algorithm coverage, searchable HTML, and embedded PDF fonts.

Critical defects: HTML numbers intended Chapters 1–15 as 9–23; manual headings duplicate Quarto numbering; PDF is A4 despite the 7 × 10 intent; bibliography has one suspicious uncited record; `date: today` rendered as 2026-08-28; PDF reached the maximum rerun count; EPUB language is `C`; a `CONTRIBUTING.md` link is unresolved.

Recommended Second Edition direction: preserve URLs and accessible voice, repair semantics/metadata first, verify technical content and sources, then standardize pedagogy and build a sober “Geometry of Computation” visual system.

### Current Publication

- Title: Advanced Computational Algorithms
- Subtitle: Concepts, Complexity, and Applied Projects
- Edition: First Edition — 2025
- ISBN: 979-8-2754-2277-1 (existing value; do not change in this pass)
- Author: Moody Amakobe
- Publisher: inconsistent: Global Data Science Institute / Global Data Science Institute (GDSI Press)
- License: CC BY 4.0
- Chapters/parts: 15 chapters, 5 parts
- Print intent: 7 × 10 in, 11 pt, `scrbook`; actual PDF: A4
- PDF: 690 pages; 1,582,315 bytes; fonts embedded/subset/Unicode; not encrypted; not tagged
- HTML/PDF/EPUB baseline: PASS, with warnings and defects recorded below

### Baseline Repository Inventory

| Item | Classification | Evidence |
|---|---|---|
| `_quarto.yml`, eight front pages, 15 chapter QMD files | ACTIVE SOURCE | Render consumes 23 entries |
| `images/cover.png` | ACTIVE SOURCE | Referenced by config and metadata |
| `cover.png` | UNUSED / POSSIBLY STALE | Different near-square image; no active reference |
| `references.bib`, `references.qmd` | PLACEHOLDER / UNUSED | One suspicious record; no bibliography config/chapter entry |
| `summary.qmd` | PLACEHOLDER / UNUSED | Says the book has “no content whatsoever” |
| `_book/**` | ACTIVE GENERATED OUTPUT | HTML, PDF, EPUB are tracked in Git |
| `.quarto/**` | GENERATED CACHE | Ignored now but present locally; not tracked |
| `.DS_Store`, `_book/.DS_Store` | STALE GENERATED OS FILE | Both tracked |
| CSS/SCSS, standalone LaTeX, Lua filters, scripts, canonical `code/` folder | ABSENT | Source promises a `/code` folder that does not exist |
| monorepo `.github/workflows/publish.yml` | ACTIVE BUILD DEPENDENCY | Runs `quarto render`, copies `_book/*` to `public/advanced-algorithms-book/` |

The minimal `.gitignore` ignores only `/.quarto/`. Future policy should ignore `.DS_Store`, caches, temporary LaTeX, and local output; decide explicitly whether release artifacts are versioned. Do not mix deployment output with canonical source.

### Structural Audit

Chapter numbering: **confirmed defective**. Quarto treats all eight front files as chapters. Intended Chapter 1 renders as 9; Chapter 15 renders as 23. Some chapter files have an unrelated H1 and a manual `## Chapter N`, while others use manual `# Chapter N`, producing inconsistent document titles and duplicate prefixes.

Manual numbering: 72 explicit `Chapter N:`/`Section N.N:` headings; **374 numeric/manual headings** when headings such as `## 12.2` and `### 12.2.1` are included. Chapter 8 repeats `Chapter 8` for “Practical Implementation Guide.” Quarto-generated numbers combine with manual text, so HTML titles, sidebar, breadcrumbs, TOC, PDF TOC/bookmarks, and EPUB navigation are semantically wrong.

Recommended architecture: declare project metadata once; use `index.qmd` as the unnumbered landing/preface; put title/copyright/edition/dedication/how-to-use in format-appropriate front matter with `.unnumbered`/`number-sections: false`; treat author/institute pages as back matter or web “About” pages. Each actual chapter gets exactly one unprefixed H1 and ordinary unnumbered source section titles; Quarto generates 1–15 and hierarchical section numbers.

Front-matter roles:

- title: print/EPUB title leaf or generated metadata page, not an ordinary chapter.
- edition/copyright/dedication: true unnumbered print/EPUB front matter; optionally web pages.
- index/preface and how-to-use: unnumbered front matter, present across formats.
- about-author/about-gdsi: preferably unnumbered back matter; web About pages; institutional claims require confirmation.
- metadata (title, subtitle, author, publisher/imprint, edition, ISBN, fixed date, license, cover): one project/profile data source, never repeated independently in page YAML and body text.

### Chapter Architecture

The detailed matrix is in `chapter-structure-audit.csv`. All chapters include substantial explanation and code; coverage of explicit key terms, review questions, references, formal correctness, and consistent learning objectives is uneven. The current project-heavy structure is valuable but inconsistent.

Future chapter model: opener; learning objectives; conceptual foundation; worked example; numbered algorithm/pseudocode; correctness/proof; complexity; implementation; visual explanation; when to use; pitfalls; real-world application; summary; key terms; review questions; exercises; project/application; verified further reading.

Chapter 14 already contains profiling, integration testing, memory profiling, vectorization, caching, and parallelization. Retain and deepen those sections while evolving it to **Algorithm Engineering and Performance Evaluation**: experimental design, test generation, scaling, cache/memory behavior, fair comparisons, and reproducibility.

Chapter 15 contains reusable technical communication, documentation, peer review, repository, visualization, and portfolio material. Evolve it to **Research, Reproducibility, and Project Synthesis**; remove fixed 15-week/final-submission framing and add benchmark reporting, paper structure, demonstrations, reproducible artifacts, and ethics of performance claims.

### Code Audit

- Total fenced blocks: **537**.
- Explicit languages: Python 279, Java 1, JavaScript 1, Bash 5, YAML 1, Markdown 7, unlabeled 243.
- Potential runnable blocks: **289** by language tag/classification; none should be assumed independently runnable without extracting surrounding definitions, dependencies, and data.
- Pseudocode/terminal/unknown blocks are frequently unlabeled. Long-line and execution status are recorded row-by-row in `code-inventory.csv`.
- All observed blocks close their Markdown fences, but many code comments written outside fences become false H1 headings; terminal output and algorithms are mixed with unlabeled blocks.

Future system: PROGRAM CODE with a supported language and verified output; ALGORITHM boxes with number/Input/Output/Steps/Complexity; TERMINAL blocks only for commands/output/benchmarks. Isolated execution should classify each sample PASS, FAIL, INCOMPLETE, PSEUDOCODE, REQUIRES DEPENDENCY/DATASET, UNSAFE, or MANUAL REVIEW. No full execution claim is made in this audit.

Pedagogical mismatches to prioritize include demonstrations that discuss a named algorithm but delegate work to built-ins/libraries, benchmark wrappers that test library operations, and prose-level pseudocode labeled as Python. These require manual comparison against the row-level inventory; do not silently replace them.

### Figure Audit

- Diagram-like fenced blocks: **38**, including project directory trees and pseudo-visual traces.
- Existing instructional raster/vector figures in chapters: **0**; the only active source image is the cover/remote license badge.
- Highest-priority SVG candidates: recursion trees, heap/balanced trees, flow/residual/min-cut networks, complexity-class diagrams, suffix structures, segment trees, and FFT-style transformations.

Future visual system: one sans-serif label family; consistent node radius/stroke; directed/undirected edge grammar; arrowheads that survive reduction; semantic labels; neutral/default, teal/current, amber/selected, and restrained red/error states; numbered captions; WCAG contrast; non-color cues (stroke/dash/shape); grayscale proofing and textual descriptions. Support graphs, flows, DP grids, string matching, heaps/trees, suffix structures, FFT butterflies, and complexity relationships.

### Technical Correctness

The manuscript’s scale requires a dedicated subject-matter pass. Priority verification areas are Master Theorem boundary conditions, randomized guarantees, Cuckoo-hashing worst-case wording, Dijkstra preconditions, approximation ratios and metric assumptions, NP-completeness reductions, numerical stability, Strassen/FFT constraints, quantum complexity shorthand, fairness impossibility wording, and all stated data-structure bounds. Treat confident superlatives and deployment claims as **REQUIRES VERIFICATION**.

### Citations and Freshness

Bibliography quality is critical/untrusted. Minimum citation issues: **22**. Chapter 13 freshness candidates: **121**. See the dedicated audits; neither facts nor references were fabricated or replaced.

### Tone

The accessible voice is an asset. Tighten repeated “magic/wizard/boom/ready” language, remove pseudo-attributed quotations, and separate textbook exposition from course administration. Targeted tone hits: 29; broad course-specific findings: **102**.

### Content Promise and Gap Analysis

The preface promises parallel and distributed algorithms. Parallel material exists mainly as extensions/examples; distributed coverage appears in Ch. 13; neither is a systematic treatment. Streaming is meaningfully introduced in Ch. 6 and reinforced in Ch. 13, but lacks a unified chapter-level progression. The promised multi-language implementation is weak: nearly all tagged program code is Python.

| Topic | Recommendation | Rationale |
|---|---|---|
| Parallel, GPU/SIMD | INTEGRATE / ADD substantial module | Promised; current coverage is scattered |
| Distributed, MapReduce | INTEGRATE INTO EXISTING CHAPTERS | Existing Ch. 13 basis; add models/costs/failure semantics |
| Streaming, Bloom, Count-Min, reservoir sampling | INTEGRATE INTO Ch. 6/13 | Already present; unify guarantees and applications |
| External-memory, cache-aware/oblivious | INTEGRATE INTO Ch. 2/12/14 | Existing mentions support engineering theme |
| Modern approximation | INTEGRATE INTO Ch. 8 | Strengthen assumptions, LP/rounding, current examples |
| Learning-augmented algorithms | OPTIONAL | Valuable 2026 bridge, but source-intensive and fast-moving |

The five-part model remains sound. The proposed Part V “Algorithm Engineering and Modern Practice” better fits revised Chapters 14–15. Part IV can become “Advanced Algorithms and Structures”; rename only during implementation.

### Print, PDF, Web, and EPUB

Print recommendation: **8.5 × 11 in**, provisional pending page/cost proof. The current PDF is A4, not 7 × 10. It is untagged, and repeated LaTeX runs reached the cap. Automated source evidence predicts overflow risk from long code and wide pseudo-figures; full rendered-page visual QA remains required.

Web: strong basic Quarto navigation/search, weak landing hierarchy, missing working downloads and modern discovery metadata, numbering defects, and no custom responsive figure/code system. Preserve URLs.

EPUB builds and contains navigation/cover assets, but declares language `C`, inherits the 23-entry structure, uses a moving 2026 date, and needs EPUBCheck/ACE/device review for wrapping, tables, MathML, Unicode, cover semantics, and accessibility.

### Metadata, Identity, Bio, and Cover

Publisher presentation should be canonically **Global Data Science Institute (GDSI Press)** if the repository-supported imprint wording is author-approved; otherwise use Global Data Science Institute consistently. No new institutional claim is recommended. Future metadata should say Second Edition — 2026 only after confirmation and use a fixed publication date. Preserve the current ISBN until the publication-lock phase.

The author bio should be fact-checked by the author. Claims about founder status, university teaching, textbooks, supervision, sector deployments, blockchain architecture, and institutional leadership require confirmation. Prepare a short jacket/web bio and a longer back-matter bio after confirmation; do not invent credentials.

The active cover is `images/cover.png` (1410 × 2250); the distinct root `cover.png` is unused. Neither is an ideal 300 ppi print master for likely trims. See the cover brief.

### Accessibility Audit

Current risks: untagged PDF; EPUB language `C`; ASCII figures without alt descriptions; heading hierarchy polluted by code comments/manual numbering; few actual image descriptions; non-descriptive bare URLs; uncertain table headers/captions; blue-link/color reliance; and no documented landmark/contrast testing. Establish semantic headings/tables, meaningful link text, figure alt plus long descriptions, non-color states, keyboard/focus QA, tagged PDF strategy, correct language metadata, and EPUB accessibility validation.

### GitHub Pages and Repository Hygiene

The monorepo workflow renders the book from `advanced-algorithms-book/`, then copies `_book/*` to `public/advanced-algorithms-book/` and deploys the shared Pages artifact. Risks include output-dir changes, renamed chapter files, additional build dependencies, missing SVG conversion support, render warnings becoming failures, and profile changes not mirrored in CI. The workflow already installs `librsvg2-bin`.

Recommended source architecture (proposal only): root configs (`_quarto.yml`, `_quarto-print.yml`); `frontmatter/`; `chapters/`; `assets/{cover,figures,author,icons}`; `styles/{web.scss,print/}`; `filters/`; `editorial/`; `scripts/`; and ignored `output/`. Keep URL-preserving output mappings.

### Visual System Proposal

Use shared design tokens across web/print: navy/cobalt base, teal emphasis, amber selection, neutral grays, restrained red errors; serif reading text and compact sans labels; consistent chapter/part openers and tables. Define semantic boxes for THEOREM, PROOF, PROOF IDEA, ALGORITHM, COMPLEXITY, INTUITION, IMPLEMENTATION NOTE, COMMON PITFALL, REAL-WORLD CONNECTION, and EXERCISE. Code, pseudocode, and terminal treatments must remain visibly distinct in grayscale and EPUB.

### Priority Matrix and Implementation Roadmap

Critical: numbering, print geometry, technical correctness, citation integrity, metadata, build convergence. High: code/figure systems, chapter consistency, PDF/EPUB/accessibility, Chapters 14–15. Medium: tone, web redesign, modern topics. Low: decorative and minor stylistic refinement. Full phase details are in `SECOND-EDITION-ROADMAP.md`.

No canonical source, title, edition, ISBN, cover, trim, chapters, figures, code, or deployment configuration was changed during this audit.
