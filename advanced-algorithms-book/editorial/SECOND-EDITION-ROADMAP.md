# Advanced Computational Algorithms — Second Edition Roadmap

## Progress update — Phase 5 technical-block design

- Phase 1 structural foundation: **COMPLETE** (commit `15f62da`).
- Phase 2 technical correctness and citation preparation: **COMPLETE WITH DOCUMENTED MANUAL-REVIEW ITEMS**. All 537 fenced blocks were classified, safe definition-only Python blocks were executed, representative core algorithms received behavioral edge-case tests, local evidence-backed defects were repaired, and claim/citation manifests plus a verified staging bibliography were created. Untested snippets, dependency-bound examples, informal proofs, and the Chapter 13 freshness rewrite are explicitly not reported as passing.
- Phase 3 content modernization: **COMPLETE**. The 121-item freshness scope and 230 unresolved citation claims have claim-level dispositions; the verified bibliography is active; Chapter 13 is edition-durable; Chapters 14–15 have been refocused on algorithm engineering, empirical evaluation, reproducibility, and research synthesis; and HTML, PDF, and EPUB builds passed final QA.
- Phase 4 instructional figures and visual pedagogy: **COMPLETE**. Fourteen canonical SVG figures replace 15 high-value legacy candidates using a shared accessible visual system. All 38 audit candidates have final dispositions; cross-format builds, chapter-based numbering, geometry, accessibility metadata, SVG/EPUB assets, and Phase 2 behavioral regression checks pass.
- Phase 5 code, pseudocode, terminal, output, and configuration design: **COMPLETE**. All 514 rendered technical blocks have canonical semantic classes and coordinated HTML/PDF/EPUB treatments. Sixteen substantial algorithms use automatic chapter-based numbering; 17 long-line candidates were resolved; physical overflow, duplicate numbering, missing glyph, and title-only-page checks are zero; and all 12 Phase 2 behavioral groups pass.

The numbered roadmap below is retained as historical planning context; Phase 1 also absorbed its original metadata/frontmatter objective, and the current Phase 2 combined its technical and citation-preparation objectives without replacing the production bibliography.

| Phase | Objective and scope | Risk / dependencies | Expected files changed | QA requirements |
|---|---|---|---|---|
| 1. Structural and numbering repair | Separate unnumbered front matter; give every chapter one semantic H1; remove manual Chapter/Section prefixes while preserving URL filenames | Critical; depends on URL map and cross-format test fixture | `_quarto.yml`, front matter, 15 chapter headings | HTML/TOC/bookmarks/EPUB nav show Chapters 1–15; URL regression |
| 2. Metadata/frontmatter normalization | Establish one metadata source; freeze publication date; canonicalize publisher presentation | Critical; publisher wording and date need author confirmation | `_quarto.yml`, title/edition/copyright/index/about files | Metadata snapshot in HTML/PDF/EPUB; no duplicated visible title blocks |
| 3. Technical correctness and code verification | Verify algorithms, claims, complexities, and runnable examples in isolated harnesses | Critical; language/dependency ambiguity and 537 blocks | Chapters, `tests/examples/`, scripts | Unit/property tests, complexity review, failure log, Unicode/width lint |
| 4. Reference and citation reconstruction | Replace untrusted bibliography with verified records and claim-level citations | Critical; primary-source verification required | `references.bib`, `references.qmd`, chapters | Bib parse, DOI/title/author validation, cited/uncited report, quotation audit |
| 5. Content modernization | Date and source fast-moving Ch. 13 claims; decide modern algorithm gaps | High; follows technical/citation audit | Primarily Chs. 6, 12, 13–15 | 2026 fact check with dated primary sources; scope review |
| 6. Pedagogical standardization | Apply standard chapter model and reconcile promised features | High; avoid formulaic repetition | All chapters, how-to-use, preface | Chapter matrix complete; objectives/exercises aligned |
| 7. Instructional SVG figure system | Create design tokens and replace high-priority ASCII diagrams | High; depends on stable content and accessibility spec | `assets/figures/`, captions, chapter references | SVG validation, contrast/grayscale, alt text, print/mobile proof |
| 8. Code/pseudocode/terminal system | Distinguish executable code, numbered algorithms, output, and terminal sessions | High; depends on verified code inventory | Chapters, SCSS/TeX/Lua filters | Highlighting, line length, page breaks, copy behavior, EPUB wrap |
| 9. Print redesign | Implement verified trim, typography, running heads, parts, openers, tables | High; follows content stabilization | `_quarto-print.yml`, print styles/templates | PDF boxes, fonts, overflow scan, 300 ppi images, physical/Ingram proof |
| 10. Web redesign | Modernize hierarchy and metadata without URL changes | Medium; depends on brand tokens | `_quarto.yml`, `styles/web.scss`, includes | URL crawl, mobile/keyboard/WCAG, search, SEO/schema, downloads |
| 11. EPUB optimization | Repair navigation/language/metadata and optimize code/tables/equations/images | High; follows structure/style work | EPUB profile/styles/metadata | EPUBCheck, ACE/accessibility review, device tests, TOC 1–15 |
| 12. Cover redesign | Produce “Geometry of Computation” family across print/web/EPUB | Medium; needs trim/page count/edition lock | `assets/cover/`, print cover package | Printer template, bleed/safe area, grayscale/thumbnail, proof approval |
| 13. ISBN/Ingram/publication lock | Apply confirmed Second Edition identity and freeze release artifacts | Critical and irreversible; depends on all prior phases | Metadata, copyright, distribution package | ISBN/publisher/date sign-off, checksums, final preflight, tagged archive |

## Priority matrix

- **Critical:** numbering (HTML proves 9–23), technical correctness, citation integrity (one suspicious uncited record), metadata consistency, deterministic builds, effective print size.
- **High:** code semantics and testing, instructional figure replacement, PDF/EPUB layout and accessibility, consistent chapter structure, Chapters 14–15 refocus.
- **Medium:** tone calibration, web visual redesign/SEO, selected modern topics, author-bio variants.
- **Low:** minor copy refinements, decorative flourishes, low-value project directory diagrams.

No phase should change the public URL map without a redirect and regression plan. Publication metadata, ISBN, cover, and printer files remain locked until Phase 13.
