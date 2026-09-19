> **SUPERSEDED BY ORIGINAL MANUSCRIPT RESET.** Historical record only; the rewritten manuscript is not authoritative. See ORIGINAL-MANUSCRIPT-LOCK.md.

# Generative AI
## Phase 0 — Book Creation and Structural Baseline

### Source

Canonical directory: `Generative-AI`.

Only the fifteen supplied QMDs were used. The separate `intro-genai-llm-oer-book` was neither changed nor used as a manuscript, asset, configuration, or filter source. Original files were untracked; `source-baseline.json` preserves exact originals and hashes. `structural-changes.diff` makes every manuscript edit reviewable. `PROJECT_ARC.md` is classified as EDITORIAL_PLAN and retained under `editorial/`, outside the book body.

### Manuscript

| Measure | Result |
|---|---:|
| Chapters found | 15 |
| Chapters configured | 15/15 |
| Current chapter-source words | 53,720 |
| Original chapter-source words | 53,687 |
| Original headings, H1–H4 | 470 |
| Current semantic headings, including the missing-illustration note | 471 |
| Technical examples | 88 |
| Fenced technical blocks retained | 87 |
| Fenced calculation converted to native math | 1 |
| Chapter tables | 15 |
| Visual references retained | 30 |
| Actual image assets supplied | 0 |
| Display math expressions after conversion | 2 |

Words include source syntax and example text; see the audit's counting method. No prose-only or final typeset-word-count claim is made. Thirty visual references comprise twenty absent image files and ten caption-only proposals. These are disclosed in the review editions, not silently removed or replaced with old-book art.

### Part Structure

1. **Foundations of Generative AI:** Chapters 1–3.
2. **Building Generative AI Systems:** Chapters 4–7.
3. **Research, Modalities, and Evaluation:** Chapters 8–11.
4. **Responsible Generative AI:** Chapters 12–14, kept separate.
5. **Production and the Frontier:** Chapter 15.

Chapter titles follow the actual manuscript. Automatic chapter and section numbering replaces manual prefixes; stable cross-references follow the new numbering. The frontier remains within Chapter 15. Front matter and references are unnumbered.

### Publication Identity

| Field | Value |
|---|---|
| Title | Generative AI |
| Subtitle | TEMPORARY — pending author approval |
| Author | Moody Amakobe |
| Publisher | Global Data Science Institute |
| Edition | First Open Edition |
| Year | 2026 |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Language | en-US |
| Print ISBN / Ebook ISBN | TBD / TBD |

Three subtitle options are in `TITLE-SUBTITLE-OPTIONS.md`. No final subtitle, cover, credentials, or identifiers were invented. The author bio is a concise verbatim excerpt from the repository's existing deep-learning author page.

### Front Matter

| Item | Result |
|---|---|
| Index / HTML landing page | PASS |
| Preface | PASS |
| Copyright | PASS |
| Acknowledgments | PASS |
| About Author | PASS |
| References | PASS — two verified starting entries, existing chapter reading lists retained |

The landing page is excluded from both PDF and EPUB body content. No empty glossary: the fifteen populated chapter terminology tables remain available while consolidation is deferred.

### Build

| Edition | Result |
|---|---|
| HTML | PASS |
| PDF | PASS |
| PDF pages | 185 |
| PDF trim, every page | 612 × 792 pt (8.5 × 11 in) |
| EPUB | PASS |

Artifacts: `output/html/index.html`, `output/pdf/Generative-AI-REVIEW.pdf`, and `output/epub/Generative-AI.epub`. Output and caches are ignored by Git. The web bundle contains both downloads. Shared metadata and the three format profiles are independent of the old book's infrastructure.

The installed Quarto version is 1.7.31. The local TeX Live 2025 mirror had package checksum failures; `fvextra` and `newunicodechar` were installed successfully from the matching frozen 2025 repository. No fonts were committed. Native TeX fonts are referenced by filename. Builds do not execute manuscript examples or call model APIs.

### Completeness

| Check | Missing after conversion |
|---|---:|
| Substantive source headings | 0 |
| Technical examples | 0 |
| Tables / table cells | 0 |
| Existing visual descriptions | 0 |
| Substantive paragraphs checked | 0 |

**Pre-existing artwork gap: 20 image files.** A successful baseline build does not mean these illustrations have been recovered. All original URIs and captions remain in the source snapshot and figure inventory. Supplying or commissioning the missing art is next-phase work.

`verify_book.py` checks chapter numbering, source-heading interpretation, exact source technical payload preservation (with the documented math conversion), rendered technical payloads, table cells, paragraphs, native math, bibliography entries, local links/fragments, EPUB archive navigation, language, and PDF page bounds. It fails the build on detected loss. A negative-control test deliberately removed one rendered prompt payload and confirmed the verifier failed; the artifact was then restored. `qa-results.json` records the latest run. Semantic labels distinguish prompts, responses, code, pseudocode, data, structured output, and verbatim/program-output examples.

### Editorial Audit

| Item | Count / disposition |
|---|---|
| Administrative course residue | 0 detected after normalization |
| Video placeholders | 0 |
| Citation-needed candidates | 510 — broad review queue, not 510 adjudicated unsupported claims |
| Freshness items | 447 — pending primary-source review |
| Figure candidates | 41 |
| Detected credential-shaped secrets | 0 |

The manuscript audit records detailed Chapter 8 methodology gaps, Chapter 10 evaluation gaps, the distinct purposes of Chapters 12–14, and Chapter 15's operational/frontier coverage. Existing factual and code issues are flagged without aggressive technical rewriting. The original course-language inventory retains benign uses such as discussion questions and real operational timing.

### QA

| Check | Result |
|---|---|
| Unintended raw source leakage | 0 detected |
| Broken local links | 0 |
| Missing referenced rendered assets | 0; original missing artwork is explicitly disclosed |
| HTML / EPUB / PDF chapter completeness | 15/15 each |
| HTML search | PASS — tested query returned 39 results |
| Desktop and mobile page overflow | None detected at 1440 px and 390 px |
| HTML technical-block overflow | None detected |
| Navigation, copy controls, dark mode | PASS |
| PDF content bounds | PASS |
| PDF visual review | Contact-sheet review plus full-size technical, table, title, reference, and milestone pages |
| EPUB XML / ZIP / internal navigation | PASS |
| Formal EPUBCheck certification | Not performed |

The in-app browser failed during initialization; local headless Chrome provided the browser verification. Intentional Markdown inside prompt examples is excluded from source-leakage scans. Print symbol equivalents for emoji and diagram characters are documented; original source and HTML/EPUB payloads remain intact. Review layout prioritizes readability and breakable blocks, not final production design.

### GitHub

Deployment configuration: **PASS** — additive integration into the established monorepo `publish.yml`; existing book build/copy steps remain intact. The new project renders all three editions, runs content QA, and copies its web/download bundle into `public/Generative-AI/`.

The user checkout was on `se-print-reconstruction` with unrelated changes. An isolated checkout based on `origin/main` is used for the authorized commit; unrelated files and commits are excluded. GitHub Actions execution and live Pages deployment must be distinguished from local build verification. Commit/push results are reported separately after execution.

### Final Status

| Gate | Status |
|---|---|
| BOOK STRUCTURE | PASS |
| READY FOR AUTHOR REVIEW | YES |
| READY FOR TECHNICAL REVIEW | YES — review work remains, as audited |
| READY FOR FINAL PRINT DESIGN | NO |
| READY FOR ISBN | NO |

Next phase: author subtitle selection, primary-source/freshness verification, technical corrections, and recovery or creation of missing illustrations.
