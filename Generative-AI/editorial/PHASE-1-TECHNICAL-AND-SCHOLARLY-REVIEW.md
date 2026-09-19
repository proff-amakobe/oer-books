# Generative AI
## Phase 1 — Technical Correctness, Freshness, and Scholarly Integrity

Verification date: **2026-09-18**. Canonical publication: Moody Amakobe; Global Data Science Institute; First Open Edition; 2026; CC BY 4.0; en-US. Print and ebook ISBNs remain TBD.

### Subtitle

**Foundations, Systems, Evaluation, and Responsible Deployment**

Applied to shared Quarto metadata, title/copyright pages, HTML, PDF, EPUB catalog metadata, citation metadata, Book JSON-LD, OpenGraph, README, and current editorial metadata. Historical Phase 0 snapshots and reports retain their original wording as provenance.

### Freshness

| Disposition | Count |
|---|---:|
| Candidates | 447 |
| Current/durable | 178 |
| Current/time-sensitive | 10 |
| Outdated, corrected | 17 |
| Removed/generalized | 109 |
| Illustrative only | 133 |
| Unverified | 0 |
| Author review required to resolve a factual claim | 0 |

These are candidate dispositions, not 447 confirmed errors. Original columns and ordering remain traceable to immutable backups. Current excerpts and correction fields distinguish original claims from revised text. The 109 generalized items include both aging specifics and unsupported universal claims; they are not all obsolete facts.

### Citations

| Disposition | Count |
|---|---:|
| Candidates | 510 |
| Required | 55 |
| Helpful | 55 |
| Common knowledge / ordinary author framing | 154 |
| Illustrative / teaching items | 182 |
| Removed/generalized | 64 |
| Unresolved | 0 |

**260 inline citation references added**, excluding the 50 Further Reading citations. Counts use source-key occurrences, so a two-source bracket counts as two references. Required claims are supported in the paragraph, glossary, or directly associated scoped technical discussion; summary restatements need not duplicate every citation.

### Bibliography

Before: **2**. After: **97**. Verified: **97**. Unverified: **0**.

Metadata and source identity were checked against original papers, provider documentation, firsthand company accounts, official legal instruments, and regulator/court materials. ArXiv entries identify preprint versions; no peer-review status is inferred from arXiv hosting. Living documents have access dates. Corporate author names are preserved as organizations. The source registry retains metadata rather than copied abstracts.

Further Reading: **50/50 reviewed** — 19 verified, 18 corrected, 13 replaced, 0 removed. Replacement is explicit where the earlier entry was vague or its identity could not be verified.

### Technical Corrections

Critical: **0**. High: **79**. Medium: **211**. Low: **74**.

Resolved correction records: **364**. Remaining recorded issues: **0**. Counts include citation/reading-list corrections as well as scientific, security, numerical, and legal changes.

Representative changes:

- Chapters 1–2: causal attention versus parallel training, non-guaranteed novelty, tokenization and sampling caveats, emergent-capability measurement, undisclosed proprietary sizes, historical provider examples, and corrected fictional cost arithmetic.
- Chapters 3–5: chain-of-thought faithfulness, heuristic injection defenses versus enforced permissions, stable A/B assignment, correct chi-square success/failure table, retry eligibility and budgets, cache authorization, agent scope, and bounded interpretation of company claims.
- Chapters 6–9: RAG limitations and ranking metrics, FAISS scope, chunking heuristics, PEFT and QLoRA hardware limits, dataset-size uncertainty, contamination and peer-review caveats, and multimodal reliability/fallback limits.
- Chapters 10–13: metric validity, judge biases and calibration, retrieval versus generation errors, token-level MoE, hardware-dependent compression, speculative decoding assumptions, TGI maintenance status, DPO, alignment limits, equalized odds, differential privacy, federated-learning limits, and non-causal feature attribution.
- Chapters 14–15: dated EU/US scope, AI Act application stages, GDPR automated decisions and transfers, case-specific copyright findings, liability versus operational ownership, optional microservices, stateful rollback constraints, model migration, reasoning-model scope, and conditional synthetic-data findings.

`technical-corrections.csv` contains exact original/replacement text and evidence. `phase-1-technical-changes.diff` exposes the complete chapter diff. `phase-1-content-changes.json` authorizes the eight changed technical payloads and two heading wording corrections; the verifier rejects other payload changes. The 15 chapters, five Parts, automatic numbering, front matter, 15 semantic tables, and semantic block identities remain intact.

### Case Studies

Reviewed: **23** (10 primary-source cases and 13 explicitly illustrative scenarios). Primary-source identity/claims verified: **10**. Real cases corrected or bounded: **10**. Removed cases: **0**. Verification and correction counts overlap: each retained real case was checked and revised.

Associated Press, Bloomberg, Notion, Klarna, Glean, Harvey, Be My Eyes, Anthropic (two distinct cases), and Microsoft are covered. Chapter 15's deprecation case is explicitly hypothetical. Company reporting verifies what the company reported; it is not independent validation of business benefits, internal architecture, safety, or current performance. Harvey's cited example concerns legal embeddings, not an inferred generative-model training pipeline.

### Technical Examples

| Classification | Count |
|---|---:|
| Total | 88 |
| Pseudocode | 19 |
| Static examples, including native math | 3 |
| Runnable offline | 1 |
| Runnable with dependencies | 0 |
| Requires API as a runnable example | 0 |
| Structured data | 9 |
| Prompts | 48 |
| Model responses | 7 |
| Program output | 1 |

Executed: **1**. Pass: **1**. Fail: **0**. The SHA-256 assignment function was checked for valid labels, both variants across 1,000 units, and reproducibility across two Python processes. Two JSON examples were parsed. Pseudocode was reviewed statically and was not reported as executed Python; some sketches assume model/network helpers that were never invoked. No external model API calls, credentials, or spending were used. Secret/destructive-command review: **0 real secrets detected**. Generated responses and dashboard values are illustrative, not experimental observations.

### Law / Policy

Jurisdiction-sensitive sourced statements reviewed: **23**. Primary-source supported: **23**. Unresolved within the stated educational scope: **0**. The count is the sourced paragraph/glossary-unit inventory in `law-policy-audit.csv`, not a count of all legal propositions or an exhaustive legal survey.

The chapter distinguishes EU and US rules, effective and future application dates, proposed rules, voluntary guidance, and a dated case-specific court holding. The review date is not a promise of continuing currency. This is educational treatment, not legal advice or certification of a particular deployment.

### Builds

HTML: **PASS**. PDF: **PASS**. EPUB: **PASS**.

PDF: **192 pages**, every page **612 × 792 pt** (8.5 × 11 in). Artifact: `output/pdf/Generative-AI-PHASE-1-REVIEW.pdf`.

HTML search, navigation, copy controls, and desktop/mobile overflow checked in local Chrome; 39 search results for “retrieval,” no overflowing technical blocks, and no mobile page overflow. Representative PDF title, contents, technical-example, law-table, and bibliography pages were rendered and visually reviewed. EPUB ZIP, XML, navigation, local links, chapter content, and catalog metadata pass; this is not a formal EPUBCheck certification.

### Completeness

Missing substantive headings: **0**. Missing technical examples: **0**. Missing tables: **0**. Missing substantive paragraphs: **0**.

The format verifier checks **471 headings**, **87 literal blocks plus the preserved native worked calculation**, **15 tables**, and **888 substantive paragraphs** against the revised source in all three formats. It also checks the Phase 0 baseline with explicitly documented heading/payload corrections, bibliography entries, chapter numbering, links, source leakage, and PDF page bounds. This is preservation of reviewed content, not a claim that intentionally corrected historical wording remains unchanged.

Twenty pre-existing missing image files and 30 visual proposals remain disclosed. No figures were generated. Missing supplied artwork is not counted as lost manuscript content.

### Reproduction and audit trail

Run `python3 scripts/build.py`, then `.venv/bin/python scripts/qa/verify_book.py` and `.venv/bin/python scripts/qa/verify_phase1.py` after installing `scripts/qa/requirements.txt`. The Phase 0 queue generator now refuses to overwrite adjudicated Phase 1 records. Current QA results are in `qa-results.json` and `phase-1-qa-results.json`.

Publication work is prepared in the isolated main checkout to avoid unrelated dirty files in the user's existing branch. Generated editions, virtual environments, and caches are not committed. Only `Generative-AI/` source, build support, and editorial records belong to this change.

### Final Status

TECHNICAL REVIEW: **PASS** within this documented editorial scope.

READY FOR PHASE 2 FIGURES: **YES**, after author review.

READY FOR FINAL PRINT DESIGN: **NO**.

READY FOR ISBN: **NO**.

Phase 2 has not begun. The author should review the substantial technical and source corrections before commissioning figures. This pass does not establish that every cited experiment reproduces on current systems, independently validate company reports, or remove the need to recheck living documentation and law.
