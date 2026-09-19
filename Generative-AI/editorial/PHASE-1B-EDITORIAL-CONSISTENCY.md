> **SUPERSEDED BY ORIGINAL MANUSCRIPT RESET.** Historical record only; the rewritten manuscript is not authoritative. See ORIGINAL-MANUSCRIPT-LOCK.md.

# Phase 1B — Editorial Consistency and Publication Cleanup

**Generative AI**

**Foundations, Systems, Evaluation, and Responsible Deployment**

Moody Amakobe · Global Data Science Institute · First Open Edition · 2026

Creative Commons Attribution 4.0 International (CC BY 4.0) · en-US

Print ISBN: TBD · Ebook ISBN: TBD

**Editorial consistency: PASS. Ready for Phase 2 figures: YES.** No figure generation, print redesign, or ISBN assignment was undertaken. The accepted Phase 1 scholarly review and its 2026-09-18 legal verification date are preserved.

## Preface and reader-facing pages

Stale review-process language removed: **PASS**. Temporary subtitle references: **0**. The Preface retains its opening account of mechanisms, engineering decisions, five Parts, and the sustained research-assistant portfolio. Its usage paragraph now includes project milestones and evaluation. Stale notices were also removed from the web landing page, copyright prose, and references introduction. Bibliographic verification dates remain visible without editorial phase labels. The Chapter 2 heading “Phase 1: Pre-Training” describes a training process, not publication status, and remains appropriate.

## Chapter openings and identities

Chapters reviewed: **15/15**. Learning Objectives sections: **17 → 15**; duplicate sections: **2 → 0**. Chapters 1 and 5 had two lists each. Chapter 1 retains its measurable list and adds the distinct generative-approach and ethical-evaluation goals from the earlier list. Chapter 5 preserves all five goals and explicitly retains choosing an architecture appropriate to a task. Obvious vague verbs and unsupported learning-outcome promises were revised.

- Chapter 12: **AI Safety, Alignment, and Interpretability**.
- Chapter 13: **Ethics and Responsible AI**.
- Chapter 14: **Law, Policy, and Governance of AI**.

Cross-reference, HTML navigation, PDF outline/TOC, and EPUB parity: **PASS**. Old Chapter 13 title occurrences in current reader-facing editions: **0**. Historical audit snapshots intentionally retain their original wording.

## TOC spacing

Double-digit number/title defects in Chapters 10–15: **28 → 0**. `styles/print.tex` reserves `3.8em` for section numbers through KOMA’s `tocnumwidth`. Semantic headings contain no inserted leading spaces. All 28 affected TOC entries have a measured **16.9 pt** number-to-title gap in the local PDF, including 10.10, 10.11, 10.12, 11.10, 12.10, 13.10, 14.10, and 15.10. PDF TOC pages 5–7, title page, and the Chapter 13 opening were rendered to PNG and visually inspected.

## Internal consistency

Audit records: **200**. Fixed: **154**. Consistent/no further action after review: **46**. Author review: **0**. Unresolved internal contradictions identified by this pass: **0**. Fixed records include editorial structure and figure-marker changes as well as claim corrections; they are not 154 newly researched technical findings.

| Topic | Result | Main alignment |
|---|---|---|
| Originality / novelty | PASS | Generation may reproduce training material; new combinations do not guarantee originality. |
| Next-token scope | PASS | Autoregressive LLMs distinguished from all generative and multimodal systems. |
| Temperature | PASS | Sampling diversity separated from creativity, factuality, and deterministic-output promises. |
| Few-shot counts | PASS | Removed context-size/count prescriptions and universal ordering advice; preserve task-specific comparison. |
| Chain-of-thought | PASS | Historical results retained; trace inspectability distinguished from faithful reasoning and correctness. |
| Role prompting | PASS | Context conditioning replaces unsupported activation-of-expertise language. |
| Prompt templates | PASS | Reusable patterns requiring evaluation, without universal proof or consistency guarantees. |
| Prompt security | PASS | Instructions and delimiters do not establish security; external permissions, testing, and residual risks remain explicit. |
| Model size and scenarios | PASS | Task benchmarking replaces deterministic size-to-quality rules; fictional medical outcomes clearly remain invented. |
| RAG | PASS | Instructions to ground an answer distinguished from successful grounding; source quality, citation checks, retrieval, and access control separated. |
| Fine-tuning / LoRA / QLoRA | PASS | No mandatory escalation; trainable fractions, memory feasibility, and quality remain configuration-dependent. |
| Multimodal | PASS | Summary promises scoped to model, input, and task; cross-modal alignment distinguished from behavioral alignment. |
| Evaluation / LLM judges | PASS | Accuracy, quality, success, preference, faithfulness, groundedness, citation correctness, precision, and recall distinguished. Judges require independent calibration. |
| Safety / ethics / law | PASS | Technical controls, normative responsibility, and jurisdiction-specific duties retain distinct chapter identities. |
| Terminology / objectives / discussion / wrap-ups | PASS | All 15 chapters scanned; substantive learning goals and distinct discussion questions preserved. |
| Punctuation and strong absolutes | PASS | Contextual scan retained justified requirements and conditional guarantees; known “jitteris” regression absent. |

The full decisions are in `PHASE-1B-CONSISTENCY-AUDIT.csv`; shared usage is documented in `TERMINOLOGY-CONSISTENCY.md`. `phase-1b-edit-log.json` replays the chapter changes against the accepted snapshot, with only standardized missing-artwork notices and blank-line cleanup handled separately. This guards against accidental paragraph loss as well as unexplained edits.

## Preserved evidence and completeness

| Measure | Before | After |
|---|---:|---:|
| Chapters / Parts | 15 / 5 | 15 / 5 |
| Source headings | 471 | 468 |
| Technical examples | 88 | 88 |
| Literal technical fences | 87 | 87 |
| Native worked-calculation example | 1 | 1 |
| Chapter terminology tables | 15 | 15 |
| Rendered substantive paragraphs checked | 888 | 888 |
| Verified bibliography entries | 97 | 97 |
| Further Reading items | 50 | 50 |
| Existing figure placeholders | 30 | 30 |
| Figure manifest candidates | 41 | 42 |

The three fewer headings are the two merged objective headings and the non-substantive “Illustration unavailable” callout heading, now a standard figure marker. **Missing substantive headings: 0; missing technical examples: 0; missing tables: 0; missing substantive paragraphs: 0.** All 88 example identities and their content are preserved; no API examples were executed. All 15 Further Reading sections are unchanged. The law chapter is byte-for-byte unchanged apart from its objective heading. Hash checks preserve the accepted Phase 1 queues, reviews, bibliography, and other recorded evidence; no freshness or citation queue was regenerated.

## Builds and checks

- **HTML: PASS.** Full source/render parity, chapter numbering, citation targets, internal links, search index, and download bundle checks pass. Browser checks passed for search results, Read Online navigation, technical-block display/copy controls, light/dark appearance, and mobile width. The in-app browser runtime was unavailable; headless Chrome was used for these checks.
- **PDF: PASS.** `output/pdf/Generative-AI-PHASE-1B-REVIEW.pdf`, **192 pages**, every page **612 × 792 pt**. The page count is an observed result, not a target. Technical-line, table-cell, paragraph, heading, bibliography, page-boundary, and TOC-spacing checks pass.
- **EPUB: PASS.** Content parity, XML/ZIP integrity, metadata, navigation, internal links, native math, and technical payload checks pass. This is not a formal EPUBCheck certification.
- `scripts/qa/verify_book.py`: **PASS**, zero errors.
- `scripts/qa/verify_phase1b.py`: **PASS**, zero errors.

Standalone Pandoc comparison emits warnings for Quarto section-reference identifiers because it does not resolve book cross-references. The verifier independently resolves the rendered labels and validates the actual link targets; these warnings do not represent broken reader-facing citations.

## Figures and next phase

Figures generated: **0**. Pending candidates: **42** — **30 existing placeholders** plus **12 additional planned candidates**. The old manifest contained 29 numbered proposals and omitted the unnumbered Marcus vignette; adding that existing placeholder accounts for the one-row increase. Every marker maps to one unique candidate. No duplicate visual proposals were added.

**READY FOR PHASE 2 FIGURES: YES**

**READY FOR FINAL PRINT DESIGN: NO**

**READY FOR ISBN: NO**

## Publication gate

The shared workflow’s previous run failed at `tlmgr: command not found` before rendering. The narrowly scoped deployment correction resolves the TinyTeX package-manager path installed by Quarto and adds the Phase 1B regression checks. Other publications’ sources and build commands remain unchanged. A successful GitHub Actions run for the pushed revision and direct checks of the live book and downloads are required for delivery; the run and live URLs accompany the final delivery record.
