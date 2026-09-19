> **SUPERSEDED BY ORIGINAL MANUSCRIPT RESET.** Historical record only; the rewritten manuscript is not authoritative. See ORIGINAL-MANUSCRIPT-LOCK.md.

# Phase 0 manuscript audit

Audit date: 2026-09-18. Canonical source: the fifteen QMD files originally at the root of `Generative-AI/`. No old-book manuscript, assets, filters, or configuration imported. All chapters inspected before restructuring. `source-baseline.json` preserves exact original text and `chapter-inventory.csv` records its SHA-256 hashes; the original files were untracked, so Git has no earlier file history to move.

## Method and counts

The reproducible inventory excludes fenced examples from heading, table, and administrative-language detection. Word counts include all source text, including examples and metadata, with Unicode word boundaries. They are not prose-only publication counts. Source lines in audit CSVs refer to the immutable baseline, not the subsequently normalized files. Tables are Markdown pipe tables. Figures count captioned visual proposals plus the raw-LaTeX Marcus image. There are 53,687 source words, 15 chapters, 454 H2–H4 headings, 88 fenced blocks, 15 tables, 30 visual references (20 image requests and 10 caption-only proposals), and no native math nodes. The calculation in Chapter 3 is improperly fenced and will become native math.

## Structural decisions

Retain the manuscript's actual chapter titles. Remove duplicate YAML titles and manual `Chapter N:` prefixes. Normalize numbered main sections to H2, notably 3.1. Preserve enumerated H4 labels as part of their meaning, but make these minor subheads unnumbered to avoid double numbering. Demote Chapter 15's second H1 to an unnumbered H2: it is a transition within the fifteenth chapter, not a sixteenth chapter. Existing numbered-section references must become stable Quarto cross-references because introduction, learning outcomes, and terminology headings also affect automatic section numbers.

Use five Parts: I Foundations of Generative AI (1–3); II Building Generative AI Systems (4–7); III Research, Modalities, and Evaluation (8–11); IV Responsible Generative AI (12–14); V Production and the Frontier (15). Chapters 12, 13, and 14 remain separate: technical alignment, ethical/societal reasoning, and legal/governance frameworks.

## Editorial plan

`PROJECT_ARC.md` is **EDITORIAL_PLAN**, explicitly describing itself as a production reference. Move it to `editorial/PROJECT_ARC.md`; preserve its text. It describes a sixteen-week course, whereas the canonical manuscript has fifteen chapters. Its Week 16 presentation is a capstone activity, not another chapter. Do not render this plan.

## Fidelity and rendering risks

- All 20 referenced image files are absent from this project. Do not retrieve them from the old book. Preserve every source URI and description in the figure inventory and baseline. Review editions must visibly disclose missing illustrations, retaining the captions as proposed visuals. Missing original artwork remains an editorial limitation, even when there are no broken rendered asset links.
- Chapter 3 embeds a raw-LaTeX wrapfigure, negative vertical spacing, clearpage, and paragraph spacing. Convert that absent illustration to a portable missing-artwork note and remove layout-only commands; preserve its narrative unchanged.
- Chapter 3 contains long prompt templates, mixed prompt/response transcripts, pseudocode, data, program output, and application fragments. Label these distinctly. Never execute chapter examples during a book build. Technical inventory classifications describe intent, not correctness or production readiness.
- The manuscript has 15 potentially long terminology tables. Use wrapping columns and page-breakable tables in print; horizontal scrolling for HTML when needed. Track long technical lines in `overflow-inventory.csv`.
- No video placeholders found. Retain discussion questions, practical exercises, and the sustained research-assistant project. Generalize weeks, semester, submission, and course administration without altering technical uses of words such as module or real operational time periods.
- Source leakage checks must exclude intentional Markdown within prompt samples; headings in fenced templates are example content, not manuscript sections.

## Substantive issues deferred to technical review

Chapter 1: qualify categorical claims about discrimination/generation, emergent abilities, deterministic temperature-zero behavior, token splits, and architectural layer specialization. Verify AP's automation case against the distinction between templates and LLM generation.

Chapter 2: highest freshness priority. Specific GPT/Claude/Llama/PaLM model tiers, context sizes, rate limits, latency, costs, and alleged GPT-4 parameter counts need primary-source checks. The Sarah and company vignettes contain unverified numerical outcomes and must not be mistaken for sourced evidence. Its Markov-chain implementation and research-assistant usage references describe activities no longer present in Chapter 1. Preserve the comparison but generalize these false backward references. The chapter points to Chapter 6 for fine-tuning; correct that structural reference to Chapter 7.

Chapter 3: implementation promises conflict with the design-first milestone. Record for later reconciliation rather than remove examples. Claims that few-shot examples guarantee valid JSON, fixed optimal example counts, or explicit reasoning guarantees transparency need review. The chi-square example uses totals where failures are required and unpacks too few returned values; Python hash randomization is not stable across interpreter runs. Retain and flag, do not silently fix technical content in this phase. Safety instructions are not enforcement. Example confidence scores are not calibrated probabilities. The quantum-computing analogy in the scored response needs expert review.

Chapters 4–7: preserve integration, bounded-agent, retrieval/access-control, and customization design exercises. Check case-study details (Klarna, Glean, Harvey) and date the claims. Avoid treating RAG as guaranteeing truth or a fixed prompting→RAG→fine-tuning ladder as universal. Fine-tuning dataset-size and PEFT hardware claims require conditions and evidence.

Chapter 8: strong research-literacy foundation: source discovery, citation chaining, critical reading, ablations, synthesis, uncertainty, and a literature-grounded proposal. It is not yet a full GenAI empirical-methods chapter. Gaps: research-question operationalization, experimental controls, preregistration, data splits/contamination, sampling and power, random seeds and run variance, uncertainty intervals, reproducible artifacts/model-version records, human-subject consent and review, and systematic-review search/inclusion protocols. Keep its full manuscript and milestone.

Chapter 9: capability and reliability statements need dates and model conditions; distinguish jointly trained models from composed pipelines. Preserve the multimodal scope and human fallback in the accessibility case.

Chapter 10: automated evaluation, human rubrics, inter-rater reliability, LLM-as-judge biases/calibration, RAG faithfulness/attribution, regression tests, red teaming, and production monitoring are present. Benchmark construction, validity, sample size, confidence intervals, judge position bias/blinding, hallucination definitions, agent trajectory/tool-use evaluation, multimodal evaluation, and disaggregated bias/safety metrics lack worked methods. The agent-evaluation learning outcome currently exceeds the chapter's actual detail. These are priority next-phase gaps, not grounds to condense or replace the chapter.

Chapter 11: verify claims that compression necessarily improves speed, that all ensemble variants improve reliability, and that MoE selects a single expert per query. Keep cost/latency design work.

Chapter 12: preserve technical safety identity; DPO and newer post-training methods are gaps, and 'two dominant approaches' needs freshness review. Distinguish jailbreaks from prompt injection more carefully in technical revision.

Chapter 13: equalized odds is not merely equal accuracy, and federated learning alone is not a formal privacy guarantee. Record both as priority corrections for technical review. Ethical frameworks, distributive impacts, and environmental costs merit deeper treatment later.

Chapter 14: preserve standalone legal/governance chapter. EU AI Act timing, GDPR explanation claims, cross-border requirements, US policy, copyright litigation, and liability all require jurisdiction-specific, dated primary sources. No legal claims were automatically updated in Phase 0.

Chapter 15: coherent culmination linking architecture, deployment strategies, observability, cost, latency, lifecycle, and frontier discussion. Security and governance are mostly cross-references and need an integrated operational checklist later; rollback/data migrations, model-version pinning, incident response, and service-level objectives need worked examples. Frontier scaling, reasoning, and synthetic-data claims require dated evidence. Replace the nonexistent Chapter 16 reference with the final portfolio presentation.

## Glossary decision

A glossary is useful, but every chapter already has a populated terminology table. Retain all fifteen tables; defer consolidation to the technical-review phase to avoid introducing unreviewed or conflicting duplicate definitions. No empty glossary page.

## Secrets

The initial scan found no live credential-shaped values, private endpoints, private keys, or apparent personal contact identifiers. Named vignette characters and published authors are not credentials. Credential guidance and dummy examples are preserved. A repeatable scan is part of final QA; pattern scanning is not proof that all possible secrets are absent.

Author bio provenance: the first biographical paragraph in `deep-learning/frontmatter/about-author.qmd`, copied verbatim as a concise repository-derived bio. No new credentials or current appointment claims added.

Print-only symbol handling: thumbs-up/down emoji use explicit `[thumbs up]`/`[thumbs down]` labels; code arrows and tree branches use equivalent ASCII or direction labels. Original QMD and HTML/EPUB payloads retain every original character. This avoids unsupported-font loss without bundling fonts. The native worked calculation contains two math expressions; it replaces one original fenced block.

Additional parser repairs: Chapter 3's `Chapter Summary` lacked a preceding blank line and rendered as literal heading markers within a paragraph; Chapter 2's thematic break lacked separation and accidentally promoted an entire project paragraph to a Setext heading. Both are now semantically correct, with all wording retained. QA independently compares line-based headings with Pandoc headings to detect recurrence.

Chapter 3's few-shot paper-metadata examples also need explicit synthetic-example labeling during technical review; they are illustrative prompt payloads, not entries added to the verified bibliography.
