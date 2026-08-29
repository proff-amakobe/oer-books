# Advanced Computational Algorithms
## Phase 2 Technical Correctness and Citation Preparation

### Executive Summary

Overall technical status: **COMPLETE WITH DOCUMENTED MANUAL-REVIEW ITEMS**.

Code blocks reviewed: 537
Runnable candidates: 289
Safely executed definition-only blocks: 175
PASS: 175
Context-dependent Python blocks that compile: 89
Isolated failures: 11
Compile-failing fragments: 4
Pseudocode/illustrative/unlabeled: 242
Behavioral verification groups: 12 of 12 PASS

An isolated failure does not necessarily mean an algorithm is wrong: the 11 failures reference names intentionally established in surrounding examples. They remain FAIL in the machine-readable isolated-execution report rather than being promoted to PASS.

### Technical Findings

Critical: 0
High: 3 — misleading quicksort benchmark, incorrect max-flow algorithm/bound pairing, odd-size Strassen output defect.
Medium: 5 — TSP decision/optimization terminology, cuckoo insertion bound, FFT allocation/peak-space distinction, empty-pattern inconsistency, HyperLogLog total-space statement.
Low: 1 — unsupported quantitative industry anecdote replaced by the formal distinction it was intended to motivate.

### Repairs Applied

- Replaced the Chapter 13 `sorted()` benchmark with explicit deterministic quicksort and comparison counting; stated that the experiment is illustrative rather than a proof.
- Corrected maximum-flow’s polynomial example to Edmonds–Karp, O(VE²).
- Corrected optimization TSP from NP-complete to NP-hard and removed unsupported performance/guarantee figures.
- Corrected cuckoo-hashing insertion to expected amortized O(1) under hashing assumptions with an O(n) rebuild caveat.
- Distinguished recursive FFT O(N) peak auxiliary space from O(N log N) total allocation.
- Made naïve and KMP search agree on empty-pattern semantics.
- Preserved original Strassen output dimensions after odd-size padding.
- Corrected HyperLogLog from a per-register bound to O(m log log n) total register bits.

### Chapter Status

Chapter 1: REQUIRES REVISION
Chapter 2: REQUIRES REVISION
Chapters 3–12: PASS WITH MINOR FIXES
Chapter 13: REQUIRES REVISION
Chapters 14–15: REQUIRES REVISION

Detailed evidence is in `editorial/technical/chapter-01.md` through `chapter-15.md`. “Requires revision” marks fragmented examples, source-heavy material, or intentionally deferred chapter work; it does not represent a structural build failure.

### Code Verification

Total: 537
Executed: 175 isolated definition-only blocks plus 12 targeted behavioral groups
Passed: 175 isolated blocks; 12/12 behavioral groups
Failed: 11 isolated context failures; 4 compile-failing fragments
Dependency/toolchain blocked: 2 tagged non-Python blocks plus context-dependent imports
Manual/not executed: 258
Corrected examples: 3 code implementations, plus five associated technical prose claims

### Complexity Audit

Claims inventoried: 343 explicit prose-level complexity items, plus block-local claims.
Corrections: 5 complexity/space/terminology corrections.
Unresolved: assumptions and advanced bounds marked MANUAL REVIEW in the technical inventory.

### Proof Audit

Proof/proof-sketch items inventoried: 89.
Correct: not claimed globally without formal rechecking.
Informal but valid: dominant category in chapter reports.
Incomplete/incorrect/manual: retained as MANUAL REVIEW where a full formal pass would require subsection-level revision.
The manuscript must not advertise all proofs as formally verified.

### Citation Audit

Claims/source-map rows: 239 (230 conservative claim candidates plus 9 verified foundational mappings).
Verified sources: 9 staging records.
Staging BibTeX entries: 9.
Unverified sources: 230 manifest rows pending claim-level integration (the staged canonical works cover themes, not every row).
Invalid original bibliography entries: 1.

### Existing Bibliography

Trust status: **UNTRUSTED — DO NOT CITE**.
Malformed/mismatched records: 1, `knuth84`, classified **INVALID — METADATA COLLISION**.

### Freshness Handoff

Technically wrong and repaired now: 2 Chapter 13 items.
Likely outdated/date-sensitive: the clusters in `phase2-freshness-handoff.md`.
Deferred: all 121 original freshness candidates remain in Phase 3 scope.

### Structural Regression

HTML: PASS
PDF: PASS
EPUB: PASS
Chapter numbering: PASS (1–15 in HTML and EPUB)
Trim: PASS (612 × 792 pt; 8.5 × 11 in)
EPUB language: PASS (`en-US`)
URLs: PASS (chapter filenames and download targets preserved)

### Pagination

Before: 716
After: 716

### Major Revisions Deferred

- Reconstruct fragmented multi-file “complete” examples and label exercise placeholders.
- Perform a formal proof pass where the current text makes theorem-strength claims.
- Complete claim-level citation insertion after resolving the manifest.
- Perform the dedicated 2026 Chapter 13 freshness rewrite.
- Preserve the planned later figure and code-presentation redesigns.

### Recommendation

Proceed to **Phase 3 — Content Modernization and Pedagogical Revision** after final structural regression QA. Phase 3 must treat the manual-review and source-required rows as open work, not as implicit passes.
