# Phase 2 Freshness Handoff

This handoff preserves the 121 Chapter 13 candidates identified in `freshness-audit.md`. Phase 2 changed only claims whose freshness or sourcing directly affected technical correctness.

## TECHNICALLY WRONG NOW

- Resolved in Phase 2: the “smoothed quicksort” experiment measured Python's built-in sort rather than quicksort.
- Resolved in Phase 2: HyperLogLog presented per-register O(log log n) storage as the total storage bound.
- No known unresolved item is classified technically wrong solely because of age; unresolved claims remain source/date dependent.

## LIKELY OUTDATED

- Quantum hardware counts, access, error-correction ratios, and factoring timelines.
- GPT model sizes, context lengths, “state-of-the-art” labels, and named model comparisons.
- Platform volumes, company deployments, and infrastructure scale.
- Cryptographic-transition timelines and cryptocurrency energy figures.

## DATE-SENSITIVE

- All “current,” “today,” “recent,” “already,” and future-year statements.
- Company valuations, subscriber counts, savings, adoption, and deployment statistics.
- Legal, standards, governance, and policy claims.
- Conference/resource lists framed as the last three or five years.

## SAFE TO DEFER

- Evergreen definitions of MapReduce patterns, Bloom filters, Count-Min Sketch, differential privacy, backpropagation, and basic quantum algorithm ideas, provided their historical and deployment attributions are sourced later.
- The full set of 121 candidates is deferred to Phase 3; none should be treated as verified merely because it appears in this category.
