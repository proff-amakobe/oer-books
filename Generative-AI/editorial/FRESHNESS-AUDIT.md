# Freshness audit — Phase 1 adjudication

Verification date: **2026-09-18**. All **447 original candidates** are retained, in order, in `freshness-items.csv`. Their original claims and line numbers remain intact; `freshness-items-phase-0.csv` preserves the exact incoming queue. A candidate is not necessarily an error.

Classification describes the original candidate and its disposition: historical facts can be durable; outdated deployment advice is replaced by dated examples; unsupported precision is removed or generalized. `correction`, `current_excerpt`, and `resolution` identify what the manuscript now says. Illustrations, exercises, author judgments, and fictional scenarios do not become empirical findings by receiving a review record. Empty primary-source fields on those records mean no external result is asserted, not that an experiment was verified.

- CURRENT_AND_DURABLE: 178
- ILLUSTRATIVE_ONLY: 133
- REMOVE_OR_GENERALIZE: 109
- OUTDATED: 17
- CURRENT_BUT_TIME_SENSITIVE: 10
- UNVERIFIED: 0
- NEEDS_AUTHOR_REVIEW: 0

The review separates model-selection principles from dated 2022–2024 provider examples. It removes proprietary parameter guesses and universal pricing, latency, throughput, rate-limit, chunking, and dataset-size recommendations. Company reports remain dated and attributed; legal examples state jurisdiction and status. No bulk substitution of the newest product names was performed.

See `PHASE-1-TECHNICAL-AND-SCHOLARLY-REVIEW.md`, `technical-corrections.csv`, and `phase-1-sources.json`. Do not rerun the Phase 0 queue generator over these adjudicated records.
