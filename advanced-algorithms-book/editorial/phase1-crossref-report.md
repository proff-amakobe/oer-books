# Phase 1 Cross-Reference Report

## Scope and result

Heading normalization removed manual numbering from headings only. A fence-aware scan found **54 prose lines** containing hard-coded chapter/section references: 31 `Section N.N` mentions and 24 `Chapter N` mentions (one line contains both categories). These are prose references, reading-roadmap labels, or external-book chapter references—not source heading prefixes.

No Pandoc `@sec-*` cross-reference labels were present to break. Public chapter filenames were not changed. New semantic anchors are derived from heading text (for example, `#heaps-and-priority-queues`, `#the-divide-and-conquer-paradigm`, and `#segment-trees-range-queries-on-steroids`).

## References preserved

- Chapter roadmaps in Chapters 2–6 refer to their intended Section 2.1–6.7 ranges. Because opener headings and descendants are explicitly unnumbered, these labels still agree with generated numbering.
- Forward/back references such as “Chapter 1,” “Chapter 2,” “Chapter 8,” and “Chapter 9” retain the intended 1–15 substantive chapter sequence.
- External references such as “CLRS Chapter 4” and “Kleinberg & Tardos Chapter 5” are legitimate bibliographic prose and were not changed.

## Pre-existing items requiring later editorial review

- Chapter 3 says Chapter 5 will cover graph algorithms, but the current Chapter 5 is Dynamic Programming.
- The closing material in Chapter 5 points backward to Chapter 4 as the next greedy chapter and calls Chapter 5 a future data-structures chapter.
- Several roadmap summaries describe section contents that no longer exactly match the current manuscript organization.

These discrepancies predate Phase 1 and were not caused by heading normalization. In accordance with scope, they remain for the technical/pedagogical edit rather than being broadly rewritten here.

## QA

- Remaining manual numeric heading defects outside fenced content: **0**.
- HTML chapter sequence: **1–15 PASS**.
- PDF TOC/bookmarks: **Chapter 1–Chapter 15 PASS**.
- EPUB navigation/headings: **1–15 PASS**.

