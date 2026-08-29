# Print Format Audit

## Baseline

The source requests `papersize: "7in,10in"`, 11 pt, `scrbook`. The rendered PDF is **690 pages, A4 (595.28 × 841.89 pt), 1,582,315 bytes, unencrypted, untagged**, with all ten reported font subsets embedded. Thus 7 × 10 is the stated intent, not the actual output. The `papersize` value is not being applied by the current LaTeX route.

## Assessment

At 7 × 10, prose is comfortable and the object is portable, but the manuscript’s 537 code blocks, wide comments, matrices, tables, recursion trees, and networks need aggressive line-length discipline. A finished book near the current 690 A4 pages would become substantially longer at 7 × 10, increasing spine width, weight, cost, and awkward code/page breaks.

At 8.5 × 11, code, pseudocode, equations, DP matrices, graphs, and classroom photocopying gain meaningful width. It supports side-by-side comparisons and fewer forced breaks, but feels more like a course text/manual, is less portable, and must be assessed against the chosen Ingram binding, paper, ink, and page-count limits.

## Recommendation

Use **8.5 × 11 in** as the working recommendation for the Second Edition, subject to an Ingram cost/binding proof after content stabilization. If portability is the overriding product goal, test 7 × 10 only after enforcing a 72–76-character code standard and reducing page count. Either route requires explicit geometry (`paperwidth`/`paperheight` or a verified KOMA option), crop-size QA, printed proofs, and PDF-box inspection.

Layout QA must address code overflow, wide tables, orphan headings, widows, blank pages, chapter openings, equations, Unicode, headers, TOC, and bookmarks. The nine-run convergence warning suggests unstable references/TOC and is a build-reliability risk.

