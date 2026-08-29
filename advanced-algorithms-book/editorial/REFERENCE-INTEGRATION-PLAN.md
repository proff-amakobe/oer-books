# Reference Integration Plan

## Trust boundary

`references.bib` remains present but untrusted. Its only entry, `knuth84`, is an invalid metadata collision: the author/title describe this textbook, while the DOI resolves to an unrelated 1984 *Computer Journal* article. It must not be cited or copied.

`references-second-edition.staging.bib` contains only records independently checked against DOI/publisher, DBLP, institutional, or original-paper metadata. It is staging material, not yet the production bibliography.

## Chapter priorities

- Chapters 2–6: cite original papers selectively for named foundational algorithms and use a recognized algorithms textbook for routine definitions and bounds.
- Chapter 7: cite Cook and Karp for NP-completeness and use a standard complexity text for definitions and reductions.
- Chapters 8–12: cite original approximation, flow, string, numerical, and advanced-data-structure sources where historical attribution or a nontrivial guarantee is taught.
- Chapter 13: require primary papers or official institutional documentation for every historical, empirical, deployment, and time-sensitive claim.
- Chapters 14–15: prefer official tool documentation and reproducibility/experimental-method sources; do not source ordinary author guidance unnecessarily.

## Integration method

1. Resolve each `REQUIRES SOURCE` row in `citation-manifest.csv`.
2. Add a record to staging only after title, author, venue, year, and DOI/official URL agree.
3. Insert Pandoc citation keys at the claim level, avoiding citation clutter on repeated nearby mentions.
4. Run duplicate-key, DOI/title, cited/uncited, and rendered-reference checks.
5. Replace the production bibliography only after the manifest is sufficiently complete and the invalid original record is quarantined.

## Bibliography organization and style

Use one consolidated bibliography rather than chapter-specific lists; this avoids duplicate canonical works and supports all formats consistently. A clean author-year CSL style is recommended for a textbook because names and historical context matter, but global citation styling should wait for the citation-integration phase.
