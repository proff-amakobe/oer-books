# Citation and Reference Audit

Status: **CRITICAL — bibliography is not publication-ready and must be treated as untrusted.** No records were changed or invented in this pass.

## Evidence

- `references.bib` contains one record, key `knuth84`, but the record names Moody A. Amakobe and *Advanced Computational Algorithms* while its DOI/URL, journal, volume, issue, pages, ISSN, and 1984-style key indicate a different publication. This is a likely composite/placeholder record: **REQUIRES VERIFICATION**.
- No manuscript citation invokes `@knuth84`; therefore the sole bibliography record is uncited.
- `references.qmd` is a valid empty bibliography container, but it is absent from `_quarto.yml`; it is currently unused.
- The chapters contain no functioning Pandoc citation keys. Python decorators such as `@staticmethod` are code, not citations.
- Chapter 13 contains at least 20 author-year attributions absent from the bibliography: Kraska et al. (2018), Dwork et al. (2006), Kleinberg et al. (2016), Zemel et al. (2013), Rumelhart et al. (1986), Kaplan et al. (2020), Child et al. (2019), Wang et al. (2020), Dao et al. (2022), Watkins (1989), Mnih et al. (2015), Williams (1992), Schulman et al. (2017), Flajolet et al. (2007), Bloom (1970), Google/Pregel (2010), Lamport (1982), and the named Shor/HHL sources. Each is **SOURCE REQUIRED**.
- Attributed quotations and historical superlatives are not sourced. Alan Kay wording, “most influential algorithm of the 20th century,” company deployment claims, theorem history, and quantitative industry claims all require primary or authoritative sources.
- The “Additional Resources” and “Further Reading” sections name books, platforms, conferences, and papers as prose rather than traceable bibliography entries.

## Issue count

Minimum discrete citation-integrity issues: **22** (one suspicious record, one uncited record condition, and at least 20 absent author-year sources). This is a lower bound; a claim-level reconstruction will produce more.

## Required reconstruction

1. Quarantine `knuth84` pending record-by-record verification; do not reuse its DOI.
2. Establish one citation style and add `bibliography: references.bib` at project level in the future architecture.
3. Convert verified author-year mentions to citation keys only after checking primary sources.
4. Source quotations, theorem attributions, historical claims, approximation ratios, deployment statistics, and date-sensitive claims.
5. Run BibTeX parsing, duplicate-key, DOI-resolution, cited/uncited, and rendered-reference QA. Any unresolved item remains **REQUIRES VERIFICATION**.

