# Volume Cross-Reference Audit

Audit date: 2026-08-27

## Method

The canonical QMD tree was searched for literal `Chapter N`, `Section N`, `Figure N`, and `Table N` references. Final PDFs were then searched for stale Volume II master numbers 9–15. Canonical chapter files were not duplicated or renumbered.

## Findings

| Source | Finding | Classification | Resolution |
|---|---:|---|---|
| `chapters/glossary.qmd` | 152 literal chapter references, including index-by-chapter headings | Edition-sensitive | `filters/volume-crossrefs.lua` maps local chapters and adds an explicit cross-volume label dynamically. |
| `chapters/06-agile-methodologies.qmd:1502` | Chapter 2 | Volume I local | Remains Chapter 2 in Volume I. |
| `chapters/07-version-control.qmd:20` | Chapter 1 | Volume I local | Remains Chapter 1 in Volume I. |
| `chapters/02-requirements-engineering.qmd:735` | Section 3 | Template-document section, not a book section | PASS; no edition change. |
| `chapters/13-maintenance-evolution.qmd:1184` | Section 4.2 | Code-comment formula reference, not a book section | PASS; no edition change. |
| `chapters/14-ethics-professionalism.qmd:603` | Section 508 | Name of United States accessibility law | PASS; must not be renumbered. |

No literal figure-number or table-number references were found in canonical prose outside automatically generated material. Existing semantic figure IDs remain source-stable and Quarto assigns edition-local numbers.

## Dynamic Results

- Volume I: 91 glossary references to master chapters 9–15 render as explicit `Volume II, Chapter 1–7` references.
- Volume II: 53 glossary references to master chapters 1–8 render as explicit `Volume I, Chapter 1–8` references.
- Volume II master chapters 9–15 render locally as Chapters 1–7.
- Final Volume II PDF contains zero stale `Chapter 9` through `Chapter 15` references.

The filter changes presentation only. The complete OER retains canonical Chapter 1–15 language and numbering.
