# Low-Utilization Page Audit

The baseline scan identified the supplied terminal-chrome-only regression pages and 14 extracted blank pages. The final scan found no terminal-title-only pages.

## Final results

| Physical pages | Reason | Resolution |
|---|---|---|
| 2, 4, 6, 8, 10 | Intentional front-matter verso blanks separating half title, title, copyright, preface, acknowledgments, and contents | Retained as publication-quality blanks |
| Former terminal-only regression pages | Terminal chrome separated from indivisible FancyVerb listing | Replaced with an integrated breakable `tcblisting`; no terminal-only pages remain |
| Former chapter-opener objective continuations | Hero and objectives card exceeded the page | Compressed hero height, card padding, objective leading, and item spacing |
| Empty List of Tables | No formally captioned Pandoc tables registered | Removed from print front matter |
| Blank before List of Figures | `cleardoublepage` forced an unnecessary verso | Replaced with `clearpage` |

An automated text-density scan found no additional non-front-matter page containing only terminal chrome. Visual checks covered the TOC, populated List of Figures, representative chapter openers, continued listings, Kubernetes code, chapter-end material, and glossary transition.
