# Print Pagination Audit

## Baseline

| Metric | Before optimization | After optimization |
|---|---:|---:|
| Physical pages | 1,076 | 863 |
| Body type | TeX Gyre Pagella, 10.75 pt | TeX Gyre Pagella, 10.15 pt |
| Body leading | 12.69 pt (1.18) | 12.38 pt (1.22) |
| Code type | Latin Modern Mono, approximately 9 pt | Latin Modern Mono, 8.1 pt with 9.5 pt leading |
| Geometry | 1.10 in inner, .76 in outer, .76 in top, .87 in bottom, .12 in binding | 1.00 in inner, .73 in outer, .70 in top, .81 in bottom, .10 in binding |
| Chapter opening | `openright` | `openany` |
| Extracted blank pages | 14 | 5 intentional front-matter versos |
| Terminal-chrome-only pages | Approximately 62; confirmed at the supplied regression locations | 0 |
| Contents | 8 physical pages | 5 physical pages |
| List of Figures | 3 physical pages | 3 compact physical pages |
| List of Tables | 1 empty page | Removed; no formally captioned Pandoc tables currently register |

The baseline PDF is US Letter and uses the established professional print design. Its primary pagination defect is that FancyVerb listings are indivisible inside an otherwise breakable terminal box, allowing terminal chrome to be stranded while the code moves to the next page.

## Optimization decisions

- Preserve all instructional prose and code.
- Replace the terminal wrapper with one integrated, breakable `tcblisting` environment.
- Change chapter opening from `openright` to `openany`.
- Compact chapter heroes and Learning Objectives without changing their identity.
- Use 10.15 pt body type with approximately 12.38 pt leading.
- Reduce default figure width to 72% of the text width and maximum height to 62% of text height; explicit manuscript widths still take precedence.
- Remove the empty List of Tables until the manuscript contains formally captioned Pandoc tables.

## Final results

The optimized edition contains 863 physical pages, saving 213 pages (19.8%). Savings interact rather than map perfectly to one feature; the largest contribution came from integrating code into breakable terminal listings, followed by tighter component/heading spacing, modest body and code typography changes, `openany`, smaller figures, and front-matter cleanup.

All 15 chapter openers were rendered for inspection. The compressed hero and objectives card keep normal objective sets on their opener, including the longer Chapter 4, Chapter 9, and Chapter 11 sets. Section headings now reserve a conservative amount of following content. The print TOC retains chapters and core early instructional sections while suppressing glossary letters and later administrative/end matter.

The final PDF is US Letter, unencrypted, contains no extracted `Figure ??`, `Table ??`, or `Chapter ??` references, and has no detected terminal-chrome-only pages. SVG-generated PDF resources still expose Type 3 font records during `pdffonts` inspection; the primary TeX Gyre and Latin Modern text fonts are embedded.
