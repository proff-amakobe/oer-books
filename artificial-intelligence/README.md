# Engineering Intelligent Systems

**Designing, Building, and Deploying Modern Artificial Intelligence**

First Open Edition · 2026 · Moody Amakobe

## About the Book

*Engineering Intelligent Systems* is a systems-oriented open textbook connecting artificial intelligence theory with implementation, deployment, and responsible practice. Its 16 chapters move from agents, search, knowledge, planning, and uncertainty through modern learning, language, vision, generative AI, ethics, and MLOps.

## Read Online

Read the published book at <https://proff-amakobe.github.io/oer-books/artificial-intelligence/>.

## Contents

1. Foundations of Intelligent Systems
2. Search, Reasoning, and Decision Making
3. Machine Learning Systems
4. Perceptual, Adaptive, and Generative Systems
5. Engineering, Deployment, and Responsible AI

## Audience

The book is intended for graduate and advanced undergraduate students, AI instructors, technical professionals, and self-directed learners. Basic programming literacy is expected; introductory probability, statistics, and algebra are helpful.

## Formats

The Quarto project publishes HTML, PDF, and EPUB editions. The HTML edition is deployed automatically with GitHub Pages.

## Build the Web Edition

Install [Quarto](https://quarto.org), then run from this directory:

```bash
quarto preview
quarto render
```

To build only the deployed HTML edition:

```bash
quarto render --to html
```

## Build the Print Edition

The print interior uses LuaLaTeX, open fonts supplied by TeX Live, and reusable components in `print/`:

```bash
quarto render --profile print --to pdf
```

This creates `_book/Engineering-Intelligent-Systems-Print.pdf` with letter-size pages, mirrored margins, and recto Part and chapter starts.

To create the closely matched, screen-oriented digital PDF with symmetric margins:

```bash
quarto render --profile digital,print --to pdf
```

TinyTeX or another current TeX Live installation with LuaLaTeX, TikZ, `tcolorbox`, `titlesec`, `fancyhdr`, `booktabs`, and `longtable` is required.

## Contributing

Corrections, accessibility improvements, examples, and carefully sourced updates are welcome through [pull requests](https://github.com/proff-amakobe/oer-books/pulls). Preserve the author's substantive meaning and confirm that the book renders successfully before submitting changes.

## Reporting Errors

Report technical or editorial problems through the repository's [issue tracker](https://github.com/proff-amakobe/oer-books/issues). Include the chapter, section, and enough context to reproduce the problem.

## Citation

Amakobe, M. (2026). *Engineering intelligent systems: Designing, building, and deploying modern artificial intelligence* (First Open Edition). https://proff-amakobe.github.io/oer-books/artificial-intelligence/

## License

Except where otherwise indicated, this Open Educational Resource is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE).
