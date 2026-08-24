# Deep Learning: A Comprehensive Guide

## About the Book

*Deep Learning: A Comprehensive Guide* is a 16-chapter Open Educational Resource covering neural-network foundations, training, vision, language, multimodal and generative systems, reinforcement learning, deployment, governance, and responsible practice.

## Read Online

The web edition is intended for <https://proff-amakobe.github.io/oer-books/deep-learning/>. This link should be considered confirmed after the GitHub Pages workflow completes successfully.

## Contents

1. Foundations of Deep Learning
2. Vision Systems
3. Sequence, Language, and Multimodal Learning
4. Generative and Adaptive Systems
5. Engineering and Responsible Deep Learning

## Audience

The book is intended for students, instructors, technical professionals, and independent learners. Basic programming literacy is expected; algebra, introductory probability and statistics, and machine-learning fundamentals are helpful.

## Formats

HTML is the primary format. The Quarto configuration also supports EPUB and a basic PDF edition. A later publication phase will address professional print design.

## Building Locally

Install [Quarto](https://quarto.org), then run from this directory:

```bash
quarto check
quarto render --to html
```

Optional formats:

```bash
quarto render --to epub
quarto render --to pdf
```

Chapter examples are static by design; rendering does not require deep-learning frameworks or a GPU.

## Repository Structure

- `chapters/` — the 16 source chapters in reading order
- `frontmatter/` — concise Phase 1 publishing pages
- `parts/` — Quarto Part introductions
- `assets/` — reserved for shared book assets
- `_quarto.yml` — book navigation and output configuration
- `references.qmd` — references infrastructure and source-status note

## Contributing

Corrections, accessibility improvements, and carefully sourced updates are welcome through pull requests. Preserve the author’s substantive meaning and render the HTML edition before submitting changes.

## Reporting Errors

Report technical or editorial problems through the repository’s [issue tracker](https://github.com/proff-amakobe/oer-books/issues). Include the chapter and section where the problem occurs.

## License

Except where otherwise indicated, this Open Educational Resource is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE).

