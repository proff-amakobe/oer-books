# Artificial Intelligence

An open textbook covering the foundations, methods, applications, deployment, and responsible practice of artificial intelligence.

## About the book

The 16 chapters progress from classical AI and reasoning through machine learning, neural networks, language, vision, reinforcement learning, expert systems, generative AI, ethics, MLOps, and future directions. The source preserves the original chapter content while using Quarto Book for navigation and publication.

## Read online

Read the deployed book at <https://proff-amakobe.github.io/oer-books/artificial-intelligence/>.

## Formats

The project configures HTML, EPUB, and PDF output. HTML is the deployment format. PDF generation requires a working LaTeX installation.

## Building locally

Install [Quarto](https://quarto.org), then run from this directory:

```bash
quarto preview
```

To build all configured formats:

```bash
quarto render
```

To build the deployable HTML edition only:

```bash
quarto render --to html
```

## Repository structure

- `_quarto.yml` defines book metadata, chapter order, formats, and navigation.
- `index.qmd` and `preface.qmd` contain the front matter.
- `chapters/` contains the 16 source chapters in reading order.
- `references.qmd` documents the current references approach.
- `.github/workflows/publish.yml` is a standalone Pages workflow template; in this monorepo, the root workflow must include this book.

## Contributing

Please report corrections through [GitHub issues](https://github.com/proff-amakobe/oer-books/issues) or submit a pull request. Preserve source meaning, identify sources for factual changes, use accessible Markdown, and confirm that the HTML edition renders successfully.

## License

This OER is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See `LICENSE` for the attribution requirements. The author name remains explicitly pending confirmation from the source owner.
