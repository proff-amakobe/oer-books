# Deep Learning: A Comprehensive Guide

**Architectures, Training, Generative Models, and Real-World AI Systems**

First Open Edition · 2026 · Moody Amakobe

## About the Book

*Deep Learning: A Comprehensive Guide* is a 16-chapter Open Educational Resource connecting neural-network foundations and mathematical training principles with vision, language, multimodal learning, generative systems, reinforcement learning, production engineering, governance, and responsible practice.

## Read Online

Read the published edition at <https://proff-amakobe.github.io/oer-books/deep-learning/>.

## Book Structure

1. Foundations of Deep Learning
2. Vision Systems
3. Sequence, Language, and Multimodal Learning
4. Generative and Adaptive Systems
5. Engineering and Responsible Deep Learning

## Audience

The book is intended for graduate and advanced undergraduate students, instructors, technical professionals, and independent learners. Basic programming literacy is expected; algebra, introductory probability and statistics, and foundational machine-learning concepts are helpful.

## Formats

The Quarto project publishes HTML, EPUB, a digital PDF, and an independently configured professional print interior.

## Building Locally

Install [Quarto](https://quarto.org), then run from this directory:

```bash
quarto check
quarto render
```

To build only the deployed web edition:

```bash
quarto render --to html
```

## Print Edition

The print interior uses an isolated Quarto profile and modular LuaLaTeX design files under `print/`. It does not replace the web, EPUB, or digital PDF configuration.

Required software:

- Quarto 1.7 or later
- TeX Live/TinyTeX with LuaLaTeX
- The open TeX Gyre Pagella and TeX Gyre Heros fonts supplied through TeX Live

Build the professional print interior with:

```bash
quarto render --profile print --to pdf
```

The output is `_book/Deep-Learning-A-Comprehensive-Guide-Print.pdf`. It is an 8.5 × 11 inch, two-sided, no-bleed interior and intentionally does not include the external digital cover as page 1. The ordinary `quarto render` command continues to produce the digital formats.

Final commercial cover production is documented in `print/COVER-PRODUCTION.md` and must wait for printer-specific spine and bleed specifications.

Chapter examples are static by design; rendering does not require deep-learning frameworks or a GPU.

## Contributing

Corrections, accessibility improvements, verified references, and carefully sourced technical updates are welcome through pull requests. Preserve the author’s substantive meaning and render the HTML edition before submitting changes.

## Reporting Errors

Report editorial or technical problems through the repository’s [issue tracker](https://github.com/proff-amakobe/oer-books/issues). Include the chapter, section, and enough context to reproduce the problem.

## Citation

Amakobe, M. (2026). *Deep learning: A comprehensive guide: Architectures, training, generative models, and real-world AI systems* (First Open Edition). https://proff-amakobe.github.io/oer-books/deep-learning/

## License

Except where otherwise indicated, this Open Educational Resource is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE).
