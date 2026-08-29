# Web Second Edition Audit

## Baseline

The local rendered edition uses Quarto 1.7.31 and the stock Cosmo theme. Search, sidebar navigation, breadcrumbs, copy buttons, and all 15 stable chapter paths are present. The homepage is titled only “Preface,” numbers itself as Chapter 8, and offers no actual PDF/EPUB links despite promising downloads.

The HTML head has author/date metadata but no observed description, canonical URL, OpenGraph, Twitter card, or JSON-LD. No project-level sitemap or robots policy is present. The sole source image has meaningful cover context; the remote CC badge has alt text, but external loading is fragile. Diagram-like content is mostly `<pre>` text, which is poor on mobile and for assistive technology. Wide code/tables need responsive testing.

## Stable URL baseline

Base: `https://proff-amakobe.github.io/oer-books/advanced-algorithms-book/`

Chapter URLs are `/chapters/01-introduction.html` through `/chapters/15-Final-Presentations.html`, preserving the exact existing filenames and capitalization. Front pages are `/title.html`, `/edition.html`, `/copyright.html`, `/dedication.html`, `/about-author.html`, `/about-gdsi.html`, and `/how-to-use.html`; the preface is `/index.html`.

## Second Edition direction

- Preserve every chapter filename or add redirects if movement becomes necessary.
- Make the homepage a clear book landing/preface page with cover, edition, description, audience, and working HTML/PDF/EPUB actions.
- Add canonical base URL, descriptions, OpenGraph/Twitter metadata, Book JSON-LD, sitemap, and robots policy.
- Replace manual ASCII instructional figures with responsive accessible SVG plus captions/descriptions.
- Add a restrained custom theme, visible focus states, WCAG contrast, semantic tables, descriptive links, and mobile overflow handling.
- Keep search and breadcrumbs, but ensure front matter is unnumbered and chapters show 1–15.
- Validate keyboard navigation, heading order, landmarks, alt text, zoom/reflow, code scrolling, table headers, and reduced-motion behavior.

