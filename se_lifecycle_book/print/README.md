# Professional print edition

Build the dedicated commercial-print interior without changing HTML or EPUB:

```sh
quarto render --profile print --to pdf
```

Output: `_book/The-Complete-Software-Engineering-Lifecycle-Print.pdf`

The complete-edition print filter adds one intentional final blank page so the
perfect-bound interior is exactly 864 pages. Cover production details are in
`COVER-PRODUCTION.md`.

## Independent two-volume editions

The complete website remains the default build. The two standalone editions
select the same canonical chapter files through mutually exclusive profiles:

```sh
quarto render --profile volume1
quarto render --profile volume2
```

Each command builds PDF and EPUB together so one format does not clean the
other from its isolated output directory:

- `output/volume1/Software-Engineering-Foundations-and-Design.pdf`
- `output/volume1/Software-Engineering-Foundations-and-Design.epub`
- `output/volume2/Software-Delivery-Operations-and-Evolution.pdf`
- `output/volume2/Software-Delivery-Operations-and-Evolution.epub`

The standalone profiles end on their final chapter content. They do not
include the canonical glossary, the former Volume I closing note, or an
artificial final verso.

Do not render the two formats separately into the same directory. Quarto's
output cleanup can remove the previously rendered format.
