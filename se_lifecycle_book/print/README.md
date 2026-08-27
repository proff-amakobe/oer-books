# Professional print edition

Build the dedicated commercial-print interior without changing HTML or EPUB:

```sh
quarto render --profile print --to pdf
```

Output: `_book/The-Complete-Software-Engineering-Lifecycle-Print.pdf`

The print filter adds one intentional final blank page so the perfect-bound
interior is exactly 864 pages. Cover production details are in
`COVER-PRODUCTION.md`.
