# Software Engineering Cover Production

## Current production lock

| Field | Locked value |
|---|---|
| Final interior pages | 864, including an intentional final blank |
| Interior PDF | `_book/The-Complete-Software-Engineering-Lifecycle-Print.pdf` |
| Print ISBN | 979-8-2957-6070-9 |
| Template | `print/cover/template/9798295760709-Perfect.pdf` |
| Trim | 8.5 x 11 inches |
| Binding | Paperback - Perfect Bound |
| Interior / paper | Color / Color 50 |
| Cover finish | Matte |
| Spine | 1.710 inches |
| Full template sheet | 21 x 12 inches |
| Bleed artwork | 18.96 x 11.25 inches |
| Final cover | `output/pdf/The-Complete-Software-Engineering-Lifecycle-Ingram-Cover.pdf` |

The previous 886-page count and 1.75180-inch spine are **OBSOLETE - DO NOT REUSE**.

## Build

Build the interior:

```sh
quarto render --profile print --to pdf
```

Build the cover from `print/cover/`:

```sh
quarto render The-Complete-Software-Engineering-Lifecycle-Ingram-Cover.qmd --to pdf
quarto render Ingram-Cover-QA-Overlay.qmd --to pdf
```

Copy the production cover to `output/pdf/` after QA. The overlay is temporary and must never be uploaded.

## Design system

The wraparound cover is a clean independent layout, not a resized version of the digital cover. It uses CMYK vector navy, royal blue, cyan, crimson, white, and ice tones; a modular lifecycle graph; and a CMYK author portrait. The lower back cover is intentionally clear for Ingram Studio to place its barcode; do not embed a second barcode or add footer text in that reserve.

The web/OER cover at `assets/images/cover.png` is separate and must not be replaced automatically.
