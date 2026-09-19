# Cover Review

## Immutable author source

`original/images/cover.png` is **724 × 1005 pixels**, aspect ratio **0.7204**. Despite its suffix, its bytes encode a JPEG. All text is baked into the raster image. It has a navy background with teal/blue light, a central luminous brain, and connected feature icons. The former title is “INTRODUCTION TO GENERATIVE AI WITH LARGE LANGUAGE MODELS”; the subtitle describes building intelligent applications with prompting, RAG, and multimodal tools. The author appears at the bottom. There is no publisher mark.

The former text cannot be independently typeset, and the brain/icon composition does not match this pass's restrained technical-book direction. The original was inspected and preserved byte-for-byte, without painting over, recompressing, or renaming it.

## Derived publication design

The publication cover preserves the navy/teal palette and connected-system theme. Clean typography and abstract architectural planes replace the old title, subtitle, and brain imagery. Title, subtitle, author, and publisher use the exact current identity. Critical text sits comfortably inside the edges; the artwork may extend to the edges. No ISBN, legal paragraph, feature list, spine, barcode, or back cover is present.

A built-in image-generation concept guided the composition; its raster size was below the requested target. The final scalable SVG was composed independently with geometric paths and system-compatible typography, then rendered natively to an opaque **2550 × 3300 PNG**, aspect ratio **0.772727**, exactly 8.5:11. No raster upscaling or proprietary font files were used. See `COVER-GENERATION-PROMPT.md` for the prompt and reproduction command.

The final PNG is the reviewed artifact used by HTML, OpenGraph/Twitter/Book metadata, EPUB, and PDF. The SVG is an editable companion; it is not an Ingram production cover. Full-size, half-size, 300 px, and 150 px checks assess cropping, text, contrast, and image quality. The title is immediately readable at 150 px; the subtitle is readable at the normal 300 px card size. White and pale-blue text contrast with the navy background. The PNG is opaque, so web color modes do not alter its internal contrast.

## Author facts and photo

Biography facts come solely from `advanced-algorithms-book/about-author.qmd`: multidisciplinary research, data science, blockchain architecture, AI/computational/data/distributed-system expertise, GDSI founder, teaching, research advising, graduate supervision, textbook authorship, and industry solutions in the named sectors. No degree, university name, award, employer, or location was added.

An existing photo is present at `se_lifecycle_book/assets/author/moody-amakobe.jpg`, with a print derivative referenced by `se_lifecycle_book/print/cover/ingram-cover-template.tex` and `print/covers/source/series-cover-common.tex`. It is reported here but not added to this book. No portrait was generated.
