# Publication cover design provenance

The built-in image-generation tool produced a visual concept using the immutable author cover as a palette/mood reference. The concept was 1103 × 1426 px, below the preferred master resolution. The final publication asset is an independently composed, editable SVG using clean system-compatible typography and geometric paths, rendered natively at 2550 × 3300 with librsvg. No raster enlargement or original-source editing was used. No font files are committed.

## Concept prompt

Create a finished professional academic technical textbook FRONT COVER, flat artwork without mockup or perspective. Reference image is ONLY a palette/mood reference; do not edit or overwrite source. Preserve deep navy, muted teal and restrained blue visual identity. Replace all former typography and brain artwork. Portrait US Letter ratio 8.5:11, requested 2550x3300 pixels or higher at that ratio. Clean high contrast Swiss-inspired typography, wide safe margins at least 7% on all sides. Title very large and strongest element, upper left, two lines: "GENERATIVE" then "AI". Subtitle below in clear readable subordinate type, EXACT text: "Foundations, Systems, Evaluation," then "and Responsible Deployment". Bottom author EXACT: "Moody Amakobe". Publisher beneath discretely but legibly EXACT: "Global Data Science Institute". No other text. Middle visual: elegant sparse abstract layered architectural planes and a few precisely connected nodes suggesting information moving through an engineered system, muted teal/blue, sophisticated flat geometric editorial art, ample negative space. No brain, robots, humanoids, faces, neon, feature lists, icons with labels, noisy texture, stock sci-fi, old title or subtitle, ISBN, logos, spine or back cover. Opaque full background, crisp typesetting, professional contemporary university textbook. Subtitle must be large enough to read on a normal web cover card. Produce the actual cover only.

## Final assets

- `assets/cover/generative-ai-cover.svg`: scalable composition, Arial/Helvetica/sans-serif font stack.
- `assets/cover/generative-ai-cover.png`: opaque 2550 × 3300 PNG used in all editions.

Re-render with `rsvg-convert -w 2550 -h 3300 assets/cover/generative-ai-cover.svg -o assets/cover/generative-ai-cover.png`. Font substitution can affect SVG text when re-rendering on another platform; the committed PNG preserves the reviewed typography.
