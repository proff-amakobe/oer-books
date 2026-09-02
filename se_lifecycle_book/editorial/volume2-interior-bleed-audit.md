# Volume II Interior Bleed Audit

Title: Software Delivery, Operations, and Evolution

ISBN: 979-8-2408-9370-4

Source pages: 409

Trim: 8.5 x 11 in (612 x 792 pt)

Bleed canvas: 8.625 x 11.25 in (621 x 810 pt)

## Element Audit

| Page type | Current behavior | Required extension | Status |
|---|---|---|---|
| Chapter openers (PDF pages 14, 87, 164, 228, 287, 340, 373) | Navy/deep-blue vector field and periwinkle rule reached top and both horizontal trim edges | Extend 0.125 in through top and mirrored outside bleed; no gutter canvas | PASS |
| Part openers (PDF pages 13, 163, 286) | White page with centered typography; no artwork touches trim | None; white bleed is intentional | PASS |
| Title page (PDF page 3) | Navy vector field and blue rule reached top and both horizontal trim edges | Extend 0.125 in through top and outside bleed; no gutter canvas | PASS |
| Other front matter | Typography and rules remain inside trim-safe area; no edge-touching art | None | PASS |
| Normal prose | Headers, footers, body, tables, and callouts remain inside locked trim-relative margins | None | PASS |
| Figures | Content-sized vector/raster figures remain inside the text area | None | PASS |
| Code/terminal pages | Boxes remain inside the locked text area | None | PASS |
| Closing page | No artwork intended to print edge-to-edge | None | PASS |

## Mirrored Geometry

| Physical parity | TrimBox | Bleed edges | Gutter |
|---|---|---|---|
| Odd / recto | `[0 9 612 801]` | top, bottom, right | left; 0 pt |
| Even / verso | `[9 9 621 801]` | top, bottom, left | right; 0 pt |

The 409 source pages are placed without scaling. The title-page field and seven vector chapter-opener fields that touch trim are continued through the top and outside bleed. Normal pages retain intentional white bleed areas.

## QA Evidence

- Automated geometry audit: PASS (all 409 pages).
- Prior locked non-bleed MD5: `5cef6d1c74ccc0b188ab6769d55ccd8f`.
- Final production/bleed-proof MD5: `0f6ca8760dcf8f9860ec0b434376a11c`.
- Independent `pdfseparate` page count: PASS (409).
- All-page raster render: PASS (409 rendered pages).
- White-sliver test: PASS on the title field and all seven chapter openers.
- Odd/even outside-edge inspection: PASS.
- Fonts: PASS; all listed fonts embedded.
- Content text comparison: PASS; extracted-text checksums are identical.
- Navigation regression: PASS; 387 bookmarks and 73 annotations preserved.
- Strict parser preflight: PASS. (`qpdf` installation was blocked by the host's outdated Xcode/Command Line Tools.)
- Representative renders: `print/qa/volume2-bleed/`.
