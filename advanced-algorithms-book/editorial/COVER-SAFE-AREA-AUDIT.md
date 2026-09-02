# Ingram Cover Safe-Area Audit

Audit date: 2026-09-01

Automated source: `editorial/qa/cover/cover-object-bounds.json`

Build/audit script: `scripts/print/build_ingram_cover.py`

The official template was raster-measured to identify the three pink safe rectangles. Coordinates below are PDF points from the lower-left corner. A critical object passes only when it is fully inside its panel safe rectangle with at least 7.2 pt (0.10 in) clearance. The template checksum was verified before the cover was generated.

| Panel | Official pink safe rectangle (x0, y0, x1, y1), pt |
|---|---|
| Back | 199.0, 72.0, 793.0, 846.0 |
| Spine | 806.5, 72.0, 886.5, 846.0 |
| Front | 900.0, 72.0, 1494.0, 846.0 |

| Element | Panel | Actual bbox, pt | Nearest safe boundary clearance | Status |
|---|---|---|---:|---|
| Front title | Front | 936.0, 624.0, 1295.851, 774.0 | 36.0 pt / 0.500 in | PASS |
| Subtitle | Front | 938.0, 578.0, 1268.720, 596.0 | 38.0 pt / 0.528 in | PASS |
| Edition statement | Front | 938.0, 537.0, 1067.177, 554.0 | 38.0 pt / 0.528 in | PASS |
| Author | Front | 938.0, 470.0, 1122.443, 491.0 | 38.0 pt / 0.528 in | PASS |
| Publisher | Front | 938.0, 102.0, 1121.166, 115.0 | 30.0 pt / 0.417 in | PASS |
| Back headline | Back | 235.0, 758.0, 705.0, 778.0 | 36.0 pt / 0.500 in | PASS |
| Back description | Back | 235.0, 565.0, 735.0, 748.0 | 36.0 pt / 0.500 in | PASS |
| Author-bio heading | Back | 235.0, 343.0, 585.0, 357.0 | 36.0 pt / 0.500 in | PASS |
| Author biography | Back | 235.0, 267.2, 625.0, 338.0 | 36.0 pt / 0.500 in | PASS |
| OER/license/publisher | Back | 235.0, 128.8, 585.0, 176.0 | 36.0 pt / 0.500 in | PASS |
| Official Ingram barcode | Back | 653.0, 86.0, 743.5, 157.5 | 14.0 pt / 0.194 in | PASS |
| Spine title and author | Spine | 847.0, 160.0, 863.0, 587.743 | 23.5 pt / 0.326 in | PASS |
| Spine publisher mark | Spine | 843.0, 106.0, 857.0, 120.0 | 29.5 pt / 0.410 in | PASS |

Minimum critical-object clearance is **14.0 pt / 0.194 in**, at the official barcode. All 13 audited critical objects pass. The rendered safe-area proof and template overlay were separately inspected; no critical object touches a safe line or fold.

**SAFE-AREA RESULT: PASS**
