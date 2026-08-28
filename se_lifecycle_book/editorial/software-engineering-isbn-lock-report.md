# Software Engineering ISBN Lock Report

## Volume I

| Field | Result |
|---|---|
| Title | Software Engineering Foundations and Design |
| Print ISBN | 979-8-2408-9097-0 |
| Ebook ISBN | TBD |
| Page Count Before | 450 |
| Page Count After | 450 |
| ISBN visible | PASS — title and copyright pages |
| PDF metadata | PASS |

## Volume II

| Field | Result |
|---|---|
| Title | Software Delivery, Operations, and Evolution |
| Print ISBN | 979-8-2408-9370-4 |
| Ebook ISBN | TBD |
| Page Count Before | 409 |
| Page Count After | 409 |
| ISBN visible | PASS — title and copyright pages |
| PDF metadata | PASS |

## ISBN Validation

| ISBN digits | Result |
|---|---|
| 9798240890970 | PASS — calculated check digit 0 |
| 9798240893704 | PASS — calculated check digit 4 |

## ISBN Isolation

| Edition | Result |
|---|---|
| Complete edition | PASS — retains 979-8-2957-6070-9 print and 979-8-2957-6071-6 ebook ISBNs |
| Volume I | PASS — contains only its assigned standalone print ISBN |
| Volume II | PASS — contains only its assigned standalone print ISBN |
| EPUB print-ISBN leakage | PASS — neither EPUB contains a print ISBN; each OPF uses a non-ISBN UUID identifier |

## Print QA

| Check | Result |
|---|---|
| Trim | PASS — 612 × 792 points |
| Fonts | PASS — embedded |
| Encryption | PASS — none |
| Figures | PASS — zero physical and text-area overflows |
| Cross References | PASS |
| Pagination Lock | PASS — 450 and 409 pages unchanged |

## Ingram Metadata

- `editorial/INGRAM-METADATA-VOLUMES.md`: **CREATED**
- Volume I template: **PENDING**
- Volume II template: **PENDING**
- Spine geometry: **DO NOT CALCULATE — USE OFFICIAL INGRAM TEMPLATE**

## Next Steps

1. Enter final metadata in IngramSpark.
2. Generate the official Volume I cover template.
3. Generate the official Volume II cover template.
4. Save both original Ingram template files in the repository.
5. Do not alter template geometry.
6. Design coordinated wraparound covers directly against the official templates.
7. Run cover/interior consistency preflight.
8. Determine final retail pricing and compensation.
9. Upload interiors and covers to IngramSpark.
