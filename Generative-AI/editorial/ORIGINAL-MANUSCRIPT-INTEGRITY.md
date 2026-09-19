# Original manuscript authorial integrity

The files under Generative-AI/original/ are the immutable author-supplied manuscript and source assets.

| Measure | Result |
|---|---:|
| Original chapter QMD files | 15 |
| Original images | 32 |
| Other manuscript files (`PROJECT_ARC.md`) | 1 |
| Incidental `.DS_Store` files, locked but excluded from Git | 2 |
| Total original files with raw-byte SHA-256 locks | 50 |
| Original bytes modified | 0 |
| Original files with changed hashes | 0 |
| Missing original files | 0 |
| Unexpected files remaining in original/ | 0 |

Manuscript prose rewritten: **NO**. Headings rewritten: **NO**. Technical examples rewritten: **NO**. Tables rewritten: **NO**. Citations rewritten: **NO**. Images replaced: **NO**.

The complete file set and raw bytes were checked before publication work and after the HTML, PDF, and EPUB builds. `scripts/lock_original.py` verifies the manifest. Git newline conversion is disabled for `original/**`; staged blobs are independently compared with the author lock before committing. The local lock covers all 50 files. Git and CI omit only the two incidental `.DS_Store` files, so CI checks 48 immutable publication files.

The original 88 technical blocks retain their classes and textual payloads, including untyped blocks, code, prompts, and Unicode symbols. `ORIGINAL-TECHNICAL-PAYLOADS.csv` records first and last meaningful lines and hashes with only newline normalization and final newline removal. HTML/EPUB comparison preserves indentation and spaces. PDF comparison ignores line wrapping and the renderer’s continuation marker, without changing source text.

All 20 referenced images use supplied source assets. The resource aliases are byte-identical; two misleading `.png` filenames contain JPEG bytes and receive additional `.jpg` aliases outside original/. The Marcus raw-LaTeX image receives a format-specific rendering bridge so it survives HTML and EPUB. Its image bytes and the original LaTeX remain unchanged.

The previous rewritten manuscript remains in `chapters/`, explicitly marked NOT ACTIVE MANUSCRIPT and excluded from all active profiles. It was not merged into the originals. All review observations are advisory and unapplied.
