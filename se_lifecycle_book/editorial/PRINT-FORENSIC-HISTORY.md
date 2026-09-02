# Print Forensic History

Status: **LEGACY / UNVALIDATED PRINT PIPELINE**

Baseline main SHA: `2520b018c47fe36a3e2cd6a300f512187af7ff03`

## Timeline

- `ceecf3ad1c65ff869f62e7f698d7510c13bfd1af` (2026-08-24), *Redesign Software Engineering professional print edition*, split digital and print profiles and introduced `_quarto-print.yml`, `print/frontmatter.qmd`, `print/preamble.tex`, and `print/filters/print-components.lua`. Its first filter removed document metadata, discarded all combined-book blocks before the exact Chapter 1 H1, replaced that material with custom front matter, and replaced the Learning Objectives heading with `SEObjectives` while retaining the objective content. It did not yet define a `CodeBlock` handler.
- `3b09d8fcf478d063fe874187e80196030a8da61a` (2026-08-24), *Optimize Software Engineering print pagination*, introduced the universal `CodeBlock` handler. Every LaTeX-target CodeBlock became raw LaTeX `SETerminal`, with only an optional language argument varying. The same commit introduced per-section TOC-depth injection, expanded the `SETerminal` styling/pagination rules, and adjusted custom front matter.
- `528c82c121dbf0428dc5029ff7ad5c1f06f03fe0` (2026-08-28), *Finalize standalone Software Engineering volumes*, introduced `_quarto-volume1.yml`, `_quarto-volume2.yml`, `filters/volume-content.lua`, and `filters/volume-crossrefs.lua`. `volume-content.lua` clears only `index.qmd` in a volume build. `volume-crossrefs.lua` rewrites volume-relative wording and chapter-number references; for Volume II it also substitutes two matched paragraphs. It does not delete chapter sections. This commit extended `print-components.lua` with volume front-matter selection, image IDs/widths, and Unicode-to-ASCII substitutions inside all CodeBlocks.
- `01dfd3ae8712e23219a134c1d6c9c4fdd5cb9165` and `2520b018c47fe36a3e2cd6a300f512187af7ff03` changed Volume II bleed/non-bleed production configuration, not the semantic CodeBlock architecture.

## Capability and risk map

| Component | Introduced | Behavior | Reconstruction disposition |
|---|---|---|---|
| `print-components.lua` | `ceecf3a`; CodeBlock conversion at `3b09d8f` | front matter, objectives, TOC, figures, universal Terminal | quarantined; not loaded |
| `print/preamble.tex` | `ceecf3a` | branded layout and `SETerminal` definition | quarantined; not loaded |
| custom front matter | `ceecf3a` | replaces all blocks before exact Chapter 1 title | quarantined |
| Learning Objectives treatment | `ceecf3a` | removes H2 and wraps following blocks | quarantined |
| print figure sizing | `528c82c` | assigns widths/identifiers by filename lists | quarantined pending visual review |
| TOC manipulation | `3b09d8f` | repeatedly injects `tocdepth` based on first seven H2s | quarantined |
| chapter opener system | preamble evolution through `3b09d8f` | custom chapter appearance/page behavior | quarantined |
| `volume-content.lua` | `528c82c` | removes volume landing input only | legacy; narrow behavior verified |
| `volume-crossrefs.lua` | `528c82c` | semantic wording/chapter-reference rewriting | legacy; not used in reconstruction |

No history was rewritten and no legacy production file was deleted or reverted.
