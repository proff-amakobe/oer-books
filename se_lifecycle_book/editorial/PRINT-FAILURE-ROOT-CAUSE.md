# Print Failure Root Cause

## Universal Terminal Conversion

At `3b09d8fcf478d063fe874187e80196030a8da61a`, `print/filters/print-components.lua` added `function CodeBlock(block)`. For every LaTeX render it constructs and returns one raw block: `\\begin{SETerminal}...block.text...\\end{SETerminal}`. There is no semantic branch that limits this result to shell sessions. The language lookup changes only an option; JavaScript, JSON, YAML, CSS, Dockerfile, Terraform, and untyped blocks still use the same environment.

## Semantic Loss

The conversion collapses program code, configuration, structured data, terminal commands/sessions, output, pseudocode, and ASCII diagrams into one presentation type. The later Unicode replacement table also changes arrows, checks, crosses, and box drawing inside every CodeBlock, regardless of meaning.

## Complete Edition Impact

Yes. The universal handler was added to the complete print profile four days before the volume profiles existed, so the complete edition inherited the defect before the split. The legacy complete profile still loads that handler.

## Volume I Missing Sections

The split selector is not the cause. `_quarto-volume1.yml` explicitly lists Chapters 1–8. `volume-content.lua` clears only the shared `index.qmd`, and `volume-crossrefs.lua` walks inlines/paragraphs without deleting sections. Comparing the current Volume I PDF against all raw source H1–H4 strings found no deleted section content: one heading is intentionally rewritten from “Your Semester Project” to “Your Project,” and “Requirements in Practice: Tools and Techniques” is present with extraction hyphenation (“Tech-” / “niques”).

The evidence supports a presentation failure: large source regions that Pandoc regards as CodeBlocks—including untyped diagrams and some manuscript-like example material—are converted to large, dark, non-semantic terminal panels. This hides heading hierarchy and makes sections appear absent. In addition, the objectives transform deliberately removes each “Learning Objectives” H2 from the body and substitutes an environment. The evidence does **not** support chapter deletion by the volume split.

## Volume II Terminal Problem

The exact cause is the unconditional CodeBlock return described above plus the large dark `SETerminal` definition in `print/preamble.tex`. Chapters 9–15 have dense mixtures of YAML, SQL, JSON, JavaScript, infrastructure configuration, output, and ASCII diagrams; all therefore receive terminal chrome.

## QA Failure

Earlier checks measured build completion, pagination/overflow, page counts, image overflow, and extracted-text presence. Those checks could succeed while the wrong environment rendered a block. Presence-only extraction also cannot distinguish JavaScript from a terminal window or detect lost heading hierarchy when heading-like text survives inside a CodeBlock.

## Safe Components

The canonical chapter lists, standard Quarto cross-references, ordinary Pandoc syntax highlighting, image assets, bibliography, `scrbook`, LuaLaTeX, letter trim, and narrow `volume-content.lua` index removal are potentially reusable after review. Any reuse of volume wording substitutions requires explicit editorial approval.

## Unsafe Components

The universal `CodeBlock` conversion, global Unicode replacement in technical blocks, custom objective transformation, dynamic TOC-depth injection, exact-title front-matter truncation, and `SETerminal`-driven pagination must not form the reconstruction baseline.
