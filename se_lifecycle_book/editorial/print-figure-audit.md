# Print Figure Audit

## Scope

All image references in the 15 chapters and glossary were checked against `assets/images/`. The print build uses the original SVG sources and lets Quarto create vector PDF intermediates.

## Findings

- The canonical cover is `assets/images/cover.png`.
- All referenced local figure files exist.
- Chapter 7 and Chapter 8 Obsidian-only embeds were previously converted to portable Markdown image references.
- The Chapter 7 branching and Chapter 8 testing diagrams are original SVG assets and are retained as vectors for print.
- Several assets are present but not referenced by the current manuscript. They were retained because their editorial intent cannot be determined safely.
- No missing figure was fabricated.

## Items requiring later editorial review

- Many images are plain Markdown figures without explicit Quarto figure identifiers or captions, so they do not populate a complete automatic List of Figures.
- The manuscript contains descriptive prose around some diagrams but relatively few formal figure captions.
- The reference bibliography remains a placeholder and was not reconstructed.
