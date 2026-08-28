# Volume Independence Audit

## Volume I — PASS

- Dedicated title, copyright, series note, preface, and usage guidance.
- Begins naturally with Introduction to Software Engineering as Chapter 1.
- Chapters run locally from 1 through 8.
- A concise closing note gives the volume a deliberate conclusion after Testing and Quality Assurance.
- Cross-volume glossary references explicitly identify Volume II.
- Core use does not require Volume II.

## Volume II — PASS

- Dedicated preface explains the transition from verified software to integration, delivery, operation, security, maintenance, and evolution.
- Begins naturally with CI/CD as local Chapter 1.
- Chapters run locally from 1 through 7; sections, TOC entries, and chapter openers follow local numbering.
- Cross-volume glossary references explicitly identify Volume I.
- Render-time synthesis language does not claim that Volume I topics were taught earlier in this standalone book.
- Final project guidance can be applied to an existing design or codebase.

## Shared Strategy

Both books include the one canonical glossary. A volume-aware filter converts its chapter annotations at build time, avoiding two manually maintained glossary copies. Bibliographic metadata remains connected to the canonical `references.bib`; the current manuscripts do not support a reliable volume-specific bibliography split, and no sources were fabricated.
