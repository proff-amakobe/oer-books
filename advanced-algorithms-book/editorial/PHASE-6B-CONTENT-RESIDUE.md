# Phase 6B — Textbook-versus-Course Residue

## Publication-breaking residue resolved

- Removed ten standalone `python` markers that were printed as prose before technical blocks.
- Removed the captured editor/UI fragment `Retry / M / Continue / Edit` and rejoined the interrupted Python program.
- Replaced due-date, submission-method, late-policy, office-hours, and discussion-forum placeholders with a concise Instructor Adaptation Note.
- Replaced the sample project's `[Your Name] - [Your Email]` placeholder and generalized its course/week milestone language.
- Removed the duplicate short “How to Use This Book” section from the Preface; the dedicated front-matter chapter remains.
- Removed the remote 72 ppi license badge while preserving the CC BY 4.0 name and canonical license URL.
- Corrected visible and embedded publication identity to Second Edition / 2026 / ISBN TBD. The First Edition ISBN is no longer presented as current metadata.

## Remaining course-oriented language for later editorial review

The focused residue scan still finds about 30 contextual uses of terms such as `this course`, `Week 1`, `Week 2`, `homework`, and `submission guidelines`. These are concentrated in Chapters 1 and 15. They are not administrative placeholders and do not break pagination, so Phase 6B leaves them for a dedicated content-editing pass rather than broadening a print rescue into another manuscript rewrite.

Representative groups:

- **Chapter 1:** course framing, environment setup narrative, “Week 1” demonstration strings, the Week 2 preview, homework preview, and submission-guideline pedagogy.
- **Chapter 15:** reflection prompts and career-planning language that refer to “this course.”
- **Legitimate analogy:** Chapter 7 uses grading homework as a plain-language complexity analogy.
- **Exercise/template placeholders:** Chapter 11 contains explicitly labeled implementation placeholders; Chapters 14–15 contain project-template tokens such as repository URLs and reviewer/name fields. These remain because the surrounding text clearly identifies them as fields learners must adapt.

## Markdown fence review

PDF searches find literal `````python`` and `````bash`` sequences only inside examples that teach or generate Markdown/README content. They are visible source data inside configuration examples, not leaked Quarto fences. No opening or closing fence from the surrounding manuscript appears accidentally in ordinary prose.

## Recommended next content pass

After print acceptance, review Chapters 1 and 15 for edition-durable textbook phrasing. Preserve useful instructor-adaptation and project-template material, but separate it visually from the core exposition and replace schedule-dependent “week” language with chapter or milestone references where pedagogically equivalent.
