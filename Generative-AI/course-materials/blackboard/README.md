# Generative AI — Blackboard course package

Sixteen graduate modules follow the fifteen original textbook chapters, ending with an integrated portfolio and demonstration. The Deep Learning Week 1 sample supplied the visual hierarchy and faculty-guidance pattern only. All Word documents are generated from scratch; its course content, deadlines, videos, and placeholder objectives were not imported.

## Materials and authority

Primary: Moody Amakobe, *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*, First Open Edition, Global Data Science Institute, 2026. The authoritative chapter introductions and milestones are in `../../original/` relative to this package. Their selected passages and source hashes are recorded in `qa/source-evidence.json`. Week 16 synthesizes the project arc and Chapter 15 handoff.

Secondary: Chip Huyen, *AI Engineering: Building Applications with Foundation Models*, O’Reilly Media, 2024, ISBN 9781098166298. Publication details and topic mappings were checked against the [publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) and [author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md). There are 14 required weekly selections and two optional selections (Weeks 14 and 16). No invented page numbers or companion quotations are used. Week 14 links the [European Commission AI Act source](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) named in the original reading guidance.

The [course map](GENERATIVE-AI-WEEKLY-COURSE-MAP.md) contains the approved full CO and PSLO statements, all weekly emphases, readings, assignments, and portfolio sequence. The approved mapping is CO 1 → PSLO 1; CO 2 → PSLOs 2, 3; CO 3 → PSLOs 3, 4; CO 4 → PSLO 5; CO 5 → PSLOs 1, 4, 6. These statements are preserved in `qa/curriculum.json` for validation.

## Folder structure

- `source/week-XX.md`: canonical editable guide content.
- `source/week-XX-start-here.md`: written student orientation, 200–350 words.
- `source/week-XX-blackboard-copy.md`: student-facing discussion, applied assignment, milestone/final assessment, rubrics, and Start Here copy.
- `docx/`: 16 weekly Word guides and one combined faculty guide.
- `qa/`: source evidence, document validation, content validation, and visual-review record. Local renders are excluded from Git.
- `BLACKBOARD-GUIDE-BUILD-REPORT.md`: final counts and week-by-week checks.
- `../../scripts/build_blackboard_guides.py`: the single DOCX generator.

## Regenerate and update

From `Generative-AI/`, use Python 3 with `python-docx` installed:

```sh
python3 scripts/lock_original.py
python3 scripts/build_blackboard_guides.py
python3 scripts/lock_original.py
```

The generator reads all 16 canonical guide Markdown files, builds fresh Word packages, reopens them, parses every XML part, and validates table geometry. It does not require the sample DOCX or modify the textbook. Edit the Markdown rather than a generated Word file. Preserve the metadata comment, four-section structure, page-break comments, author passages, approved alignment, and point totals. Synchronize any student-facing assignment or orientation changes with the corresponding Blackboard-copy and Start Here files; those separate files are deliberately human-editable.

The supported Markdown subset is headings, paragraphs, simple bold/italic/code, links, single-level lists, pipe tables, and the `GUIDANCE`/`SUBMISSION` blockquote callouts. Use `<!-- pagebreak -->` for planned pagination. Render the changed guides and combined guide with LibreOffice and inspect every page before distributing regenerated files. The validated edition has eight pages per week and 128 pages combined; pagination can vary with font availability and Word settings.

For repeatable content/render checks, `qa/validate_package.py` uses `python-docx`, `pypdf`, and Pillow after PDFs/PNGs have been rendered into `qa/render/week-XX/` and `qa/render/combined/`. Run it from `Generative-AI/`. It checks source fidelity, mappings, rubric arithmetic, Start Here lengths, and individual/combined page equivalence. Visual review remains required.

## Blackboard setup

1. Create 16 weekly modules with titles from the course map.
2. Create a written Start Here content item using that week’s `start-here.md`. Copy rendered Markdown text into the editor and preserve headings and links; Markdown is not a Blackboard course-import archive.
3. Add the primary chapter link and the focused companion reading from the guide. Keep Weeks 14 and 16 companion readings optional.
4. Create a discussion plus two assignment items using `blackboard-copy.md`. Paste the student-facing prompt and its rubric; remove the literal `[SUBMISSION]` marker if your editor displays it. Faculty guidance is already excluded from these files.
5. Set the displayed due dates and course time zone using the actual syllabus. Schedule day groups are pacing suggestions. Configure Weeks 1–15 as 50/75/50 points and Week 16 as 50/100/100 points, for 2,875 total points. No existing syllabus point policy was found, so this package uses the prompt’s fallback allocation.
6. Preserve the Week 1 personal introduction before the academic discussion and the substantive peer-response requirements. Arrange the Week 16 10–15 minute presentation or slide-and-notes walkthrough; a live prototype is optional and no recording is required.
7. Retain the combined Word guide for faculty reference. Weekly Word guides include pale-yellow faculty guidance, so use the student-copy files for direct assignment publication. If sharing a guide with students, review its faculty notes first.
8. Preview the module as a student, check links and rubric points, and confirm the upload previews. This package has not been uploaded to a Blackboard instance.

## Downloads

[Combined faculty guide — Weeks 1–16](docx/Generative_AI_Weeks01-16_Module_Guides.docx)

| Week | Individual guide | Start Here | Blackboard copy |
| --- | --- | --- | --- |
| 1 | [Week 1 DOCX](docx/Generative_AI_Week01_Module_Guide.docx) | [Written overview](source/week-01-start-here.md) | [Assignment text](source/week-01-blackboard-copy.md) |
| 2 | [Week 2 DOCX](docx/Generative_AI_Week02_Module_Guide.docx) | [Written overview](source/week-02-start-here.md) | [Assignment text](source/week-02-blackboard-copy.md) |
| 3 | [Week 3 DOCX](docx/Generative_AI_Week03_Module_Guide.docx) | [Written overview](source/week-03-start-here.md) | [Assignment text](source/week-03-blackboard-copy.md) |
| 4 | [Week 4 DOCX](docx/Generative_AI_Week04_Module_Guide.docx) | [Written overview](source/week-04-start-here.md) | [Assignment text](source/week-04-blackboard-copy.md) |
| 5 | [Week 5 DOCX](docx/Generative_AI_Week05_Module_Guide.docx) | [Written overview](source/week-05-start-here.md) | [Assignment text](source/week-05-blackboard-copy.md) |
| 6 | [Week 6 DOCX](docx/Generative_AI_Week06_Module_Guide.docx) | [Written overview](source/week-06-start-here.md) | [Assignment text](source/week-06-blackboard-copy.md) |
| 7 | [Week 7 DOCX](docx/Generative_AI_Week07_Module_Guide.docx) | [Written overview](source/week-07-start-here.md) | [Assignment text](source/week-07-blackboard-copy.md) |
| 8 | [Week 8 DOCX](docx/Generative_AI_Week08_Module_Guide.docx) | [Written overview](source/week-08-start-here.md) | [Assignment text](source/week-08-blackboard-copy.md) |
| 9 | [Week 9 DOCX](docx/Generative_AI_Week09_Module_Guide.docx) | [Written overview](source/week-09-start-here.md) | [Assignment text](source/week-09-blackboard-copy.md) |
| 10 | [Week 10 DOCX](docx/Generative_AI_Week10_Module_Guide.docx) | [Written overview](source/week-10-start-here.md) | [Assignment text](source/week-10-blackboard-copy.md) |
| 11 | [Week 11 DOCX](docx/Generative_AI_Week11_Module_Guide.docx) | [Written overview](source/week-11-start-here.md) | [Assignment text](source/week-11-blackboard-copy.md) |
| 12 | [Week 12 DOCX](docx/Generative_AI_Week12_Module_Guide.docx) | [Written overview](source/week-12-start-here.md) | [Assignment text](source/week-12-blackboard-copy.md) |
| 13 | [Week 13 DOCX](docx/Generative_AI_Week13_Module_Guide.docx) | [Written overview](source/week-13-start-here.md) | [Assignment text](source/week-13-blackboard-copy.md) |
| 14 | [Week 14 DOCX](docx/Generative_AI_Week14_Module_Guide.docx) | [Written overview](source/week-14-start-here.md) | [Assignment text](source/week-14-blackboard-copy.md) |
| 15 | [Week 15 DOCX](docx/Generative_AI_Week15_Module_Guide.docx) | [Written overview](source/week-15-start-here.md) | [Assignment text](source/week-15-blackboard-copy.md) |
| 16 | [Week 16 DOCX](docx/Generative_AI_Week16_Module_Guide.docx) | [Written overview](source/week-16-start-here.md) | [Assignment text](source/week-16-blackboard-copy.md) |
