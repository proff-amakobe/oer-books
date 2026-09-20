# Blackboard Guide Build Report

## Package results

| Check | Result |
| --- | --- |
| Weeks built | 16/16 |
| Individual DOCX valid | 16/16 |
| Combined DOCX | PASS — 128 pages; each week starts on a new page |
| Start Here pages | 16/16 — 254–267 words including headings |
| Blackboard copy files | 16/16 |
| Primary chapters used | 15/15 |
| Author introductions used | 15/15, selected original opening passages preserved verbatim |
| Original project milestones | 15/15, all requirements preserved |
| Week 16 synthesis | PASS — no new chapter |
| Secondary text mappings | 16 — 14 required, 2 optional |
| Discussion prompts | 16/16 |
| Applied assignments | 16/16 — 15 analyses plus final integrated portfolio |
| Project/final assessments | 16/16 — 15 milestones plus final demonstration |
| Rubrics | 48 |
| Rubric point arithmetic | PASS — 50 points × 16 = 800; separate residency 200; course total 1,000 |
| CO alignment | PASS |
| PSLO alignment | PASS |
| Week 1 author introduction | PASS — original welcome and generative-AI problem question |
| Week 1 student introduction | PASS — 100–150 words before 350–500-word academic response |
| Final portfolio | PASS — all 15 artifacts revised and integrated |
| Final demonstration | PASS — 10–15 minutes; live prototype optional |
| Videos required | 0 |
| DOCX ZIP/XML/reopen/table geometry | PASS — 17/17 files |
| Visual QA | PASS — all 128 individual pages inspected; 128 combined pages pixel-identical |
| Original manuscript modified | 0 |
| Original hash integrity | PASS — 50 files checked before and after generation |

## Week-by-week QA

For Week 16, the author-source column verifies synthesis from the established project arc and Chapter 15 handoff, rather than a nonexistent sixteenth chapter. Secondary-reading PASS includes the explicit optional status of Weeks 14 and 16.

| Week | Chapter | Introduction from author source | CO mapping | PSLO mapping | Primary reading | Secondary reading | Discussion | Applied assignment | Milestone/final | Rubric sum | DOCX validation | Visual QA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 2 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 3 | 3 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 4 | 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 5 | 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 6 | 6 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 7 | 7 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 8 | 8 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 9 | 9 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 10 | 10 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 11 | 11 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 12 | 12 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 13 | 13 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 14 | 14 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 15 | 15 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | Synthesis | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Evidence and scope

`qa/docx-validation.json` records package-level validation for every Word file. `qa/content-validation.json` records weekly word counts, points, source fidelity, alignment, page counts, and all 128 page comparisons. `qa/visual-review.json` records the completed visual review and final DOCX hashes. `qa/source-evidence.json` binds the original introductions and milestones to chapter hashes. A final chapter-end transition is excluded from the milestone excerpt; no milestone requirement is removed.

The author’s case studies and source readings anchor the applied assignments. Companion selections were checked against O’Reilly’s listing and the author’s table of contents, linked in the course map/README. No companion page ranges were invented. Historical or illustrative claims in the original narrative remain the author’s text; prompts ask students to distinguish them from current independent evidence.

All weekly guides use Letter pages, 0.75-inch margins, Arial 10.5-point body text, 9.5-point table text, navy table headers, blue headings, yellow faculty notes, and green submission instructions. The sample served as a design reference, not as a binary template or instructional authority. Lists are real Word lists; tables use fixed column geometry and repeating headers. Render review found no clipping, overlap, broken tables, or orphaned headings.

Only this Blackboard package and `scripts/build_blackboard_guides.py` are included in the change. No original chapter, Quarto manuscript, homepage, Preface, cover, or chapter asset is changed. The materials are ready for faculty upload/copy; no Blackboard account was modified.

## Revised grading allocation

The author approved 800 points distributed equally across the sixteen weeks and 200 reserved for a separate residency assessment. Weeks 1–15 use 10 discussion / 25 applied / 15 milestone points. Week 16 uses 10 reflection / 20 portfolio / 20 demonstration points. All 48 rubric totals and weekly summary tables were reweighted. Residency has a distinct gradebook allocation; no residency activity or grading criteria were invented.
