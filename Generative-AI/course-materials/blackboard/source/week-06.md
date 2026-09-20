<!-- metadata: {"week": 6, "title": "Retrieval-Augmented Generation", "chapter": "Chapter 6: Retrieval-Augmented Generation", "milestone": "RAG Design Document", "co": [2, 3, 5], "pslo": [1, 2, 3, 4, 6], "points": [10, 25, 15], "case": "How Glean Built Enterprise Search with RAG"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 6 Module Guide

## Retrieval-Augmented Generation

**Chapter:** Chapter 6: Retrieval-Augmented Generation

**Project:** RAG Design Document

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 6 Introduction & Objectives

#### Module Introduction

Large Language Models possess impressive knowledge from their training data, but they face fundamental limits: a knowledge cutoff, no access to private or recent information, and a persistent tendency toward confident fabrication when they don't actually know something, the hallucination problem from Chapter 1. Retrieval-Augmented Generation, RAG, addresses these limits not by making the model smarter, but by giving it something to actually look at before it answers.

RAG represents a shift from closed-book AI, answer from memory alone, to open-book AI, retrieve relevant information first, then answer grounded in what was found. This is essential for working with proprietary documents, staying current with information that changes faster than any training run, and producing responses a user can actually verify against a real source.

In this chapter, you'll design the knowledge layer for your research assistant: how it would ingest documents, build a searchable knowledge base, and ground its responses in specific, retrievable sources, the architecture behind everything from customer support systems to legal research tools.

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Creation | Design a document-processing and chunking strategy for a specific knowledge source. | CO 2 / PSLOs 2, 3 |
| 2 | Analysis | Compare semantic, keyword, and hybrid retrieval against domain requirements. | CO 2 / PSLOs 2, 3 |
| 3 | Evaluation | Evaluate retrieval and generation failures separately. | CO 3 / PSLOs 3, 4 |
| 4 | Evaluation | Critique research evidence for a chosen retrieval design. | CO 5 / PSLOs 1, 4, 6 |
| 5 | Creation | Develop a worked retrieval-to-cited-answer example with explicit source boundaries. | CO 2 / PSLOs 2, 3 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 2, CO 3, CO 5 / PSLOs 1, 2, 3, 4, 6**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 2:** Design and implement applications that leverage Large Language Models and generative AI techniques to solve complex computational problems.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

**CO 5:** Investigate and synthesize advanced methodologies, architectures, emerging technologies, and current research trends in Large Language Models and generative artificial intelligence.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 6 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 6 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 6 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop RAG Design Document. | 100–140 min |
| Sunday | Check the rubric, reconcile the milestone with earlier decisions, and submit the major work through the relevant Blackboard items. | 40–70 min |

The ranges total approximately 300–420 minutes (5–7 hours). Reading, discussion, applied work, and project writing are included. Use focused sections and reuse evidence across connected tasks rather than starting separate projects.

#### Reminders

- Consult the due dates shown in Blackboard; the schedule above is not a deadline policy.
- Keep one research-assistant domain and maintain a revision history for the same portfolio.
- Cite sources for claims. Label measured results, scenario assumptions, and projected behavior distinctly.
- Use public, synthetic, or appropriately authorized data. Keep credentials and private personal information out of submitted examples.
- Retain feedback and note one change to carry into the next milestone.


<!-- pagebreak -->

## SECTION 3 — READINGS & RESOURCES

### 3. Readings and Resources

> [GUIDANCE] Course Author Guidance — Readings: Assign the primary chapter first. Keep the companion selection focused on the named topics. Optional sources extend a question, not the mandatory workload. Retain the author’s case and milestone as the anchors for assessment.

#### Required Reading 1 — Primary Textbook

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 6: Retrieval-Augmented Generation**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/06-rag.html).

Read the introduction and learning goals, the technical discussion, **How Glean Built Enterprise Search with RAG**, the wrap-up, and **Project Milestone: RAG Design Document**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 6: RAG and Agents — RAG selections only**

Focus: RAG Architecture, Retrieval Algorithms, Retrieval Optimization, and RAG Beyond Texts.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 6 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — The paper that introduced the RAG approach this chapter is built around.

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 6 Assignments

The discussion develops a judgment about **When Does RAG Increase Trust — and When Doesn't It**. The applied work tests that judgment against How Glean Built Enterprise Search with RAG. The milestone carries the evidence into RAG Design Document.

### Assignment 1 — When Does RAG Increase Trust — and When Doesn't It?

> [SUBMISSION] Type: Discussion Board. Points: 10. Submit one initial post and two peer responses in the Week 6 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Choose a question your model cannot answer reliably from training alone. Describe what evidence would make an answer supportable.

**Part B — Comparison:** Compare semantic, keyword, or hybrid retrieval for that question. Explain a case where the most relevant document must still be excluded.

**Part C — Judgment:** Decide when to answer, qualify, or abstain after retrieval. Name separate retrieval and generation checks.

#### Peer Response Guidance

Reply to **two classmates, 150–200 words each**. Challenge an assumption, introduce a counterexample, identify an overlooked trade-off, ask a substantive question, or connect the design to another course concept. Agreement, praise, or summary alone is insufficient. Explain why your contribution matters to the peer’s decision.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Conceptual application | Applies the week’s concepts accurately to the concrete problem and compares alternatives. | 3 |
| Evidence and judgment | Uses chapter evidence to defend a position, identifies a limitation, and explains a trade-off. | 3 |
| Project connection | Connects the argument to the same research assistant and its users. | 2 |
| Peer engagement | Two substantive replies meet the length guidance and advance the reasoning. | 2 |
| Total |  | 10 |

Cite the chapter or companion selection when it supports your reasoning. Cite outside claims to identifiable sources; an unsupported assertion does not become evidence because a model generated it.


<!-- pagebreak -->

### Assignment 2 — Retrieval Failure Analysis / Glean Case

> [SUBMISSION] Type: Applied analysis. Points: 25. Submit one PDF or DOCX to Week 6 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **How Glean Built Enterprise Search with RAG** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Use the Glean case to sketch permission-aware enterprise retrieval. Make clear where access checks occur before generation.

2. Create three small document excerpts and two user roles with different permissions. Walk through one query where the most similar passage is not authorized for the requester.

3. Compare a retrieval failure with a generation failure using the same query. Propose a chunking or retrieval repair and a separate response-grounding repair.

4. Design a two-part evaluation: retrieval relevance/access correctness and answer support by cited evidence. Explain what a fluent but unsupported answer should score.

#### Evidence and Submission Guidance

Use the same assumptions consistently across the response. A design exercise can be completed as a documented walkthrough; do not disguise a projected outcome as an experiment. Where a task asks for live observations, use accessible tools and retain the relevant outputs. If access prevents the required comparison, request an instructor-provided output set rather than inventing results. Explain what your evidence cannot establish.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Case and problem framing | Accurately identifies the case’s design tension and separates given facts from assumptions. | 5 |
| Analysis and comparison | Completes the requested comparison or experiment with a defensible method and evidence. | 8 |
| Limits and trade-offs | Evaluates failure modes, alternative explanations, and relevant constraints. | 5 |
| Recommendation and transfer | Connects a justified recommendation to the research-assistant design. | 5 |
| Traceability and clarity | Includes requested artifacts, source references, and clear labels for measured/projected results. | 2 |
| Total |  | 25 |


<!-- pagebreak -->

### Assignment 3 — Research Assistant Milestone 6: RAG Design Document

> [SUBMISSION] Type: Project portfolio milestone. Points: 15. Submit a PDF or DOCX to Week 6 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add a **RAG Design Document** to your project portfolio:

1. **Knowledge source**: The specific documents or data your assistant would retrieve from.
2. **Chunking strategy**: Your approach and reasoning, per section 6.3.
3. **Retrieval approach**: Semantic, hybrid, or otherwise, justified for your domain.
4. **Worked example**: One realistic query, walked through end to end, from retrieval through a grounded, cited response.

#### What Makes a Strong Milestone

A strong **RAG Design Document** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Knowledge source | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 4 |
| Chunking strategy | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 3 |
| Retrieval approach | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 5 |
| Worked example | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 3 |
| Total |  | 15 |


<!-- pagebreak -->

### Week 6 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| When Does RAG Increase Trust — and When Doesn't It? | Initial post + two replies | See Blackboard | 10 |
| Retrieval Failure Analysis / Glean Case | Analysis + requested artifacts | See Blackboard | 25 |
| RAG Design Document | Portfolio milestone | See Blackboard | 15 |
| TOTAL |  |  | 50 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **RAG Design Document** becomes part 6 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
