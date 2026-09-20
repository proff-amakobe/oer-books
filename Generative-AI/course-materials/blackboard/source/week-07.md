<!-- metadata: {"week": 7, "title": "Fine-tuning and Customization", "chapter": "Chapter 7: Fine-tuning and Customization", "milestone": "Customization Decision Memo", "co": [1, 2, 3, 5], "pslo": [1, 2, 3, 4, 6], "points": [10, 25, 15], "case": "Harvey AI's Legal Model Development"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 7 Module Guide

## Fine-tuning and Customization

**Chapter:** Chapter 7: Fine-tuning and Customization

**Project:** Customization Decision Memo

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 7 Introduction & Objectives

#### Module Introduction

Prompt engineering and RAG, the last two chapters, can take you remarkably far. But both have real limits: a prompt can only shape behavior the model is already capable of, and RAG can only supply facts, not a genuinely different writing style, a consistent specialized format, or fluency in a narrow technical vocabulary the base model barely saw during training. When you hit that ceiling, fine-tuning, continuing a pre-trained model's training on your own data, becomes the tool worth considering.

Fine-tuning is powerful and genuinely complex, and it's also the customization approach most often reached for prematurely. This chapter is built around a decision framework first and an implementation overview second, because the most common fine-tuning mistake isn't a bad training run, it's fine-tuning when a better prompt or a RAG system would have solved the problem for a fraction of the cost and complexity.

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Analysis | Compare full fine-tuning and parameter-efficient approaches conceptually. | CO 1 / PSLOs 1 |
| 2 | Creation | Design a customization decision using prompting, retrieval, and fine-tuning alternatives. | CO 2 / PSLOs 2, 3 |
| 3 | Evaluation | Evaluate a proposed dataset for quality, coverage, and sourcing limitations. | CO 3 / PSLOs 3, 4 |
| 4 | Evaluation | Critique evidence that customization improves the target task over a baseline. | CO 5 / PSLOs 1, 4, 6 |
| 5 | Evaluation | Justify the maintenance and evaluation burden of the recommended approach. | CO 3 / PSLOs 3, 4 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 1, CO 2, CO 3, CO 5 / PSLOs 1, 2, 3, 4, 6**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 1:** Analyze the theoretical foundations, architectures, and underlying mechanisms of Large Language Models and generative artificial intelligence systems.

**CO 2:** Design and implement applications that leverage Large Language Models and generative AI techniques to solve complex computational problems.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

**CO 5:** Investigate and synthesize advanced methodologies, architectures, emerging technologies, and current research trends in Large Language Models and generative artificial intelligence.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 7 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 7 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 7 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop Customization Decision Memo. | 100–140 min |
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

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 7: Fine-tuning and Customization**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/07-finetuning.html).

Read the introduction and learning goals, the technical discussion, **Harvey AI's Legal Model Development**, the wrap-up, and **Project Milestone: Customization Decision Memo**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 7: Finetuning**

Focus: When to fine-tune, reasons not to, fine-tuning and RAG, PEFT, quantization, and tactics.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 7 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models" — The paper behind the technique covered in section 7.2.

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 7 Assignments

The discussion develops a judgment about **Prompting, RAG, or Fine-Tuning**. The applied work tests that judgment against Harvey AI's Legal Model Development. The milestone carries the evidence into Customization Decision Memo.

### Assignment 1 — Prompting, RAG, or Fine-Tuning?

> [SUBMISSION] Type: Discussion Board. Points: 10. Submit one initial post and two peer responses in the Week 7 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Identify whether your assistant’s current shortfall concerns missing facts, inconsistent behavior, or a format/style requirement.

**Part B — Comparison:** Compare prompting, RAG, and fine-tuning using the Chapter 7 decision framework and a baseline.

**Part C — Judgment:** Recommend an approach and specify the evidence threshold that would make you escalate or reverse the decision.

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

### Assignment 2 — Harvey Case + Customization Decision Analysis

> [SUBMISSION] Type: Applied analysis. Points: 25. Submit one PDF or DOCX to Week 7 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **Harvey AI's Legal Model Development** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Analyze the Harvey case as a layered customization decision, identifying where retrieval and behavioral adaptation serve different needs.

2. Build a matrix comparing prompting, RAG, and fine-tuning for one legal-drafting task. Include data needs, expected benefit, evaluation burden, cost, and maintenance.

3. For a fine-tuning option, outline a data-sourcing and quality plan and compare PEFT with full fine-tuning. If you reject that option, still explain why its burdens outweigh the anticipated benefit.

4. Specify a baseline and an evaluation that distinguishes style compliance from factual grounding. Recommend the least complex approach justified by the evidence.

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

### Assignment 3 — Research Assistant Milestone 7: Customization Decision Memo

> [SUBMISSION] Type: Project portfolio milestone. Points: 15. Submit a PDF or DOCX to Week 7 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add a **Customization Decision Memo** to your project portfolio:

1. **Recommendation**: Prompting, RAG, fine-tuning, or some combination, for your specific assistant, stated clearly.
2. **Reasoning**: Walk through the decision framework from section 7.1 explicitly; don't just assert the conclusion.
3. **If fine-tuning is recommended**: A rough data-sourcing plan, approach (PEFT vs. full), and the baseline you'd compare against.
4. **If fine-tuning is not recommended**: What would need to be true about your project for that to change.

#### What Makes a Strong Milestone

A strong **Customization Decision Memo** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Recommendation | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 4 |
| Reasoning | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 3 |
| If fine-tuning is recommended | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 5 |
| If fine-tuning is not recommended | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 3 |
| Total |  | 15 |


<!-- pagebreak -->

### Week 7 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| Prompting, RAG, or Fine-Tuning? | Initial post + two replies | See Blackboard | 10 |
| Harvey Case + Customization Decision Analysis | Analysis + requested artifacts | See Blackboard | 25 |
| Customization Decision Memo | Portfolio milestone | See Blackboard | 15 |
| TOTAL |  |  | 50 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **Customization Decision Memo** becomes part 7 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
