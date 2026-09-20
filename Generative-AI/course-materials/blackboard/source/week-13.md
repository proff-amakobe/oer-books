<!-- metadata: {"week": 13, "title": "Safety, Ethics, and Responsible AI", "chapter": "Chapter 13: Safety, Ethics, and Responsible AI", "milestone": "Ethics & Fairness Review", "co": [4, 5], "pslo": [1, 4, 5, 6], "points": [50, 75, 50], "case": "How Microsoft Developed Responsible AI Practices"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 13 Module Guide

## Safety, Ethics, and Responsible AI

**Chapter:** Chapter 13: Safety, Ethics, and Responsible AI

**Project:** Ethics & Fairness Review

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 13 Introduction & Objectives

#### Module Introduction

A healthcare AI system saves clinicians real time, until a systematic bias in its recommendations is discovered, quietly affecting thousands of patients across months of routine use before anyone catches it. Nothing about the system's engineering was sloppy; it performed exactly as it was optimized to perform. The technical success and the ethical failure happened in the same system, at the same time, which is precisely the point: good engineering and responsible AI are not the same achievement, and a project can score highly on one while quietly failing the other.

This chapter is where Chapter 1's opening ethical questions, who creates the training data, who benefits, who's responsible when something goes wrong, become concrete, practical design work rather than open discussion prompts. Chapter 12 covered the technical machinery of alignment, how a model is trained to behave safely in the first place. This chapter asks the applied question that sits on top of that machinery: given a model that's been aligned as well as current techniques allow, what does it actually take to deploy it responsibly, and who's accountable when that deployment falls short?

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Analysis | Analyze responsibilities across individual, organizational, and industry levels. | CO 4 / PSLOs 5 |
| 2 | Evaluation | Evaluate a domain-specific bias risk and competing mitigations. | CO 4 / PSLOs 5 |
| 3 | Creation | Design a privacy and explainability approach for affected stakeholders. | CO 4 / PSLOs 5 |
| 4 | Evaluation | Critique ethical frameworks using the chapter’s scholarly readings. | CO 5 / PSLOs 1, 4, 6 |
| 5 | Evaluation | Justify a responsible deployment decision that may change or limit the system. | CO 4 / PSLOs 5 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 4, CO 5 / PSLOs 1, 4, 5, 6**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 4:** Assess ethical, societal, legal, and security considerations associated with the development and deployment of generative artificial intelligence systems.

**CO 5:** Investigate and synthesize advanced methodologies, architectures, emerging technologies, and current research trends in Large Language Models and generative artificial intelligence.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 13 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 13 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 13 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop Ethics & Fairness Review. | 100–140 min |
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

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 13: Safety, Ethics, and Responsible AI**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/13-ethics.html).

Read the introduction and learning goals, the technical discussion, **How Microsoft Developed Responsible AI Practices**, the wrap-up, and **Project Milestone: Ethics & Fairness Review**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 10: selected guardrails, monitoring, and user-feedback material**

Focus: Engineering context for production responsibility; use the primary chapter’s Further Reading for ethical scholarship.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 13 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: Jobin, A., et al. (2019). "The Global Landscape of AI Ethics Guidelines" — A systematic review of AI ethics frameworks worldwide, useful for seeing how much, and how little, consensus actually exists.

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 13 Assignments

The discussion develops a judgment about **Who Bears Responsibility When an AI System Causes Harm**. The applied work tests that judgment against How Microsoft Developed Responsible AI Practices. The milestone carries the evidence into Ethics & Fairness Review.

### Assignment 1 — Who Bears Responsibility When an AI System Causes Harm?

> [SUBMISSION] Type: Discussion Board. Points: 50. Submit one initial post and two peer responses in the Week 13 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Identify a harm that could occur even if your assistant meets its technical metric. Map who benefits and who bears the risk.

**Part B — Comparison:** Compare an individual, organizational, and industry-level responsibility for that harm. Identify an accountability gap.

**Part C — Judgment:** Recommend a design change or a reason to withhold deployment, supported by ethical scholarship from the chapter.

#### Peer Response Guidance

Reply to **two classmates, 150–200 words each**. Challenge an assumption, introduce a counterexample, identify an overlooked trade-off, ask a substantive question, or connect the design to another course concept. Agreement, praise, or summary alone is insufficient. Explain why your contribution matters to the peer’s decision.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Conceptual application | Applies the week’s concepts accurately to the concrete problem and compares alternatives. | 15 |
| Evidence and judgment | Uses chapter evidence to defend a position, identifies a limitation, and explains a trade-off. | 15 |
| Project connection | Connects the argument to the same research assistant and its users. | 10 |
| Peer engagement | Two substantive replies meet the length guidance and advance the reasoning. | 10 |
| Total |  | 50 |

Cite the chapter or companion selection when it supports your reasoning. Cite outside claims to identifiable sources; an unsupported assertion does not become evidence because a model generated it.


<!-- pagebreak -->

### Assignment 2 — Microsoft Responsible AI Case + Stakeholder/Fairness Analysis

> [SUBMISSION] Type: Applied analysis. Points: 75. Submit one PDF or DOCX to Week 13 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **How Microsoft Developed Responsible AI Practices** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Use the Microsoft Responsible AI case to distinguish a written principle from an operational review process that can stop a deployment.

2. Create a stakeholder/pipeline map and locate one plausible bias pathway. Specify evidence needed to assess the effect on an affected group rather than assuming a numerical disparity.

3. Compare two mitigations, including their costs and possible new harms. Explain what affected users should be able to understand or contest.

4. Draft a review decision with an accountable owner, a condition for reconsideration, and a privacy control. Identify a concrete change to your research-assistant design.

#### Evidence and Submission Guidance

Use the same assumptions consistently across the response. A design exercise can be completed as a documented walkthrough; do not disguise a projected outcome as an experiment. Where a task asks for live observations, use accessible tools and retain the relevant outputs. If access prevents the required comparison, request an instructor-provided output set rather than inventing results. Explain what your evidence cannot establish.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Case and problem framing | Accurately identifies the case’s design tension and separates given facts from assumptions. | 15 |
| Analysis and comparison | Completes the requested comparison or experiment with a defensible method and evidence. | 25 |
| Limits and trade-offs | Evaluates failure modes, alternative explanations, and relevant constraints. | 15 |
| Recommendation and transfer | Connects a justified recommendation to the research-assistant design. | 15 |
| Traceability and clarity | Includes requested artifacts, source references, and clear labels for measured/projected results. | 5 |
| Total |  | 75 |


<!-- pagebreak -->

### Assignment 3 — Research Assistant Milestone 13: Ethics & Fairness Review

> [SUBMISSION] Type: Project portfolio milestone. Points: 50. Submit a PDF or DOCX to Week 13 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add an **Ethics & Fairness Review** to your project portfolio:

1. **Stakeholder map**: Who's directly and indirectly affected by your assistant.
2. **Bias risk**: One concrete, domain-specific bias risk and where in your pipeline it could enter.
3. **Explainability approach**: The level(s) your project needs, and why.
4. **Privacy posture**: What sensitive data your project touches, if any, and your approach to protecting it.

#### What Makes a Strong Milestone

A strong **Ethics & Fairness Review** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Stakeholder map | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Bias risk | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| Explainability approach | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Privacy posture | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| Total |  | 50 |


<!-- pagebreak -->

### Week 13 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| Who Bears Responsibility When an AI System Causes Harm? | Initial post + two replies | See Blackboard | 50 |
| Microsoft Responsible AI Case + Stakeholder/Fairness Analysis | Analysis + requested artifacts | See Blackboard | 75 |
| Ethics & Fairness Review | Portfolio milestone | See Blackboard | 50 |
| TOTAL |  |  | 175 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **Ethics & Fairness Review** becomes part 13 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
