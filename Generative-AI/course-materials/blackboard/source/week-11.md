<!-- metadata: {"week": 11, "title": "Advanced Techniques and Optimization", "chapter": "Chapter 11: Advanced Techniques and Optimization", "milestone": "Optimization Plan", "co": [2, 3, 5], "pslo": [1, 2, 3, 4, 6], "points": [10, 25, 15], "case": "Serving Search at Scale"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 11 Module Guide

## Advanced Techniques and Optimization

**Chapter:** Chapter 11: Advanced Techniques and Optimization

**Project:** Optimization Plan

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 11 Introduction & Objectives

#### Module Introduction

Chapter 2 introduced model routing as a design problem: match query complexity to model capability. Chapter 4 covered cost optimization at the integration layer: caching, batching, provider comparison. Both were genuinely useful, and both stopped short of a harder question that only shows up once a system is actually running at real scale: what do you do when routing and caching alone aren't enough, when you need the underlying models themselves to run faster, cheaper, and more reliably under real load, not just be chosen intelligently?

This chapter goes one level deeper into that question: combining multiple models for better results than any one alone, shrinking models without gutting their quality, and the performance engineering that separates a system that works in a demo from one that holds up under thousands of simultaneous users. None of this is required reading for every project, if your assistant handles modest traffic, Chapters 2 and 4's routing and caching may be all you ever need, but understanding what's available at this next level of scale is part of being able to recognize when you'd actually need it.

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Analysis | Compare ensemble and compression approaches by benefit and complexity. | CO 2 / PSLOs 2, 3 |
| 2 | Creation | Design an experiment-driven optimization plan with a defensible baseline. | CO 2 / PSLOs 2, 3 |
| 3 | Evaluation | Evaluate quality, cost, and latency trade-offs under explicit load assumptions. | CO 3 / PSLOs 3, 4 |
| 4 | Evaluation | Critique technical evidence for a serving optimization. | CO 5 / PSLOs 1, 4, 6 |
| 5 | Evaluation | Justify when advanced optimization is unnecessary for the project. | CO 3 / PSLOs 3, 4 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 2, CO 3, CO 5 / PSLOs 1, 2, 3, 4, 6**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 2:** Design and implement applications that leverage Large Language Models and generative AI techniques to solve complex computational problems.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

**CO 5:** Investigate and synthesize advanced methodologies, architectures, emerging technologies, and current research trends in Large Language Models and generative artificial intelligence.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 11 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 11 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 11 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop Optimization Plan. | 100–140 min |
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

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 11: Advanced Techniques and Optimization**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/11-optimization.html).

Read the introduction and learning goals, the technical discussion, **Serving Search at Scale**, the wrap-up, and **Project Milestone: Optimization Plan**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 9: Inference Optimization**

Focus: Inference performance, AI accelerators, model optimization, and inference-service optimization.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 11 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: Any current vLLM or TGI (Text Generation Inference) documentation on continuous batching and KV-caching, these projects are the practical, widely used implementations of section 11.3's concepts.

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 11 Assignments

The discussion develops a judgment about **When Is Optimization Worth the Complexity**. The applied work tests that judgment against Serving Search at Scale. The milestone carries the evidence into Optimization Plan.

### Assignment 1 — When Is Optimization Worth the Complexity?

> [SUBMISSION] Type: Discussion Board. Points: 10. Submit one initial post and two peer responses in the Week 11 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Identify the likely bottleneck in your assistant under an explicit workload.

**Part B — Comparison:** Compare a simple routing/cache change with a more advanced optimization. Include quality regression, operational cost, and maintenance complexity.

**Part C — Judgment:** Define a measurable trigger for adopting the advanced option and a rollback criterion if it disappoints.

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

### Assignment 2 — Serving-at-Scale Trade-off Analysis

> [SUBMISSION] Type: Applied analysis. Points: 25. Submit one PDF or DOCX to Week 11 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **Serving Search at Scale** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Use Serving Search at Scale to explain how batching, caching, and model optimization can interact rather than deliver independent gains.

2. Define a baseline workload, latency target, and cost assumption. Compare two optimization proposals, including one simple and one advanced technique.

3. Design an experiment that changes one factor at a time before considering combinations. Include quality checks, tail latency, and a workload where the proposed improvement could fail.

4. Recommend a staged optimization plan. Quantify only what your assumptions or evidence support, and explain whether the technique fits your own project now, later, or not at all.

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

### Assignment 3 — Research Assistant Milestone 11: Optimization Plan

> [SUBMISSION] Type: Project portfolio milestone. Points: 15. Submit a PDF or DOCX to Week 11 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add an **Optimization Plan** to your project portfolio:

1. **Expected scale**: Your realistic query volume estimate.
2. **Cost and latency budget**: What you'd expect to pay and how fast a response needs to be.
3. **First bottleneck**: What would break first under significantly higher load, and why.
4. **Applicability call**: Whether this chapter's advanced techniques apply to your project now, later, or not at all, with reasoning.

#### What Makes a Strong Milestone

A strong **Optimization Plan** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Expected scale | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 4 |
| Cost and latency budget | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 3 |
| First bottleneck | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 5 |
| Applicability call | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 3 |
| Total |  | 15 |


<!-- pagebreak -->

### Week 11 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| When Is Optimization Worth the Complexity? | Initial post + two replies | See Blackboard | 10 |
| Serving-at-Scale Trade-off Analysis | Analysis + requested artifacts | See Blackboard | 25 |
| Optimization Plan | Portfolio milestone | See Blackboard | 15 |
| TOTAL |  |  | 50 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **Optimization Plan** becomes part 11 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
