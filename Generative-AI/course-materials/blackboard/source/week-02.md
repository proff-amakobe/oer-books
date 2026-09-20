<!-- metadata: {"week": 2, "title": "The Architecture of Understanding", "chapter": "Chapter 2: The Architecture of Understanding", "milestone": "Model Selection Memo", "co": [1, 3, 5], "pslo": [1, 3, 4, 6], "points": [50, 75, 50], "case": "Bloomberg's Domain-Specific Financial Model"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 2 Module Guide

## The Architecture of Understanding

**Chapter:** Chapter 2: The Architecture of Understanding

**Project:** Model Selection Memo

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 2 Introduction & Objectives

#### Module Introduction

Sarah, a data scientist at a healthcare startup, was frustrated. Her AI-powered patient triage system worked brilliantly in testing; it could answer medical questions, understand symptoms, and provide helpful guidance. Then came production day.

Within hours, problems emerged. Simple questions like "What's a normal temperature?" were taking eight seconds and costing \$0.03 each, using the company's most powerful (and expensive) AI model for what should be instant, cheap answers. Meanwhile, complex diagnostic questions were being routed to the fast but limited model, producing oversimplified responses that missed important nuances.

The monthly API bill projection: \$47,000. For 50,000 queries.

Sarah's CTO was blunt: "We can't ship this. Figure out what's wrong or we're pulling the plug."

That weekend, Sarah dove into something she'd previously skipped: understanding how these AI models actually worked under the hood. Why were there so many different models? What made GPT-4 cost 20 times more than GPT-3.5 Turbo? How could she tell which model was right for which task?

As she studied transformer architecture, attention mechanisms, and model training processes, everything clicked. The models weren't mysteriously different, they had fundamentally different designs, training approaches, and capabilities. More importantly, she realized she could build a system that automatically routed each query to the optimal model based on its complexity and requirements.

Monday morning, Sarah deployed her intelligent routing system. Simple queries hit the fast, cheap models. Complex diagnostics went to the powerful ones. Moderate questions found the sweet spot in between.

New monthly cost projection: \$8,200. Response times: 90% under 2 seconds. Diagnostic accuracy: actually improved.

Her CTO's response: "This is why we need to understand our tools, not just use them."

This chapter is about developing Sarah's level of understanding, not as an academic exercise, but as practical knowledge that transforms how you build AI applications. You'll learn why different models exist, how their architecture shapes their capabilities, and most importantly, how to intelligently choose and orchestrate them.

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Analysis | Analyze how transformer components affect the handling of a research query. | CO 1 / PSLOs 1 |
| 2 | Analysis | Compare pre-training, post-training, and inference as explanations for model behavior. | CO 1 / PSLOs 1 |
| 3 | Evaluation | Evaluate capability, context, cost, and latency trade-offs using project requirements. | CO 3 / PSLOs 3, 4 |
| 4 | Evaluation | Critique evidence for domain specialization versus model scale. | CO 5 / PSLOs 1, 4, 6 |
| 5 | Evaluation | Justify a model-selection recommendation with an explicit unresolved question. | CO 3 / PSLOs 3, 4 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 1, CO 3, CO 5 / PSLOs 1, 3, 4, 6**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 1:** Analyze the theoretical foundations, architectures, and underlying mechanisms of Large Language Models and generative artificial intelligence systems.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

**CO 5:** Investigate and synthesize advanced methodologies, architectures, emerging technologies, and current research trends in Large Language Models and generative artificial intelligence.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 2 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 2 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 2 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop Model Selection Memo. | 100–140 min |
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

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 2: The Architecture of Understanding**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/02-llms.html).

Read the introduction and learning goals, the technical discussion, **Bloomberg's Domain-Specific Financial Model**, the wrap-up, and **Project Milestone: Model Selection Memo**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 2: Understanding Foundation Models**

Focus: Training data, architecture, model size, post-training, sampling, structured outputs, and probabilistic behavior.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 2 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: **Vaswani, A., et al. (2017). "Attention Is All You Need"**

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 2 Assignments

The discussion develops a judgment about **When Is a More Capable Model Actually Worth It**. The applied work tests that judgment against Bloomberg's Domain-Specific Financial Model. The milestone carries the evidence into Model Selection Memo.

### Assignment 1 — When Is a More Capable Model Actually Worth It?

> [SUBMISSION] Type: Discussion Board. Points: 50. Submit one initial post and two peer responses in the Week 2 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Identify two query types in your domain that require different capabilities. Decide when paying for a stronger model is justified.

**Part B — Comparison:** Connect architecture, training, or sampling behavior to that decision. Separate the chapter’s illustrative prices from evidence you would need for a real selection.

**Part C — Judgment:** Propose a routing rule and an exception that could defeat it. Explain which evidence would cause you to revise your recommendation.

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

### Assignment 2 — Architecture, Model Size, and BloombergGPT Decision Analysis

> [SUBMISSION] Type: Applied analysis. Points: 75. Submit one PDF or DOCX to Week 2 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **Bloomberg's Domain-Specific Financial Model** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Analyze Bloomberg’s Domain-Specific Financial Model as presented in Chapter 2. Distinguish benefits attributable to domain data from those attributable to parameter count.

2. Compare a domain-specialized model with a general model for two financial tasks. Build a decision matrix with capability fit, context needs, cost, latency, and evidence gaps; label assumptions rather than invent benchmark scores.

3. Trace one simple and one difficult request through a proposed model router. Explain a possible misrouting and its consequence.

4. Make a recommendation for a research-assistant domain and identify an evaluation that could disprove it. Connect the argument to your Week 1 comparison evidence.

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

### Assignment 3 — Research Assistant Milestone 2: Model Selection Memo

> [SUBMISSION] Type: Project portfolio milestone. Points: 50. Submit a PDF or DOCX to Week 2 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add a **Model Selection Memo** to your project portfolio. In one to two pages:

1. **Candidate models**: Name 2–3 models or model families you'd realistically consider for your assistant, spanning at least two tiers of capability/cost.
2. **Comparison**: For each candidate, assess capability fit, context window, approximate cost, and latency against your project's actual requirements from the Chapter 1 charter.
3. **Recommendation**: State which model you'd use for the *majority* of your assistant's traffic, and which (if any) you'd reserve for harder queries, with reasoning grounded in this chapter's architecture and training discussion, not just vendor marketing claims.
4. **Open question**: Name one thing about your domain that makes this decision genuinely uncertain, the honest kind of uncertainty a real engineering team would flag, not a rhetorical one.

#### What Makes a Strong Milestone

A strong **Model Selection Memo** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Candidate models | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Comparison | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| Recommendation | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Open question | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| Total |  | 50 |


<!-- pagebreak -->

### Week 2 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| When Is a More Capable Model Actually Worth It? | Initial post + two replies | See Blackboard | 50 |
| Architecture, Model Size, and BloombergGPT Decision Analysis | Analysis + requested artifacts | See Blackboard | 75 |
| Model Selection Memo | Portfolio milestone | See Blackboard | 50 |
| TOTAL |  |  | 175 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **Model Selection Memo** becomes part 2 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
