<!-- metadata: {"week": 4, "title": "Working with APIs and Integration", "chapter": "Chapter 4: Working with APIs and Integration", "milestone": "Integration Architecture Diagram", "co": [2, 3], "pslo": [2, 3, 4], "points": [10, 25, 15], "case": "A Provider Outage, Worked Through"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 4 Module Guide

## Working with APIs and Integration

**Chapter:** Chapter 4: Working with APIs and Integration

**Project:** Integration Architecture Diagram

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 4 Introduction & Objectives

#### Module Introduction

Now that you've mastered the art of crafting effective prompts, it's time to think through how a well-chosen model and a well-designed prompt actually reach your users reliably. This chapter bridges the gap between experimental AI projects and applications that handle real users, real budgets, and real uptime expectations.

Working with AI APIs presents challenges that don't show up in a quick prototype: managing rate limits, handling unpredictable latency, optimizing cost across providers, and staying available when an external service fails. These are the engineering practices that separate a weekend demo from something a business can actually depend on.

In this chapter, you'll design, not implement, the integration layer for your research assistant: how it talks to one or more providers, what happens when a call fails, how caching and cost tracking fit together, and what a provider outage would actually mean for your users. As with every chapter so far, the deliverable is a design document, not a working system, though nothing stops you from building it if you want the extra practice.

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Creation | Design a provider integration flow that matches workload and reliability needs. | CO 2 / PSLOs 2, 3 |
| 2 | Analysis | Compare retries, circuit breakers, and failover across distinct failure modes. | CO 3 / PSLOs 3, 4 |
| 3 | Evaluation | Evaluate caching choices against freshness and latency requirements. | CO 3 / PSLOs 3, 4 |
| 4 | Creation | Develop a cost and rate-limit plan with explicit assumptions. | CO 2 / PSLOs 2, 3 |
| 5 | Creation | Design monitoring and key/data-handling boundaries for the integration architecture. | CO 2 / PSLOs 2, 3 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 2, CO 3 / PSLOs 2, 3, 4**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 2:** Design and implement applications that leverage Large Language Models and generative AI techniques to solve complex computational problems.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 4 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 4 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 4 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop Integration Architecture Diagram. | 100–140 min |
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

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 4: Working with APIs and Integration**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/04-apis-integration.html).

Read the introduction and learning goals, the technical discussion, **A Provider Outage, Worked Through**, the wrap-up, and **Project Milestone: Integration Architecture Diagram**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 10: AI Engineering Architecture and User Feedback**

Focus: Select integration-related material on architecture, model router/gateway, caches, monitoring, and orchestration.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 4 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: Nygard, M. (2018). *Release It!* — The standard reference for resilience patterns (circuit breakers, bulkheads, timeouts) that this chapter applies to AI APIs specifically.

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 4 Assignments

The discussion develops a judgment about **Reliability, Cost, and Vendor Dependence**. The applied work tests that judgment against A Provider Outage, Worked Through. The milestone carries the evidence into Integration Architecture Diagram.

### Assignment 1 — Reliability, Cost, and Vendor Dependence

> [SUBMISSION] Type: Discussion Board. Points: 10. Submit one initial post and two peer responses in the Week 4 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Choose between single-provider simplicity and multi-provider resilience for your assistant. State the workload and budget assumptions.

**Part B — Comparison:** Compare what happens under rate limiting, timeout, and a provider outage; explain where retrying helps and where it makes the incident worse.

**Part C — Judgment:** Defend a fallback or degraded-service choice using quality, cost, latency, and user communication.

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

### Assignment 2 — Provider Outage Response and Integration Trade-off Analysis

> [SUBMISSION] Type: Applied analysis. Points: 25. Submit one PDF or DOCX to Week 4 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **A Provider Outage, Worked Through** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Use A Provider Outage, Worked Through: 100,000 queries a day and a three-day maintenance window starting tomorrow. Separate known scenario facts from your assumptions.

2. Sketch a request flow with timeout, retry, circuit-breaker, fallback, and cache decisions. Include one case where stale cached content is unacceptable.

3. Estimate fallback volume and cost symbolically or with clearly sourced unit prices. Identify quality and latency differences users may notice.

4. Write a brief incident communication and recovery plan. Specify two signals that would trigger escalation or restoration of normal routing.

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

### Assignment 3 — Research Assistant Milestone 4: Integration Architecture Diagram

> [SUBMISSION] Type: Project portfolio milestone. Points: 15. Submit a PDF or DOCX to Week 4 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add an **Integration Architecture Diagram** to your project portfolio, a visual diagram plus a short narrative (one page is plenty):

1. **The diagram**: Show your assistant's request flow, from user input through provider selection to response, including where failover and caching sit in that flow.
2. **Failure handling table**: List your three most likely failure modes (from section 4.3) and your designed response to each.
3. **Cost and rate-limit plan**: One paragraph on your expected usage pattern and which cost-optimization lever from section 4.4 you'd apply first.

#### What Makes a Strong Milestone

A strong **Integration Architecture Diagram** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| The diagram | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 5 |
| Failure handling table | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 5 |
| Cost and rate-limit plan | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 5 |
| Total |  | 15 |


<!-- pagebreak -->

### Week 4 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| Reliability, Cost, and Vendor Dependence | Initial post + two replies | See Blackboard | 10 |
| Provider Outage Response and Integration Trade-off Analysis | Analysis + requested artifacts | See Blackboard | 25 |
| Integration Architecture Diagram | Portfolio milestone | See Blackboard | 15 |
| TOTAL |  |  | 50 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **Integration Architecture Diagram** becomes part 4 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
