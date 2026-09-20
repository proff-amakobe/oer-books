<!-- metadata: {"week": 5, "title": "Agents — AI That Takes Action", "chapter": "Chapter 5: Agents — AI That Takes Action", "milestone": "Agent Capability Plan", "co": [2, 3, 4], "pslo": [2, 3, 4, 5], "points": [50, 75, 50], "case": "Klarna's AI Customer Service Agent"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 5 Module Guide

## Agents — AI That Takes Action

**Chapter:** Chapter 5: Agents — AI That Takes Action

**Project:** Agent Capability Plan

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 5 Introduction & Objectives

#### Module Introduction

Picture two versions of the same assistant. You ask both: "Do I have a scheduling conflict next Tuesday, and if so, can you fix it?"

The first version reads your calendar, if you've pasted it in, and tells you: "Yes, you have two meetings booked at 2 PM: the budget review and the client call." That's the end of the interaction. You now know something you didn't know before, and you're on your own to actually resolve it, open your calendar, decide which meeting to move, draft a message to the other attendees, send it.

The second version does something categorically different. It checks your calendar directly. It identifies the conflict. It proposes that the client call be moved thirty minutes later, checks that the new time also works for everyone else on the invite, drafts a reschedule message, and asks you one question before sending anything: "Move the client call to 2:30 PM and notify attendees? Yes/No." You say yes, and it's done.

Both versions used the same underlying language model. The difference between them has nothing to do with how smart the model is and everything to do with what it's allowed to *do*. That difference, from answering questions to taking action in the world, is what this chapter is about, and it's arguably the single most consequential shift happening in applied generative AI right now. It's also the shift that raises this book's safety and control questions most sharply, not as an abstract concern for a later chapter, but as a design problem you'll need to solve for your own project this week.

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Analysis | Analyze the change in risk when an assistant can act rather than only answer. | CO 4 / PSLOs 5 |
| 2 | Analysis | Compare agent architectures against a bounded research-assistant workflow. | CO 2 / PSLOs 2, 3 |
| 3 | Creation | Design a narrowly scoped tool specification and permission boundary. | CO 2 / PSLOs 2, 3 |
| 4 | Evaluation | Evaluate action quality and recovery using cost and reversibility. | CO 3 / PSLOs 3, 4 |
| 5 | Evaluation | Justify a human checkpoint for a concrete agent failure mode. | CO 4 / PSLOs 5 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 2, CO 3, CO 4 / PSLOs 2, 3, 4, 5**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 2:** Design and implement applications that leverage Large Language Models and generative AI techniques to solve complex computational problems.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

**CO 4:** Assess ethical, societal, legal, and security considerations associated with the development and deployment of generative artificial intelligence systems.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 5 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 5 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 5 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop Agent Capability Plan. | 100–140 min |
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

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 5: Agents — AI That Takes Action**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/05-agents.html).

Read the introduction and learning goals, the technical discussion, **Klarna's AI Customer Service Agent**, the wrap-up, and **Project Milestone: Agent Capability Plan**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 6: RAG and Agents — agent selections only**

Focus: Agent Overview, Tools, Planning, Agent Failure Modes and Evaluation, and Memory.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 5 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: Yao, S., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models" — The paper that introduced the ReAct pattern covered in section 5.3.

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 5 Assignments

The discussion develops a judgment about **How Much Autonomy Should an AI Agent Have**. The applied work tests that judgment against Klarna's AI Customer Service Agent. The milestone carries the evidence into Agent Capability Plan.

### Assignment 1 — How Much Autonomy Should an AI Agent Have?

> [SUBMISSION] Type: Discussion Board. Points: 50. Submit one initial post and two peer responses in the Week 5 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Propose one action your assistant might take and place it on the human-in-the-loop spectrum.

**Part B — Comparison:** Compare autonomous execution with approval before execution using cost, reversibility, and ambiguity.

**Part C — Judgment:** Explain how goal hijacking changes the risk and name the narrowest tool permission that still serves the user.

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

### Assignment 2 — Klarna Agent Case + Human-in-the-Loop Boundary Analysis

> [SUBMISSION] Type: Applied analysis. Points: 75. Submit one PDF or DOCX to Week 5 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **Klarna's AI Customer Service Agent** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Analyze the Klarna case as presented in Chapter 5. Separate answering a customer from changing an order or issuing a refund.

2. Build an action/approval matrix for three customer-service actions. For each, state inputs, allowed scope, reversibility, and escalation conditions.

3. Walk through a malformed input or ambiguous refund request. Show how a narrow tool schema and a human checkpoint change the outcome.

4. Compare ReAct with plan-and-execute or reflection for this bounded workflow. Recommend a pattern and state one unresolved failure mode.

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

### Assignment 3 — Research Assistant Milestone 5: Agent Capability Plan

> [SUBMISSION] Type: Project portfolio milestone. Points: 50. Submit a PDF or DOCX to Week 5 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add an **Agent Capability Plan** to your project portfolio:

1. **Chosen capability**: The one action-taking capability from section 5.8 that you designed, stated clearly.
2. **Human-in-the-loop placement**: Where it sits on the spectrum, and your reasoning, grounded explicitly in cost and reversibility.
3. **Tool specification**: The narrowly scoped tool this capability would need, named and described the way you'd write it for a model to actually use correctly.
4. **Risk and mitigation**: One plausible failure mode (goal hijacking, a malformed input, an ambiguous instruction) and the specific safeguard, sandboxing, scoping, or a human checkpoint, that addresses it.

#### What Makes a Strong Milestone

A strong **Agent Capability Plan** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Chosen capability | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Human-in-the-loop placement | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| Tool specification | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Risk and mitigation | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| Total |  | 50 |


<!-- pagebreak -->

### Week 5 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| How Much Autonomy Should an AI Agent Have? | Initial post + two replies | See Blackboard | 50 |
| Klarna Agent Case + Human-in-the-Loop Boundary Analysis | Analysis + requested artifacts | See Blackboard | 75 |
| Agent Capability Plan | Portfolio milestone | See Blackboard | 50 |
| TOTAL |  |  | 175 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **Agent Capability Plan** becomes part 5 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
