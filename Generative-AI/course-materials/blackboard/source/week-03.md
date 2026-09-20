<!-- metadata: {"week": 3, "title": "The Art and Science of Prompting", "chapter": "Chapter 3: The Art and Science of Prompting", "milestone": "Prompt Library v1", "co": [2, 3, 4], "pslo": [2, 3, 4, 5], "points": [50, 75, 50], "case": "How Notion Built Its AI Writing Assistant"} -->
# GENERATIVE AI

Graduate Course | First Open Edition

# Week 3 Module Guide

## The Art and Science of Prompting

**Chapter:** Chapter 3: The Art and Science of Prompting

**Project:** Prompt Library v1

**Est. Total Time:** Approx. 5–7 hours across the week

## SECTION 1 — MODULE INTRODUCTION & OBJECTIVES

### 1. Week 3 Introduction & Objectives

#### Module Introduction

Welcome to the art and science of prompt engineering—the critical skill that transforms generic AI models into powerful, specialized tools. While the previous chapters focused on understanding and selecting the right models, this chapter is about learning to communicate effectively with those models to achieve precise, reliable, and high-quality results.

Prompt engineering is often described as the "new programming language" of the AI era. Just as traditional programming requires understanding syntax, logic, and best practices, prompt engineering requires understanding how to structure requests, provide context, and guide model behavior to achieve desired outcomes.

In this chapter, you'll master the fundamental techniques that separate novice AI users from experts: few-shot learning, chain-of-thought prompting, prompt templates, and safety considerations. You'll enhance your research assistant with a sophisticated prompt management system that can automatically select and optimize prompts based on the type of research query being processed.

> [GUIDANCE] Course Author Guidance Note: Keep the author’s narrative and the research-assistant project connection intact. Adapt examples used in discussion to the cohort, but retain the approved outcomes, milestone requirements, and assessment totals. The written Start Here item supplies Blackboard orientation.


<!-- pagebreak -->

### Weekly Objectives

After completing this week, you will be able to demonstrate the following outcomes through the discussion, applied work, and portfolio evidence.

| # | Bloom’s Level | Objective | Aligns to |
| --- | --- | --- | --- |
| 1 | Creation | Design task-specific prompts with explicit role, context, task, format, and constraints. | CO 2 / PSLOs 2, 3 |
| 2 | Analysis | Compare zero-shot and few-shot prompt behavior with controlled inputs. | CO 3 / PSLOs 3, 4 |
| 3 | Creation | Develop reusable prompt templates for distinct research-assistant requests. | CO 2 / PSLOs 2, 3 |
| 4 | Evaluation | Evaluate prompt quality with stated criteria and an honest failure example. | CO 3 / PSLOs 3, 4 |
| 5 | Evaluation | Assess a prompt-injection risk and justify a concrete defensive boundary. | CO 4 / PSLOs 5 |

#### Approved CO / PSLO Alignment

This week emphasizes **CO 2, CO 3, CO 4 / PSLOs 2, 3, 4, 5**. Each objective uses only the approved CO-to-PSLO mapping; a PSLO code indicates curricular alignment, not that this single week independently demonstrates the entire program outcome.

**CO 2:** Design and implement applications that leverage Large Language Models and generative AI techniques to solve complex computational problems.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

**CO 4:** Assess ethical, societal, legal, and security considerations associated with the development and deployment of generative artificial intelligence systems.

> [GUIDANCE] Assessment alignment: Judge the discussion for conceptual reasoning and evidence, the applied activity for analysis of a concrete case, and the milestone for a justified project decision. Preserve the approved CO wording. The course map contains the full approved PSLO statements.


<!-- pagebreak -->

## SECTION 2 — WEEK PLAN

### 2. Week 3 Plan

> [GUIDANCE] Course Author Guidance — Suggested Schedule: These day groups are pacing suggestions, not four separate deadlines. Set actual due dates and the course time zone in Blackboard before release. Preserve the sequence and point totals; adjust pacing for the cohort without adding unrelated tasks.

#### Start Here

Begin with **Week 3 Start Here: Module Overview**, a written Blackboard content page. It identifies the reading order, the connection among assignments, and the portfolio artifact. Keep the primary textbook open while working; the companion selection supports the week’s decision rather than replacing the chapter.

#### Suggested Weekly Schedule

| Pacing group | Activities | Estimated time |
| --- | --- | --- |
| Monday–Tuesday | Read Week 3 Start Here: Module Overview. Begin the primary chapter and focused companion selection. Annotate one decision that matters to your assistant. | 90–120 min |
| Wednesday–Thursday | Finish focused reading; work through the activity setup and draft the initial discussion post with chapter evidence. | 70–90 min |
| Friday–Saturday | Complete the case analysis, respond to two peers, and develop Prompt Library v1. | 100–140 min |
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

Moody Amakobe. *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*. First Open Edition. Global Data Science Institute, 2026. **Chapter 3: The Art and Science of Prompting**.

[Read the primary chapter online](https://proff-amakobe.github.io/oer-books/Generative-AI/original/03-prompt-engineering.html).

Read the introduction and learning goals, the technical discussion, **How Notion Built Its AI Writing Assistant**, the wrap-up, and **Project Milestone: Prompt Library v1**. Keep notes on the assumptions that make the chapter’s recommendation appropriate. The case is assigned as presented in the chapter; do not treat a reported result as a current independent benchmark.

#### Required Reading 2 — Companion Textbook

Chip Huyen. *AI Engineering: Building Applications with Foundation Models*. O’Reilly Media, 2024. ISBN 9781098166298.

**Chapter 5: Prompt Engineering**

Focus: Zero/few-shot learning, system/user prompts, context, prompt practices, evaluation, and defensive prompting.

Read to compare how the two books frame the same engineering decision. Note one useful connection and one boundary of the companion’s coverage. No page numbers are prescribed. [Publisher listing](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [Author’s contents](https://github.com/chiphuyen/aie-book/blob/main/ToC.md).

#### Required Blackboard Item

**Week 3 Start Here: Module Overview** — a written content page in this week’s Blackboard module. Read it before beginning the assignments. It explains what to read first, how the tasks connect, what portfolio artifact you are producing, and what deserves special attention.

#### Additional / Optional Readings

Choose one item from the primary chapter’s **Further Reading** that helps resolve a question in your project. Read for the claim, evidence, limitations, and relevance; no additional response is required.

Author-listed starting point: **Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"**

#### Reading-to-Work Notes

Keep brief notes under three headings: the claim or design principle; the evidence or example supporting it; and the consequence for your assistant. These notes support your submitted work and are not a fourth assignment.


<!-- pagebreak -->

## SECTION 4 — ASSIGNMENTS

### 4. Week 3 Assignments

The discussion develops a judgment about **Is Better Prompting a Substitute for a Better Model**. The applied work tests that judgment against How Notion Built Its AI Writing Assistant. The milestone carries the evidence into Prompt Library v1.

### Assignment 1 — Is Better Prompting a Substitute for a Better Model?

> [SUBMISSION] Type: Discussion Board. Points: 50. Submit one initial post and two peer responses in the Week 3 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Choose one task your assistant performs poorly. Decide whether its limiting factor is the prompt, the model, or missing knowledge.

**Part B — Comparison:** Compare a more deliberate prompt with changing models. Specify a common input and evaluation criterion that make the comparison fair.

**Part C — Judgment:** Explain how a prompt-injection attempt could change your recommendation. Identify a limitation prompting alone cannot resolve.

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

### Assignment 2 — Prompt Redesign, Few-Shot Comparison, and Injection Test

> [SUBMISSION] Type: Applied analysis. Points: 75. Submit one PDF or DOCX to Week 3 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **How Notion Built Its AI Writing Assistant** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Use the Notion writing-assistant case to select two distinct task types that should not share one generic instruction. Explain why separation matters.

2. Write a baseline and revised prompt using role, context, task, format, and constraints. Compare zero-shot with few-shot behavior on the same three safe test inputs, keeping the model and other settings fixed where possible.

3. Create one harmless injection test against your own draft prompt. Record the input, observed or explicitly projected response, and the boundary it tests; use synthetic data.

4. Evaluate results with a small, stated rubric. Recommend a revision and explain a remaining failure. Label projected responses clearly and do not present them as measured results.

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

### Assignment 3 — Research Assistant Milestone 3: Prompt Library v1

> [SUBMISSION] Type: Project portfolio milestone. Points: 50. Submit a PDF or DOCX to Week 3 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add a **Prompt Library v1** to your project portfolio:

1. **System prompt**: The full text of your assistant's system prompt, plus 2–3 sentences justifying the role, tone, and constraints you chose.
2. **3–5 task prompts**: The full text of each, covering the distinct request types your assistant needs to handle.
3. **One worked test**: For at least one prompt, show a typical input, the response it produced (real or realistically projected), and one honest note on where it still falls short.
4. **Adversarial check**: Briefly describe one prompt-injection attempt you tested against your system prompt, and what happened.

#### What Makes a Strong Milestone

A strong **Prompt Library v1** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| System prompt | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| 3–5 task prompts | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| One worked test | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Adversarial check | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 10 |
| Total |  | 50 |


<!-- pagebreak -->

### Week 3 Assignment Summary

| Assignment | Submission | Due | Points |
| --- | --- | --- | --- |
| Is Better Prompting a Substitute for a Better Model? | Initial post + two replies | See Blackboard | 50 |
| Prompt Redesign, Few-Shot Comparison, and Injection Test | Analysis + requested artifacts | See Blackboard | 75 |
| Prompt Library v1 | Portfolio milestone | See Blackboard | 50 |
| TOTAL |  |  | 175 |

#### Before You Submit

- Check that each required prompt component is visibly addressed, using descriptive headings.
- Match the discussion claim to the analysis evidence and the portfolio decision. Explain a disagreement rather than hiding it.
- Keep references traceable: name the chapter or section, identify external sources, and date time-sensitive evidence.
- Confirm that diagrams and tables are readable and that the submitted files open normally.
- Label assumptions, illustrative examples, measured observations, and projections consistently.
- Read the rubric once as a reviewer. Identify where the evidence for each criterion appears.

#### Portfolio Continuity

This week’s **Prompt Library v1** becomes part 3 of the same research-assistant portfolio. Keep the submitted version, feedback, and a revised version with a short change note. In Week 16, reconcile the fifteen artifacts into one design. Do not change the project domain casually; if a change is necessary, explain its effect on earlier decisions.

#### Self-Check for Understanding

Before closing the week, ask yourself: What decision can I now defend more clearly? Which alternative did I reject, and why? What evidence would change my mind? Where does the user experience a failure if my assumptions are wrong? These are reflection prompts to improve the three submissions, not an additional graded assignment.

> [SUBMISSION] Submission check: Use the correct Blackboard item for each assignment. Retain a local editable copy and verify the uploaded file preview. Use the dates and time zone configured by your instructor.
