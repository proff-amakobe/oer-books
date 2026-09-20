# Week 12 — Blackboard Assignment Copy

## Discussion Board

### Assignment 1 — Can We Know Whether a Model Is Aligned?

> [SUBMISSION] Type: Discussion Board. Points: 50. Submit one initial post and two peer responses in the Week 12 Discussion item. Use Blackboard’s displayed due dates.

Write an initial post of approximately **350–500 words**. Make a defensible claim, compare alternatives, and use evidence from the assigned material. Connect your judgment to your research-assistant design rather than summarizing the chapter.

#### Discussion Prompt

**Part A — Position:** Choose an observable behavior that you would count as evidence of alignment in your domain.

**Part B — Comparison:** Compare training-based alignment with runtime guardrails. Explain why success on a test does not establish alignment in all contexts.

**Part C — Judgment:** Name one conflict between helpfulness and accuracy/honesty, then propose a bounded red-team test and mitigation.

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


## Assignment 2

### Assignment 2 — Constitutional AI + Structured Red-Team Exercise

> [SUBMISSION] Type: Applied analysis. Points: 75. Submit one PDF or DOCX to Week 12 Assignment 2. Aim for 600–800 words of analysis plus the requested matrix, diagram, prompts, or evidence appendix; these artifacts are not included in the word guide.

Use **Anthropic's Constitutional AI in Practice** as the anchor. Your task is to explain and test the design reasoning, not retell the case. Work through the following connected steps with informative headings.

#### Detailed Response Prompt

1. Analyze the Constitutional AI case: distinguish explicit guiding principles from proof that every output obeys them. Compare the role of human feedback with principle-based critique.

2. Write three concrete evaluation principles for your assistant. Design three safe adversarial scenarios using synthetic data and your own system or a paper walkthrough.

3. Record the attempted behavior, expected boundary, observed or projected outcome, and severity. Keep results distinguishable from predictions and avoid testing other people’s systems.

4. Recommend one specific mitigation and a retest. Explain what interpretability or further evaluation might reveal and what remains unknown.

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


## Project Milestone

### Assignment 3 — Research Assistant Milestone 12: Red-Team Review

> [SUBMISSION] Type: Project portfolio milestone. Points: 50. Submit a PDF or DOCX to Week 12 Project Milestone. Preserve the author’s required elements below and retain an editable copy for Week 16.

#### Detailed Milestone Instructions — Author’s Requirements

This week, add a **Red-Team Review** to your project portfolio:

1. **Adversarial walkthrough**: What a deliberately adversarial user would try against your current design, and whether it would hold up.
2. **Domain-specific alignment risk**: One concrete way your assistant's implicit objectives could conflict with accuracy or honesty in your specific domain.
3. **Mitigation**: One concrete safeguard addressing the risk you identified, grounded in specific techniques from this book, not a general statement of good intentions.

#### What Makes a Strong Milestone

A strong **Red-Team Review** addresses the actual users and constraints from your charter. Make the decision explicit, explain the rejected alternative, and connect claims to evidence from the week’s reading or applied work. The required elements above control the submission; additional pages or a working implementation do not substitute for missing reasoning.

Use the scope and length stated by the author. Where no length is specified, a focused 1–3 pages plus necessary diagrams or evidence is normally sufficient; the literature review may need more space for its 5–8 sources. Identify one uncertainty or decision to revisit. Label any change to a prior milestone so your portfolio remains coherent.

#### Grading Criteria

| Criterion | Full-credit evidence | Points |
| --- | --- | --- |
| Adversarial walkthrough | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 20 |
| Domain-specific alignment risk | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Mitigation | Addresses this author-required element specifically for the project, with reasoning and evidence consistent with prior milestones. | 15 |
| Total |  | 50 |


## Start Here Page

# Week 12 Start Here: Module Overview

Welcome to Week 12. This week focuses on **AI Safety, Alignment, and Interpretability** and moves your AI-powered research-assistant portfolio toward **Red-Team Review**. Keep the same domain and target user in view: the purpose is to make a connected design decision, not to collect an isolated technique.

Start with the Chapter 12 introduction and learning goals in Moody Amakobe’s primary textbook. Then read the focused Huyen selection listed in the guide; it adds engineering context to the primary chapter. Note the assumptions behind one claim and what evidence would make it useful for your assistant. The written module guide gives the full prompts and rubrics.

Your first assignment is the discussion, **Can We Know Whether a Model Is Aligned?**. Use the initial post to defend a decision, compare an alternative, and identify a meaningful limitation. Respond to two peers with analysis that extends their reasoning. The second assignment, **Constitutional AI + Structured Red-Team Exercise**, tests your judgment against the chapter case and a concrete exercise. The third assignment adds the author’s milestone to your portfolio.

Pay special attention to the difference between evidence and assumption. Preserve outputs or sources where needed, label projected behavior, and explain how your decision serves the user. Use the case to sharpen Red-Team Review rather than submitting a case summary.

Plan approximately 5–7 hours across the week. The guide’s day groups suggest pacing; Blackboard displays the actual due dates. Before submitting, review every rubric, check your files, and retain the editable work for later revision and the final portfolio.
