# Week 4 — Blackboard Assignment Copy

## Discussion Board

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


## Assignment 2

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


## Project Milestone

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


## Start Here Page

# Week 4 Start Here: Module Overview

Welcome to Week 4. This week focuses on **Working with APIs and Integration** and moves your AI-powered research-assistant portfolio toward **Integration Architecture Diagram**. Keep the same domain and target user in view: the purpose is to make a connected design decision, not to collect an isolated technique.

Start with the Chapter 4 introduction and learning goals in Moody Amakobe’s primary textbook. Then read the focused Huyen selection listed in the guide; it adds engineering context to the primary chapter. Note the assumptions behind one claim and what evidence would make it useful for your assistant. The written module guide gives the full prompts and rubrics.

Your first assignment is the discussion, **Reliability, Cost, and Vendor Dependence**. Use the initial post to defend a decision, compare an alternative, and identify a meaningful limitation. Respond to two peers with analysis that extends their reasoning. The second assignment, **Provider Outage Response and Integration Trade-off Analysis**, tests your judgment against the chapter case and a concrete exercise. The third assignment adds the author’s milestone to your portfolio.

Pay special attention to the difference between evidence and assumption. Preserve outputs or sources where needed, label projected behavior, and explain how your decision serves the user. Use the case to sharpen Integration Architecture Diagram rather than submitting a case summary.

Plan approximately 5–7 hours across the week. The guide’s day groups suggest pacing; Blackboard displays the actual due dates. Before submitting, review every rubric, check your files, and retain the editable work for later revision and the final portfolio.
