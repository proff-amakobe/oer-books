# Generative AI: 16-Week Course Syllabus and Teaching Outline

**Textbook:** Moody Amakobe, *Generative AI: Foundations, Systems, Evaluation, and Responsible Deployment*, First Open Edition, Global Data Science Institute, 2026.

**Duration:** 16 weeks. **Level:** Graduate. **Format:** Weekly reading, guided discussion, applied analysis, and a cumulative project portfolio. Plan approximately 6–8 hours of weekly work, consistent with the existing module guides; the institution determines credit hours and contact requirements.

**Scheduling fields to complete before distribution:** Course number, term and dates, meeting location or online platform, instructor and contact details, office hours, weekly deadlines and time zone, residency date and rubric, institutional grading scale, and required institutional policies.

## Course description

Study the foundations, architectures, capabilities, and limitations of generative AI and large language models. Progress from prompting and API integration through agents, retrieval, customization, research methods, multimodal systems, evaluation, and optimization. Examine safety, ethics, law, governance, and deployment as parts of system design. Throughout the semester, develop and defend the design of an AI-powered research assistant for a chosen domain.

The central project emphasizes design and evaluation. Students submit diagrams, prompts, analyses, and evidence-backed decisions; a working prototype is optional. The approved course outcomes retain their implementation language, while the required portfolio follows the textbook’s design-centered project arc. Instructors must identify any additional implementation evidence required by their program before the term begins.

## Preparation and materials

Students should be comfortable reading technical material, comparing evidence, and writing analytical reports. Basic programming, software architecture, and introductory machine learning knowledge are useful; production software development is not required for the portfolio.

Use the fifteen authoritative chapters in [original/](../original/), in order. The [existing course map](blackboard/GENERATIVE-AI-WEEKLY-COURSE-MAP.md) identifies focused companion readings from Chip Huyen’s *AI Engineering: Building Applications with Foundation Models*; its Week 14 and Week 16 selections are optional. The textbook and weekly module guides are the primary course materials. Students need a document editor, diagramming tool, and access to the course platform. The instructor should provide sample outputs or an approved model-access option for comparison exercises so required work does not depend on purchasing API credits.

## Course learning outcomes

By the end of the course, students will be able to demonstrate the following approved course objectives (COs):

**CO 1:** Analyze the theoretical foundations, architectures, and underlying mechanisms of Large Language Models and generative artificial intelligence systems.

Maps to PSLOs 1.

**CO 2:** Design and implement applications that leverage Large Language Models and generative AI techniques to solve complex computational problems.

Maps to PSLOs 2, 3.

**CO 3:** Evaluate the performance, capabilities, limitations, and reliability of Large Language Models using appropriate experimental methodologies and evaluation metrics.

Maps to PSLOs 3, 4.

**CO 4:** Assess ethical, societal, legal, and security considerations associated with the development and deployment of generative artificial intelligence systems.

Maps to PSLOs 5.

**CO 5:** Investigate and synthesize advanced methodologies, architectures, emerging technologies, and current research trends in Large Language Models and generative artificial intelligence.

Maps to PSLOs 1, 4, 6.

The full program student learning outcome (PSLO) statements appear in the linked course map. The approved alignment is CO 1 → PSLO 1; CO 2 → PSLOs 2, 3; CO 3 → PSLOs 3, 4; CO 4 → PSLO 5; CO 5 → PSLOs 1, 4, 6.

## Chapter-to-week schedule

Read the assigned chapter before that week’s discussion. Chapters 1–15 map directly to Weeks 1–15. Week 16 integrates all fifteen chapters and introduces no new chapter. Chapter 15’s metadata title is “Deployment and the Frontier”; its body heading is “Deployment, and the Road Ahead.” Both refer to the same assigned chapter.

| Week | Required textbook reading | Portfolio milestone | Course outcomes |
| --- | --- | --- | --- |
| 1 | [Chapter 1: Foundations of Generative AI](../original/01-foundations.qmd) | Project Charter | CO 1, 3, 4 |
| 2 | [Chapter 2: The Architecture of Understanding](../original/02-llms.qmd) | Model Selection Memo | CO 1, 3, 5 |
| 3 | [Chapter 3: The Art and Science of Prompting](../original/03-prompt-engineering.qmd) | Prompt Library v1 | CO 2, 3, 4 |
| 4 | [Chapter 4: Working with APIs and Integration](../original/04-apis-integration.qmd) | Integration Architecture Diagram | CO 2, 3 |
| 5 | [Chapter 5: Agents — AI That Takes Action](../original/05-agents.qmd) | Agent Capability Plan | CO 2, 3, 4 |
| 6 | [Chapter 6: Retrieval-Augmented Generation](../original/06-rag.qmd) | RAG Design Document | CO 2, 3, 5 |
| 7 | [Chapter 7: Fine-tuning and Customization](../original/07-finetuning.qmd) | Customization Decision Memo | CO 1, 2, 3, 5 |
| 8 | [Chapter 8: Research Methods and Literature Synthesis](../original/08-research-methods.qmd) | Literature Review & Project Proposal | CO 3, 5 |
| 9 | [Chapter 9: Multimodal Capabilities](../original/09-multimodal.qmd) | Multimodal Extension Plan | CO 1, 2, 3, 5 |
| 10 | [Chapter 10: Evaluating Generative AI Systems](../original/10-evaluation.qmd) | Evaluation Plan | CO 3, 5 |
| 11 | [Chapter 11: Advanced Techniques and Optimization](../original/11-optimization.qmd) | Optimization Plan | CO 2, 3, 5 |
| 12 | [Chapter 12: AI Safety, Alignment, and Interpretability](../original/12-safety-alignment.qmd) | Red-Team Review | CO 1, 3, 4, 5 |
| 13 | [Chapter 13: Safety, Ethics, and Responsible AI](../original/13-ethics.qmd) | Ethics & Fairness Review | CO 4, 5 |
| 14 | [Chapter 14: Law, Policy, and Governance of AI](../original/14-law-policy.qmd) | Compliance Checklist | CO 4, 5 |
| 15 | [Chapter 15: Deployment, and the Road Ahead](../original/15-deployment-frontier.qmd) | Deployment Plan & Forward Look | CO 2, 3, 4, 5 |
| 16 | Synthesis and review of Chapters 1–15; no new chapter | Final Integrated Portfolio & Demonstration | CO 1, 2, 3, 4, 5 |

## Weekly teaching outline and measurable outcomes

Each week follows a read–discuss–apply–revise sequence. Use a short conceptual lesson, a worked example or case discussion, an applied exercise, and portfolio feedback. The outcomes below retain the approved weekly objectives from the existing module guides. Assessment prompts, detailed rubrics, and student-facing instructions are available in the [Blackboard course package](blackboard/README.md).

### Week 1: Foundations of Generative AI

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze how generative and discriminative approaches address the same concrete problem.
2. Compare model families using capability, cost, context, and openness rather than one ranking.
3. Evaluate factual accuracy, uncertainty, and ambiguity handling in a controlled model comparison.
4. Assess who bears a specific training-data or hallucination risk in the chosen domain.
5. Justify the research-assistant charter using a simpler alternative and observable success criteria.

**Teaching activities:** Discuss “Introduce Yourself + Does This Problem Need Generative AI?” and facilitate Comparative Model Study + Bounded Automation Analysis.

**Assessment and portfolio evidence:** Project Charter. **Alignment:** CO 1, 3, 4.

### Week 2: The Architecture of Understanding

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze how transformer components affect the handling of a research query.
2. Compare pre-training, post-training, and inference as explanations for model behavior.
3. Evaluate capability, context, cost, and latency trade-offs using project requirements.
4. Critique evidence for domain specialization versus model scale.
5. Justify a model-selection recommendation with an explicit unresolved question.

**Teaching activities:** Discuss “When Is a More Capable Model Actually Worth It?” and facilitate Architecture, Model Size, and BloombergGPT Decision Analysis.

**Assessment and portfolio evidence:** Model Selection Memo. **Alignment:** CO 1, 3, 5.

### Week 3: The Art and Science of Prompting

**Learning outcomes:** By the end of the week, students will be able to:

1. Design task-specific prompts with explicit role, context, task, format, and constraints.
2. Compare zero-shot and few-shot prompt behavior with controlled inputs.
3. Develop reusable prompt templates for distinct research-assistant requests.
4. Evaluate prompt quality with stated criteria and an honest failure example.
5. Assess a prompt-injection risk and justify a concrete defensive boundary.

**Teaching activities:** Discuss “Is Better Prompting a Substitute for a Better Model?” and facilitate Prompt Redesign, Few-Shot Comparison, and Injection Test.

**Assessment and portfolio evidence:** Prompt Library v1. **Alignment:** CO 2, 3, 4.

### Week 4: Working with APIs and Integration

**Learning outcomes:** By the end of the week, students will be able to:

1. Design a provider integration flow that matches workload and reliability needs.
2. Compare retries, circuit breakers, and failover across distinct failure modes.
3. Evaluate caching choices against freshness and latency requirements.
4. Develop a cost and rate-limit plan with explicit assumptions.
5. Design monitoring and key/data-handling boundaries for the integration architecture.

**Teaching activities:** Discuss “Reliability, Cost, and Vendor Dependence” and facilitate Provider Outage Response and Integration Trade-off Analysis.

**Assessment and portfolio evidence:** Integration Architecture Diagram. **Alignment:** CO 2, 3.

### Week 5: Agents — AI That Takes Action

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze the change in risk when an assistant can act rather than only answer.
2. Compare agent architectures against a bounded research-assistant workflow.
3. Design a narrowly scoped tool specification and permission boundary.
4. Evaluate action quality and recovery using cost and reversibility.
5. Justify a human checkpoint for a concrete agent failure mode.

**Teaching activities:** Discuss “How Much Autonomy Should an AI Agent Have?” and facilitate Klarna Agent Case + Human-in-the-Loop Boundary Analysis.

**Assessment and portfolio evidence:** Agent Capability Plan. **Alignment:** CO 2, 3, 4.

### Week 6: Retrieval-Augmented Generation

**Learning outcomes:** By the end of the week, students will be able to:

1. Design a document-processing and chunking strategy for a specific knowledge source.
2. Compare semantic, keyword, and hybrid retrieval against domain requirements.
3. Evaluate retrieval and generation failures separately.
4. Critique research evidence for a chosen retrieval design.
5. Develop a worked retrieval-to-cited-answer example with explicit source boundaries.

**Teaching activities:** Discuss “When Does RAG Increase Trust — and When Doesn't It?” and facilitate Retrieval Failure Analysis / Glean Case.

**Assessment and portfolio evidence:** RAG Design Document. **Alignment:** CO 2, 3, 5.

### Week 7: Fine-tuning and Customization

**Learning outcomes:** By the end of the week, students will be able to:

1. Compare full fine-tuning and parameter-efficient approaches conceptually.
2. Design a customization decision using prompting, retrieval, and fine-tuning alternatives.
3. Evaluate a proposed dataset for quality, coverage, and sourcing limitations.
4. Critique evidence that customization improves the target task over a baseline.
5. Justify the maintenance and evaluation burden of the recommended approach.

**Teaching activities:** Discuss “Prompting, RAG, or Fine-Tuning?” and facilitate Harvey Case + Customization Decision Analysis.

**Assessment and portfolio evidence:** Customization Decision Memo. **Alignment:** CO 1, 2, 3, 5.

### Week 8: Research Methods and Literature Synthesis

**Learning outcomes:** By the end of the week, students will be able to:

1. Evaluate source credibility by publication type, methods, and evidence.
2. Analyze the comparability of benchmark claims and baselines.
3. Synthesize literature around themes and disagreements rather than separate summaries.
4. Critique uncertainty and evidence gaps relevant to a project decision.
5. Develop a literature-grounded revision of the project proposal.

**Teaching activities:** Discuss “What Makes an AI Claim Credible?” and facilitate Benchmark Claim Investigation + Literature Synthesis.

**Assessment and portfolio evidence:** Literature Review & Project Proposal. **Alignment:** CO 3, 5.

### Week 9: Multimodal Capabilities

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze how vision-language models relate visual and textual information.
2. Design a modality choice grounded in a specific user need.
3. Evaluate multimodal failure consequences and fallback effectiveness.
4. Critique evidence from a model system card or relevant research source.
5. Justify adding or rejecting a multimodal extension for the project.

**Teaching activities:** Discuss “When Is Multimodal AI Actually Necessary?” and facilitate Be My Eyes Case + Modality/Fallback Analysis.

**Assessment and portfolio evidence:** Multimodal Extension Plan. **Alignment:** CO 1, 2, 3, 5.

### Week 10: Evaluating Generative AI Systems

**Learning outcomes:** By the end of the week, students will be able to:

1. Design evaluation criteria tied to concrete user outcomes.
2. Compare automated, human, and model-judge methods and their limitations.
3. Develop a human-evaluation rubric and typical, edge, and adversarial cases.
4. Critique research evidence for an evaluation method.
5. Evaluate RAG or agent failures with component-specific checks.

**Teaching activities:** Discuss “What Does It Mean for a Generative AI System to Be Good?” and facilitate Evaluation Design + LLM-as-Judge / Anthropic Analysis.

**Assessment and portfolio evidence:** Evaluation Plan. **Alignment:** CO 3, 5.

### Week 11: Advanced Techniques and Optimization

**Learning outcomes:** By the end of the week, students will be able to:

1. Compare ensemble and compression approaches by benefit and complexity.
2. Design an experiment-driven optimization plan with a defensible baseline.
3. Evaluate quality, cost, and latency trade-offs under explicit load assumptions.
4. Critique technical evidence for a serving optimization.
5. Justify when advanced optimization is unnecessary for the project.

**Teaching activities:** Discuss “When Is Optimization Worth the Complexity?” and facilitate Serving-at-Scale Trade-off Analysis.

**Assessment and portfolio evidence:** Optimization Plan. **Alignment:** CO 2, 3, 5.

### Week 12: AI Safety, Alignment, and Interpretability

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze the alignment problem and the role of post-training.
2. Compare RLHF and Constitutional AI as behavioral-shaping approaches.
3. Evaluate the limits of behavioral tests and interpretability evidence.
4. Design a bounded red-team review addressing a domain-specific risk.
5. Critique scholarly evidence for an alignment approach.

**Teaching activities:** Discuss “Can We Know Whether a Model Is Aligned?” and facilitate Constitutional AI + Structured Red-Team Exercise.

**Assessment and portfolio evidence:** Red-Team Review. **Alignment:** CO 1, 3, 4, 5.

### Week 13: Safety, Ethics, and Responsible AI

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze responsibilities across individual, organizational, and industry levels.
2. Evaluate a domain-specific bias risk and competing mitigations.
3. Design a privacy and explainability approach for affected stakeholders.
4. Critique ethical frameworks using the chapter’s scholarly readings.
5. Justify a responsible deployment decision that may change or limit the system.

**Teaching activities:** Discuss “Who Bears Responsibility When an AI System Causes Harm?” and facilitate Microsoft Responsible AI Case + Stakeholder/Fairness Analysis.

**Assessment and portfolio evidence:** Ethics & Fairness Review. **Alignment:** CO 4, 5.

### Week 14: Law, Policy, and Governance of AI

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze legal, ethical, and governance questions as related but distinct.
2. Evaluate which regulatory frameworks plausibly apply to a defined deployment.
3. Critique official-source evidence and identify genuine regulatory uncertainty.
4. Design a governance response connecting copyright, liability, and verification.
5. Justify a compliance checklist with explicit limits and review needs.

**Teaching activities:** Discuss “Ethical Is Not the Same as Legal — Why Does That Matter?” and facilitate Legal Research Assistant + Compliance/Governance Analysis.

**Assessment and portfolio evidence:** Compliance Checklist. **Alignment:** CO 4, 5.

### Week 15: Deployment and the Frontier

**Learning outcomes:** By the end of the week, students will be able to:

1. Design a production architecture and staged deployment strategy.
2. Evaluate migration readiness using tests, monitoring, and rollback criteria.
3. Assess ethical, legal, and security responsibilities during system evolution.
4. Synthesize evidence about an emerging direction and its project implications.
5. Justify a lifecycle plan that extends beyond initial launch.

**Teaching activities:** Discuss “How Do You Design an AI System to Survive Change?” and facilitate Model Deprecation + Production Evolution Scenario.

**Assessment and portfolio evidence:** Deployment Plan & Forward Look. **Alignment:** CO 2, 3, 4, 5.

### Week 16: Final Project Demonstration, Portfolio Synthesis, and Reflection

**Learning outcomes:** By the end of the week, students will be able to:

1. Analyze how model architecture and behavior informed the final design.
2. Design a coherent system architecture that integrates the portfolio decisions.
3. Evaluate the design using tests, criteria, and explicitly bounded evidence.
4. Assess ethical, legal, security, and societal responsibilities across the system.
5. Synthesize research and project revisions into a defensible final presentation.

**Teaching activities:** Discuss “What Changed in Your Design — and Why?” and facilitate Integrated Project Portfolio.

**Assessment and portfolio evidence:** Final Integrated Portfolio & Demonstration. **Alignment:** CO 1, 2, 3, 4, 5.

## Semester project and assessment checkpoints

Select a domain and target user in Week 1 and retain that context across all milestones. Most weekly artifacts should be concise documents of approximately 1–3 pages or a diagram with explanatory text. Revise earlier decisions as new evidence emerges rather than assembling inconsistent, independent assignments.

- **Weeks 1–7:** Establish the problem, model choice, prompting approach, integration, autonomy boundaries, retrieval design, and customization decision.
- **Week 8:** Submit a literature review using 5–8 sources and a formal project proposal. Explain how the evidence supports or changes the design.
- **Weeks 9–15:** Address modality choices, evaluation, optimization, safety, fairness, governance, and deployment. A decision against a multimodal extension or fine-tuning is acceptable when justified.
- **Week 16:** Submit the integrated portfolio and a 10–15 minute presentation or slide-and-notes walkthrough. Demonstrate a coherent user journey with a live prototype or storyboard, explain revisions and trade-offs, and distinguish observed results from proposed tests. A live prototype and recording are not required.

For feedback, examine conceptual accuracy, evidence quality, justification of alternatives, consistency across the system, and recognition of limitations. Apply the existing weekly rubrics when assigning points. Do not award measured-performance claims without supporting observations or label a planned experiment as a completed evaluation.

## Grading

This syllabus preserves the existing approved 1,000-point model.

| Assessment | Calculation | Points | Course weight |
| --- | --- | --- | --- |
| Weeks 1–15 discussions | 15 × 10 | 150 | 15% |
| Weeks 1–15 applied analyses | 15 × 25 | 375 | 37.5% |
| Weeks 1–15 portfolio milestones | 15 × 15 | 225 | 22.5% |
| Week 16 reflection | 1 × 10 | 10 | 1% |
| Week 16 integrated portfolio | 1 × 20 | 20 | 2% |
| Week 16 demonstration | 1 × 20 | 20 | 2% |
| Separate residency assessment | Separate gradebook item | 200 | 20% |
| **Total** | **800 weekly points + 200 residency points** | **1,000** | **100%** |

Every week contributes 50 points. The Week 8 proposal is part of its weekly milestone allocation; the final portfolio and demonstration use the Week 16 allocation. Residency is separate from Week 16. Its activity, date, and rubric must be supplied by the instructor; this syllabus reserves the approved points without inventing those requirements. Publish the institutional letter-grade thresholds before the course begins.

## Participation, evidence, and responsible AI use

Complete the reading before discussion, support claims with traceable sources, and provide substantive peer feedback under the weekly prompts. Keep a revision history so the final portfolio shows how feedback changed the design.

Disclose generative AI assistance in submitted work, including the tool and model when available, its role, and how outputs were checked. Students remain responsible for accuracy, citations, and the decisions they defend. Do not submit fabricated sources or results. Use public, synthetic, or appropriately authorized data; classroom exercises should avoid uploading confidential or identifiable personal information to external tools. Maintain human approval boundaries for consequential actions in agent designs.

The Week 14 task is a jurisdiction- and date-specific research exercise: identify relevant primary legal or regulatory sources and record when they were checked. The instructor should refresh that material before teaching the module.

## Instructor preparation and course administration

Before release, populate the scheduling fields above; publish attendance, late-work, revision, academic integrity, accessibility, and accommodation policies from the institution; and supply the residency assessment. Set actual due dates in the learning management system. The weekly guides’ day groups are pacing suggestions rather than separate deadlines.

Use the [weekly course map](blackboard/GENERATIVE-AI-WEEKLY-COURSE-MAP.md) for full CO/PSLO statements and companion selections, and the [Blackboard package](blackboard/README.md) for sixteen module guides, student orientation text, assignment prompts, and rubrics. This outline adds a syllabus overview without changing those materials or the canonical manuscript.

