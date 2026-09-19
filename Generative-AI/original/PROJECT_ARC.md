# The Semester Project: An AI-Powered Research Assistant
## Project Arc Reference — Introduction to Generative AI with Large Language Models

This document is a production reference, not a chapter. It exists so every chapter's "Project Milestone" section stays consistent with the ones before and after it. Update it if the chapter sequence changes.

---

## Why one project, sixteen weeks

The book's philosophy is understanding over implementation, so the semester project is deliberately **design-and-evaluate**, not **build-and-ship**. Students are not expected to produce a deployed application by Week 16 (though ambitious students may). They are expected to produce a complete, defensible *design* for an AI-powered research assistant, with every major decision — model choice, prompting strategy, retrieval architecture, evaluation plan, safety review, deployment plan — justified in writing and grounded in the concepts from that week's chapter.

This keeps the project accessible to students without a strong engineering background while still meeting the course's PSLO on designing and evaluating software systems: the design artifacts *are* the deliverable, and a working prototype (built with no-code/low-code tools, notebooks, or a chosen framework) is optional extra credit, not a requirement.

Each week's milestone is a short document (1–3 pages) or diagram that gets added to a single, growing **Project Portfolio**. By Week 16, the portfolio itself is most of the demo.

## Project premise

Each student (or team) selects a real domain early in Week 1 and keeps it for the entire semester: legal research, medical literature review, customer support, academic tutoring, journalism, internal company knowledge search, and so on. Every weekly milestone is answered *for that domain*, which is what makes the same 16-week arc produce sixteen different, personal projects rather than one prescribed build.

## Milestone map

| Wk | Chapter | Milestone deliverable |
|---|---|---|
| 1 | Foundations of Generative AI | **Project Charter** — domain selection, target user, problem statement, and a one-paragraph justification for why a generative (not purely rule-based or retrieval-only) approach fits the problem |
| 2 | The Architecture of Understanding | **Model Selection Memo** — compare 2–3 candidate models on capability, context window, cost, and latency for this domain |
| 3 | The Art and Science of Prompting | **Prompt Library v1** — the assistant's system prompt plus 3–5 core task prompts, with a rationale for each design choice |
| 4 | Working with APIs and Integration | **Integration Architecture Diagram** — request flow, fallback strategy, and cost/rate-limit plan (diagram plus narrative, no implementation required) |
| 5 | Agents: AI That Takes Action | **Agent Capability Plan** — which actions the assistant may take autonomously, which require human approval, and why |
| 6 | Retrieval-Augmented Generation | **RAG Design Document** — knowledge source, chunking strategy, and a worked example of one retrieval-then-generate exchange |
| 7 | Fine-tuning and Customization | **Customization Decision Memo** — prompting vs. RAG vs. fine-tuning for this domain, with a data-sourcing plan if fine-tuning is chosen |
| 8 | Research Methods and Literature Synthesis | **Literature Review & Project Proposal** — 5–8 sources grounding the design, submitted as the formal midterm proposal |
| 9 | Multimodal Capabilities | **Multimodal Extension Plan** — could/should the assistant handle images, audio, or documents; if not, why not |
| 10 | Evaluating Generative AI Systems | **Evaluation Plan** — metrics, test cases, and a human-evaluation rubric for this specific assistant |
| 11 | Advanced Techniques and Optimization | **Optimization Plan** — cost and latency budget, caching strategy, model-routing decisions |
| 12 | AI Safety, Alignment, and Interpretability | **Red-Team Review** — an adversarial self-review of the design's failure modes and alignment risks |
| 13 | Safety, Ethics, and Responsible AI | **Ethics & Fairness Review** — bias risks, affected stakeholders, and mitigations specific to this domain |
| 14 | Law, Policy, and Governance of AI | **Compliance Checklist** — applicable regulations (e.g., HIPAA, GDPR, EU AI Act risk tier) for this domain and how the design addresses them |
| 15 | Deployment / The Frontier | **Deployment Plan & Forward Look** — how this would actually ship, plus one paragraph on how an emerging technique (from the Frontier chapter) could change the design in the next 12 months |
| 16 | — | **Final Demo** — a 10–12 minute presentation synthesizing the full portfolio, including a live or storyboarded walkthrough |

## Grading philosophy (for instructor adaptation)

Weekly milestones are checked for completion and thoughtfulness (low-stakes, formative). The Week 8 proposal and Week 16 demo are the two high-stakes checkpoints. This keeps grading load manageable across 16 weeks while still producing a portfolio substantial enough to support a real capstone-quality demo.

## Naming note

The project is referred to as "the AI-Powered Research Assistant" throughout the book for continuity with material already drafted in Chapters 4, 6, 7, 9, and 10. Individual students' domain choices (legal, medical, etc.) are instances of this same template project.
