# Phase 1B terminology consistency

The accepted Phase 1 scholarly review remains the baseline. This pass aligns reader-facing definitions, summaries, examples, and learning goals without restarting source verification.

| Term | Consistent usage |
|---|---|
| Generation / originality | Sampling from learned distributions can produce new combinations and can reproduce training material. Originality is not guaranteed. |
| Autoregressive LLM | Next-token generation describes this model class, not every generative or multimodal architecture. |
| Temperature | Adjusts sampling probabilities. Lower settings generally concentrate probability; higher settings generally increase diversity. Supported controls vary; neither accuracy nor creativity is guaranteed. |
| Context window | Request token budget with model/API-specific input, output, and combined limits. Capacity does not establish reliable use of every passage. |
| Few-shot prompting | In-context demonstrations, without weight updates. Relevance, order, diversity, count, and budget require task-specific evaluation. |
| Chain-of-thought | Generated intermediate text whose benefit and faithfulness require separate evaluation; inspectability is not proof of correctness or causal explanation. |
| Role framing / template | Context conditioning and reusable patterns. Neither hidden expert activation nor a proven universal solution. |
| Prompt-injection mitigation | Layered risk reduction; delimiters and instructions are not authorization boundaries. |
| Model size / tier | A resource characteristic or product positioning, not task quality, clinical validation, or a known proprietary parameter count. |
| Hallucination | Incorrect or evidence-unsupported generated content. Fluency and confidence are separate properties; citations must be checked. |
| Agent | A system whose model can request actions through tools controlled by the surrounding application. |
| RAG | Retrieval supplies evidence to generation. Evaluate retrieval, source quality, faithfulness, citation correctness, and abstention separately. |
| Fine-tuning | Additional training for adaptation. Compare prompting, retrieval, and adaptation; there is no mandatory escalation sequence. |
| LoRA / QLoRA | Low-rank updates, with a frozen quantized base for QLoRA. Trainable fraction, memory feasibility, and quality depend on configuration. |
| Multimodal capability | Task-, input-, and model-specific evidence; document, counting, transcription, and grounding tests differ. |
| Accuracy / quality | Accuracy requires a correctness criterion and denominator; quality names multiple task-relevant dimensions. |
| Groundedness / faithfulness | Support in supplied evidence / accurate reflection of that evidence; neither proves external truth. |
| Citation correctness / coverage | Whether a cited passage supports its claim / whether material claims have support. |
| Precision / recall | Relevant retrieved items divided by returned items / all judged relevant items, with cutoff and relevance unit specified. |
| Task success / human preference | Explicit completion criteria / raters’ choices; neither is interchangeable with factual correctness. |
| LLM-as-judge | An imperfect measurement instrument calibrated against independent human/domain judgments. |
| Alignment | Behavioral correspondence with specified intentions and values in Chapters 2 and 12. Cross-modal alignment in Chapter 9 instead relates representations across modalities. |
| Responsible AI | Applied fairness, privacy, accountability, and social responsibility in Chapter 13; technical safety in Chapter 12 and jurisdiction-specific law in Chapter 14 remain distinct. |

All fifteen terminology tables, introductions, objective lists, discussion questions, practice activities, project milestones, and wrap-ups were scanned for these relationships. Measurable Learning Objectives appear once per chapter. Instructions inside the 88 technical examples remain semantically unchanged. Strong language expressing a design requirement or a mathematical guarantee under stated assumptions is retained; unsupported outcome guarantees are revised in the consistency audit. Conversational “actually” remains where it clarifies a practical question rather than claiming unsupported certainty.

Figure markers use `[Figure pending: descriptive title]`. The manifest contains 30 existing placeholders (29 numbered proposals and the Marcus vignette) and 12 additional planned candidates, without duplicate visual proposals or generated art.
