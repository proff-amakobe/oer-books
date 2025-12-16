# STYLE GUIDE
## Introduction to Generative AI with Large Language Models

---

## Core Philosophy

This textbook teaches through **narrative, analogy, and conceptual depth**—not code tutorials. Students should finish each chapter understanding *why* things work, not just *how* to implement them.

**The guiding principle:** Explain concepts so clearly that readers could teach them to someone else without looking at the book.

---

## Voice and Tone

### Primary Voice: Friendly Expert
Imagine explaining AI to a bright colleague who's new to the field. You're knowledgeable but never condescending, enthusiastic but never hyperbolic.

### Tone Characteristics
- **Welcoming:** "Let's explore..." not "You must understand..."
- **Curious:** Model the intellectual excitement of the field
- **Honest:** Acknowledge limitations, unknowns, and debates
- **Grounded:** Connect abstract ideas to concrete experience
- **Balanced:** Present multiple perspectives on contested topics

### Example Voice Comparisons

**✅ Good (Friendly Expert):**
> "Here's something surprising about how language models work: they don't actually 'know' anything in the way humans know things. Instead, they've learned incredibly sophisticated patterns for predicting what text should come next. This simple idea—predict the next word—turns out to be remarkably powerful. Let's explore why."

**❌ Too Academic:**
> "Language models employ statistical inference mechanisms derived from extensive pre-training corpora to generate probabilistically coherent text sequences based on learned distributional semantics."

**❌ Too Casual:**
> "So LLMs are basically just really good at guessing the next word, which is kind of wild when you think about it!"

**❌ Too Promotional:**
> "These revolutionary AI systems represent the most transformative technological breakthrough in human history, fundamentally reshaping every aspect of how we work and live."

---

## Chapter Structure

Every chapter follows this arc:

### 1. Opening Narrative (300-500 words)
A story, scenario, or analogy that makes the chapter's central question tangible and urgent. The reader should think, "I want to understand this."

**Example opening styles:**
- A professional encountering a real problem that the chapter addresses
- A historical moment that changed everything
- A thought experiment that reveals a surprising truth
- A day-in-the-life scenario showing the technology in action

### 2. Learning Objectives
Clear, measurable outcomes. Use verbs like: explain, distinguish, evaluate, design, recognize, apply.

### 3. Main Content Sections
5-7 major sections, each building toward the chapter's goals. Each section should:
- Begin with a clear statement of what we're exploring
- Develop ideas progressively from simple to complex
- Include concrete examples and analogies
- Reference figures where visual explanation helps
- Conclude with a connection to the larger picture

### 4. Hands-On Exploration
Not heavy coding—focused experimentation and observation. Activities should:
- Reinforce conceptual understanding through direct experience
- Be completable in 30-60 minutes
- Require thought and analysis, not just execution
- Generate insights worth discussing

### 5. Case Study
A real company or project that illustrates the chapter's principles in practice.

### 6. Chapter Summary
Synthesis of key takeaways—what should stick.

### 7. Discussion Questions
Thought-provoking questions that:
- Have no single right answer
- Connect to broader implications
- Encourage debate and reflection
- Might appear on exams as essays

### 8. Further Reading
3-5 high-quality sources for deeper exploration, with brief annotations.

---

## Writing Techniques

### Lead with Narrative
Don't start sections with definitions. Start with questions, scenarios, or surprises that create a need for the concept.

**❌ Definition-first:**
> "Attention mechanisms are a component of neural networks that allow models to focus on relevant parts of the input when producing output."

**✅ Narrative-first:**
> "Imagine you're at a crowded party, trying to follow a conversation. Despite the noise, you can focus on one voice, tuning out the chaos around you. Your brain is performing a remarkable feat of selective attention. Transformer models face a similar challenge: given a long sequence of words, how do they know which words are relevant to each other? The answer lies in a mechanism that, fittingly, is called 'attention.'"

### Use Concrete Analogies
Abstract concepts need grounding. Find comparisons to everyday experience.

**Examples of effective analogies:**
- Context window → working memory or a desk's limited surface area
- Embeddings → GPS coordinates for meaning
- Fine-tuning → an experienced chef adapting to a new restaurant's style
- Temperature → the difference between following a recipe exactly vs. improvising
- RAG → an open-book exam vs. relying on memory

### Show Before You Tell
Where possible, let readers experience something before explaining it.

> "Try this: Ask the same question to three different AI models. Notice anything? The answers differ—sometimes subtly, sometimes dramatically. Why would that be? After all, they're all 'large language models.' The answer lies in the details of their training..."

### Acknowledge Complexity
Avoid false simplicity. When topics are genuinely complex or debated, say so.

> "This is one of the most contested questions in AI research. Some argue that scale alone will lead to general intelligence; others believe fundamental architectural innovations are needed. We'll explore both perspectives, but honest answer is: we don't know yet."

### Connect to Prior Knowledge
Explicitly reference earlier chapters and build on them.

> "Remember how we discussed the attention mechanism in Chapter 2? That same principle—allowing the model to focus on relevant context—is what makes RAG work. But instead of attending to other words in the prompt, the model attends to retrieved documents."

---

## Formatting Standards

### Headings
- **Chapter Title:** `# Chapter N: Title`
- **Major Sections:** `## 1.1 Section Title`
- **Subsections:** `### Subsection Title`
- **Minor Points:** `#### Minor heading` (use sparingly)

### Figure References
Reference figures in flowing prose, not as interruptions:

**✅ Integrated:**
> "The transformer processes information in parallel layers, as illustrated in Figure 2.3. Each layer refines the representation, building increasingly abstract understanding."

**❌ Interruptive:**
> "See Figure 2.3 below for how transformers work."

### Callout Boxes

Use sparingly for emphasis:

> **💡 Key Insight**  
> One-sentence distillation of a crucial concept.

> **⚠️ Common Misconception**  
> Correct a frequent misunderstanding.

> **🔍 Going Deeper**  
> Pointer to additional depth for interested readers.

> **🌍 Real-World Connection**  
> Brief example of concept in practice.

### Code (When Necessary)
Code should illustrate concepts, not be the focus. Keep examples:
- Short (under 20 lines ideally)
- Heavily commented to explain the "why"
- Conceptually focused, not production-ready
- Preceded and followed by prose explanation

### Lists
Use lists sparingly. Prose is usually better for explanation. When lists are appropriate:
- Use for enumeration (steps in a process, items in a category)
- Keep items parallel in structure
- Limit to 5-7 items when possible

---

## Language Conventions

### Technical Terms
- **First use:** Bold and define in context
- **Subsequent uses:** Regular text
- **Acronyms:** Spell out on first use with acronym in parentheses

### Numbers
- Spell out one through nine
- Use numerals for 10+
- Always use numerals with units: "4 GB", "128K tokens"

### Model and Company Names
- Use official capitalization: OpenAI, Anthropic, GPT-4, Claude
- Use code formatting for API model strings: `gpt-4o-mini`

### Hedging
Be appropriately confident. Avoid excessive hedging ("might possibly potentially") but don't overclaim.

**✅ Appropriate:**
> "This approach typically improves performance on reasoning tasks."

**❌ Over-hedged:**
> "This approach might potentially improve performance in some cases, though results may vary."

**❌ Over-claimed:**
> "This approach always dramatically improves performance."

---

## Ethical Integration

Don't relegate ethics to a single chapter. Weave ethical considerations throughout:

- When discussing capabilities, note potential misuse
- When discussing training, acknowledge labor and data issues
- When discussing deployment, consider who benefits and who might be harmed
- Present multiple perspectives on contested issues
- Avoid both techno-utopianism and doom-saying

---

## Figures

Each chapter should include 8-12 figures. Figures should:

### Serve Clear Purposes
- **Conceptual diagrams:** Illustrate relationships and architectures
- **Process flows:** Show how systems work step-by-step
- **Comparisons:** Side-by-side visualization of alternatives
- **Examples:** Concrete instances of abstract concepts
- **Data visualizations:** When quantitative relationships matter

### Design Principles
- Clean, uncluttered design
- Consistent visual language across chapters
- Accessible colors (consider colorblindness)
- Clear labels and legends
- Alt-text descriptions for accessibility

### Standard Figure Types
- Architecture diagrams (Chapter 2, 5, 8)
- Timeline/evolution diagrams (Chapter 1)
- Decision trees/flowcharts (Chapters 3, 4, 6)
- Comparison matrices (Chapters 2, 4, 7)
- Process pipelines (Chapters 5, 6)
- Conceptual illustrations (throughout)

---

## Review Checklist

Before finalizing any chapter:

- [ ] Opens with compelling narrative that creates curiosity
- [ ] Learning objectives are clear and measurable
- [ ] Concepts build progressively from simple to complex
- [ ] Analogies ground abstract ideas in concrete experience
- [ ] Figures are referenced in context, not as afterthoughts
- [ ] Ethical implications are woven throughout, not siloed
- [ ] Code (if any) is minimal and conceptually focused
- [ ] Hands-on section promotes understanding, not just doing
- [ ] Case study connects to real-world practice
- [ ] Discussion questions provoke genuine thought
- [ ] Further reading provides quality next steps
- [ ] Tone remains consistently warm and intellectually engaging
- [ ] Word count meets target (7,000-8,000 words minimum)
