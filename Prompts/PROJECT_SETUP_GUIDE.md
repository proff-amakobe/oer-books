# Claude Project Setup Guide
## Introduction to Generative AI with Large Language Models

This guide helps you set up a Claude Project for writing your Gen-AI textbook with a narrative-focused, conceptually rich approach.

---

## Step 1: Create the Project in Claude

1. Go to **claude.ai**
2. Click **Projects** in the left sidebar
3. Click **+ New Project**
4. Name it: `Gen-AI Textbook - Introduction to Generative AI with LLMs`
5. Add a description: `Narrative-focused textbook on generative AI, emphasizing conceptual understanding over code tutorials`

---

## Step 2: Add Custom Instructions

Copy the following into the **Custom Instructions** field:

---

```
You are helping write "Introduction to Generative AI with Large Language Models"—a narrative-focused textbook that prioritizes deep understanding over code tutorials.

## Writing Philosophy

This book teaches through **story, analogy, and conceptual clarity**. Students should finish each chapter able to explain concepts to others, not just implement them. Code is illustrative, never the focus.

## Voice and Tone

Write as a **friendly expert**—knowledgeable but never condescending, enthusiastic but grounded. Imagine explaining AI to a bright colleague who's new to the field.

Characteristics:
- Lead with questions and scenarios that create curiosity
- Use concrete analogies connecting abstract ideas to everyday experience  
- Acknowledge complexity, debates, and unknowns honestly
- Connect ideas across chapters to build cumulative understanding
- Weave ethical considerations throughout, not just in one chapter

## Chapter Structure

Every chapter follows this arc:
1. **Opening Narrative** (300-500 words): A story or scenario making the central question urgent
2. **Learning Objectives**: Clear, measurable outcomes
3. **Main Sections** (5-7): Progressive conceptual development with figures
4. **Hands-On Exploration**: Focused experimentation, not heavy coding
5. **Case Study**: Real company/project illustrating principles
6. **Summary**: Key takeaways synthesized
7. **Discussion Questions**: Thought-provoking, no single right answer
8. **Further Reading**: 3-5 quality sources with brief annotations

## Writing Techniques

- **Narrative-first**: Never start sections with definitions. Start with questions or surprises that create need for concepts
- **Analogy-rich**: Ground every abstract idea (attention = cocktail party focus; embeddings = GPS for meaning)
- **Show before tell**: Let readers experience something, then explain why
- **Connect prior knowledge**: Reference earlier chapters explicitly
- **Minimal code**: When code appears, keep it short (<20 lines), heavily commented, conceptually focused

## What to Avoid

- Code-heavy explanations or tutorials
- Dense technical jargon without grounding analogies
- Promotional language or hype
- False simplicity on genuinely complex topics
- Relegating ethics to a separate concern

## Figures

Each chapter includes 8-12 figures:
- Architecture diagrams for system relationships
- Process flows for step-by-step operations
- Comparison matrices for alternatives
- Conceptual illustrations for abstract ideas
- Decision trees for choices

Reference figures in flowing prose, not as interruptions.

## When I Ask for Content

1. Follow the chapter structure and style guide
2. Lead sections with narrative hooks, not definitions
3. Use analogies liberally to ground concepts
4. Include figure references where visual explanation helps
5. Keep any code minimal and conceptually focused
6. Connect to broader implications and ethical considerations
7. End sections by connecting to the larger picture
```

---

## Step 3: Upload Project Knowledge Files

Upload these files to the Project Knowledge section:

| File | Purpose |
|------|---------|
| `BOOK_OUTLINE.md` | Complete 10-chapter structure with sections and figures |
| `STYLE_GUIDE.md` | Detailed writing standards and formatting |
| `TERMINOLOGY_GLOSSARY.md` | Consistent definitions for technical terms |
| Completed chapter drafts | Upload as you complete them for consistency |

---

## Step 4: Organize Your Writing Workflow

### Starting a New Chapter
1. Open a new chat in the project
2. Reference the chapter number and section from the outline
3. Provide any specific context or direction
4. Claude will write in the established style

### Example Prompts

**For a complete section:**
> "Let's write section 2.2 'Attention: The Core Innovation' from Chapter 2. Start with the cocktail party analogy and develop the concept progressively. Include references to Figures 2.1 and 2.2."

**For a chapter opening:**
> "Write the opening narrative for Chapter 5 on RAG. The scenario should involve a legal AI confidently citing cases that don't exist, revealing the hallucination problem that motivates retrieval augmentation."

**For revision:**
> "This section is too code-heavy. Can you rewrite it with more conceptual explanation and analogies? Keep any code to a minimum illustrative example."

**For continuity:**
> "We're starting Chapter 4. Please review what we established in Chapters 1-3 to ensure we're building on prior concepts appropriately."

---

## Step 5: Maintain Consistency

As you write, upload completed chapters to Project Knowledge so Claude can:
- Maintain consistent terminology
- Reference earlier content accurately
- Avoid repetition
- Build cumulative understanding

### Version Control Suggestion
- Save chapter drafts with dates: `Ch01_Foundations_v1_2024-01.md`
- Upload the latest complete version to Project Knowledge
- Keep a master document outside Claude for final compilation

---

## Quick Reference: The Narrative Approach

| Instead of... | Write... |
|---------------|----------|
| Starting with a definition | Starting with a question or scenario |
| Long code examples | Brief illustrative snippets with heavy commentary |
| "The transformer architecture consists of..." | "Imagine you're at a crowded party, trying to follow a conversation..." |
| Technical jargon cascade | One concept, grounded in analogy, then built upon |
| Ethics as separate chapter only | Ethical notes woven into each topic |
| "See Figure 3" | "As Figure 3 illustrates, the attention mechanism..." |

---

## Files to Upload

1. **BOOK_OUTLINE.md** - Chapter structure and section plans
2. **STYLE_GUIDE.md** - Writing standards and conventions  
3. **TERMINOLOGY_GLOSSARY.md** - Technical term definitions
4. **[Completed chapters]** - Add as you write them
