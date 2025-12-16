# TERMINOLOGY GLOSSARY
## Introduction to Generative AI with Large Language Models

Use these exact terms and definitions throughout the book for consistency.

---

## Core Concepts

### Generative AI (GenAI)
A subset of artificial intelligence focused on creating new, original content—such as text, images, music, or code—rather than merely analyzing or classifying existing data.

### Large Language Model (LLM)
A neural network trained on massive text datasets that can understand and generate human-like text. Examples include GPT-4, Claude, and Llama.

### Transformer
The neural network architecture underlying modern LLMs, introduced in the 2017 paper "Attention Is All You Need." Characterized by self-attention mechanisms.

### Token
The basic unit of text processing in LLMs. Can be a word, part of a word, or punctuation. Approximately 4 characters = 1 token in English.

### Context Window
The maximum number of tokens an LLM can process in a single interaction, including both input and output. Also called "context length."

### Embedding
A dense vector representation of text (or other data) that captures semantic meaning. Similar texts have similar embeddings.

---

## Model Operations

### Inference
The process of running a trained model to generate predictions or outputs. What happens when you send a prompt to an LLM.

### Fine-tuning
Continuing the training of a pre-trained model on a specific dataset to specialize it for particular tasks or domains.

### Pre-training
The initial training phase where a model learns from large, general datasets before being specialized.

### RLHF (Reinforcement Learning from Human Feedback)
A technique for aligning LLM behavior with human preferences using reward models trained on human rankings.

### LoRA (Low-Rank Adaptation)
An efficient fine-tuning technique that trains only small adapter layers instead of the full model weights.

### Quantization
Reducing model precision (e.g., from 32-bit to 8-bit) to decrease memory usage and increase inference speed.

---

## Prompting

### Prompt
The input text sent to an LLM to elicit a response. Includes system prompts, user messages, and context.

### System Prompt
Instructions that define the model's behavior, role, or constraints. Set before the conversation begins.

### Zero-shot Prompting
Asking the model to perform a task without providing examples in the prompt.

### Few-shot Prompting
Including several examples in the prompt to guide the model's response pattern.

### Chain-of-Thought (CoT)
A prompting technique that encourages the model to show its reasoning step-by-step before giving a final answer.

### Prompt Engineering
The practice of designing and optimizing prompts to achieve desired model outputs.

### Prompt Injection
A security vulnerability where malicious input attempts to override system instructions or manipulate model behavior.

---

## RAG (Retrieval-Augmented Generation)

### RAG
A technique that enhances LLM responses by first retrieving relevant information from a knowledge base, then using that context in generation.

### Vector Database
A database optimized for storing and querying embedding vectors. Examples: ChromaDB, Pinecone, Weaviate.

### Semantic Search
Finding information based on meaning rather than keyword matching, typically using embedding similarity.

### Chunking
Breaking documents into smaller pieces for embedding and retrieval. Strategy significantly impacts RAG quality.

### Reranking
A second-stage retrieval step that reorders initial results based on relevance to the query.

---

## Agent Concepts

### AI Agent
An autonomous system that uses an LLM to reason about tasks and take actions to achieve goals.

### Function Calling / Tool Use
The ability of an LLM to invoke external functions or tools based on user requests.

### ReAct (Reasoning + Acting)
An agent architecture that alternates between reasoning about what to do and taking actions.

### Orchestration
Coordinating multiple AI components, tools, or agents to complete complex tasks.

---

## Technical Terms

### API (Application Programming Interface)
A set of protocols allowing different software applications to communicate. LLM providers expose APIs for model access.

### Endpoint
A specific URL where an API can be accessed. Example: `https://api.openai.com/v1/chat/completions`

### Rate Limiting
Restrictions on how many API requests can be made in a given time period.

### Streaming
Receiving model output token-by-token as it's generated, rather than waiting for the complete response.

### Latency
The time delay between sending a request and receiving the first response token.

### Throughput
The number of requests or tokens a system can process per unit of time.

---

## Model Families (Use Official Names)

| Family | Provider | Example Models |
|--------|----------|----------------|
| GPT | OpenAI | GPT-4, GPT-4o, GPT-4o-mini |
| Claude | Anthropic | Claude 3.5 Sonnet, Claude 3 Opus |
| Llama | Meta | Llama 3, Llama 3.1, Llama 3.2 |
| Gemini | Google | Gemini Pro, Gemini Ultra |
| Mistral | Mistral AI | Mistral 7B, Mixtral 8x7B |

---

## Abbreviations Reference

| Abbreviation | Full Term |
|--------------|-----------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| CoT | Chain-of-Thought |
| CPU | Central Processing Unit |
| GenAI | Generative AI |
| GPU | Graphics Processing Unit |
| JSON | JavaScript Object Notation |
| LLM | Large Language Model |
| LoRA | Low-Rank Adaptation |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| OCR | Optical Character Recognition |
| RAG | Retrieval-Augmented Generation |
| RLHF | Reinforcement Learning from Human Feedback |
| SDK | Software Development Kit |
| TTS | Text-to-Speech |
| STT | Speech-to-Text |

---

## Style Notes

- Always capitalize "Large Language Model" on first use, then can use "LLM"
- Use "GenAI" only after defining "Generative AI" 
- Refer to "the model" not "the AI" when discussing technical operations
- Use "generate" for text creation, "create" for images
- "Prompt" is a noun; "prompting" is the verb form
