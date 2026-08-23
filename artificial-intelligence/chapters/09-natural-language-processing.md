# Chapter 9: Machines That Read

**Natural Language Processing and the Transformer Revolution**

*CSC5350 · Artificial Intelligence*

---

## Opening Narrative

### Eight Researchers, One Paper, and an Architecture That Changed Everything

In the summer of 2017, a team of eight researchers at Google Brain submitted a paper to the Neural Information Processing Systems conference. The title was deliberately provocative: "Attention Is All You Need." The claim was equally bold: the recurrent architectures that had dominated natural language processing for the previous decade were not merely inferior to what they proposed — they were unnecessary. Everything that mattered could be accomplished with a single mechanism, scaled up: attention.

The architecture they described — the **transformer** — would, within three years, make every other approach to language modeling obsolete.

The transformer's core insight is deceptively simple. When a human reads "The animal didn't cross the street because *it* was too tired," understanding what "it" refers to requires attending to the earlier word "animal." The reader does not process words sequentially in strict order — they attend to the parts of the sentence relevant to resolving each ambiguity. The transformer formalizes this attending: for every word, compute a weighted combination of all other words' representations, where the weights reflect relevance.

This sounds simple. What it enabled was not.

Within two years, Google released BERT — a transformer trained to fill in masked words by reading both before and after them. BERT shattered eleven language understanding benchmarks on the day of its release. Within three years, OpenAI released GPT-3 — a transformer trained to predict the next word, scaled to 175 billion parameters. GPT-3 could write essays, debug code, translate between sixty languages, and answer questions with apparent comprehension, all from a single model trained on a single objective.

Neither capability was a direct consequence of the attention mechanism alone. What the transformer provided was an architecture that *scaled*. More data, more parameters, more compute — and the model improved, in a smooth curve with no evident ceiling. The scaling laws that AlexNet had revealed for vision, the transformer revealed for language.

> **"A transformer does not understand language the way a human does. It has found a different relationship with language — one that is neither worse nor better, but genuinely different. Whether that difference matters depends on what you are trying to do."**

---

## Learning Objectives

After completing this chapter, you will be able to:

1. Describe the NLP pipeline from raw text to model-ready features, including tokenization, normalization, and subword vocabulary construction.
2. Explain word embeddings and describe how Word2Vec captures semantic relationships geometrically.
3. Derive the self-attention mechanism and explain how it computes contextual representations.
4. Describe the complete transformer architecture: positional encoding, multi-head attention, feed-forward sublayers, encoder, and decoder.
5. Distinguish BERT (encoder-only) from GPT (decoder-only) and explain which is appropriate for which tasks.
6. Use the Hugging Face Transformers library to apply pre-trained models for sentiment analysis, NER, and question answering.
7. Use spaCy for practical NLP preprocessing: tokenization, POS tagging, dependency parsing, and named entity recognition.
8. Reason about the ethical implications of large language models: hallucination, bias, intellectual property, and societal impact.
9. Build the IAAIS Language Module — a natural language interface that classifies user intent and routes queries to appropriate modules.

---

## Key Terminology

| Term | Plain-Language Definition |
|---|---|
| **Token** | The basic unit a language model processes. May be a word, subword, or character. "unbelievable" might become ["un", "##believ", "##able"] under subword tokenization. |
| **Tokenization** | Splitting raw text into tokens. Modern NLP uses subword tokenization (BPE, WordPiece) to handle rare words without requiring a fixed vocabulary of whole words. |
| **Vocabulary** | The complete set of tokens a model knows. Subword tokenization ensures any word can be represented as a sequence of known pieces. |
| **Bag of Words (BoW)** | A text representation counting word occurrences in a document, discarding order. Simple but effective for topic classification and spam filtering. |
| **TF-IDF** | Term Frequency–Inverse Document Frequency. Upweights words common in a document but rare across the corpus — the words that distinguish this document from others. |
| **Word Embedding** | A dense vector representing a word in a continuous semantic space. Similar words have similar vectors. Learned by training on co-occurrence in large corpora. |
| **Word2Vec** | A model learning word embeddings by predicting surrounding words from a center word (skip-gram) or vice versa (CBOW). Captures analogy relationships: king - man + woman ≈ queen. |
| **Contextual Embedding** | A word embedding that depends on context — the same word in different sentences gets different representations. BERT and GPT produce contextual embeddings; Word2Vec produces static ones. |
| **Attention** | A mechanism computing a weighted combination of value vectors, where weights reflect query-key similarity. Allows the model to focus on relevant parts of the input. |
| **Self-Attention** | Attention applied within a single sequence — each position attends to all other positions. The fundamental operation of the transformer. |
| **Query, Key, Value (Q, K, V)** | Three projections used in attention. The query represents "what am I looking for?"; the key represents "what do I contain?"; the value represents "what do I contribute?" |
| **Multi-Head Attention** | Running attention in parallel with multiple learned projections (heads), then concatenating results. Each head can specialize in different relationship types. |
| **Positional Encoding** | A vector added to each token's embedding encoding its position in the sequence. Necessary because self-attention is permutation-invariant without it. |
| **Encoder** | Transformer component reading a full input sequence and producing rich contextual representations. BERT is encoder-only. |
| **Decoder** | Transformer component generating output token by token, attending to prior tokens and (in encoder-decoder models) the encoder's output. GPT is decoder-only. |
| **BERT** | Bidirectional Encoder Representations from Transformers. Pre-trained by predicting masked tokens from both directions simultaneously. Fine-tuned for classification, NER, and question answering. |
| **GPT** | Generative Pre-trained Transformer. Pre-trained to predict the next token (causal/left-to-right). Excels at text generation, summarization, and in-context learning. |
| **Fine-Tuning** | Adapting a pre-trained model to a specific task by continuing training on task-specific labeled data with a small learning rate. |
| **Named Entity Recognition (NER)** | Identifying and classifying named entities in text — people, organizations, locations, dates, medical terms — into predefined categories. |
| **Hallucination** | Confident generation of factually incorrect, fabricated, or internally inconsistent content by a language model. A structural feature of autoregressive models. |

---

## Section 1 — Why Language Is Hard

Language is extraordinarily complex. "The bank was steep" and "The bank was closed" share a surface form but mean entirely different things. "He told John that he had won" is ambiguous in a way that depends on context no rule can fully capture. "I can't recommend this restaurant too highly" means either that the recommendation is strong or that it is undeserved, depending on intonation that text does not encode.

These ambiguities are not edge cases — they are the norm. Language is designed for efficiency, and efficiency requires exploiting shared context, common knowledge, and real-time pragmatic inference to resolve ambiguity.

The NLP pipeline addresses this through a hierarchy of tasks:

| Level | Task | Example |
|---|---|---|
| Lexical | Tokenization, POS tagging | "running" → verb, present participle |
| Syntactic | Dependency parsing | Subject-verb-object structure |
| Semantic | NER, word sense disambiguation | "Apple" = company vs. fruit |
| Discourse | Coreference resolution | "he" refers to which antecedent? |
| Pragmatic | Intent detection, sentiment | Sarcasm, implied meaning |

Modern transformer-based models address many of these simultaneously, learning the lower levels implicitly as part of solving higher-level objectives.

### Text Preprocessing

Before any model can process text, raw characters must become structured numbers. Modern NLP pipelines follow a standard sequence:

**Cleaning:** Lowercase, remove or normalize URLs, punctuation, and special characters, collapse whitespace.

**Subword tokenization:** The standard choice for production systems. Byte Pair Encoding (BPE, used by GPT) and WordPiece (used by BERT) iteratively merge frequent character pairs until a target vocabulary size is reached. The key advantage: any word can be represented as a sequence of known subwords. "COVID-19" becomes ["CO", "##VI", "##D", "-", "19"]; novel words are handled gracefully.

**Normalization:** Stemming (rule-based suffix stripping) or lemmatization (dictionary-based reduction to base form) reduces morphological variation. "running," "ran," and "runs" all map to "run."

---

## Section 2 — Word Embeddings: Meaning as Geometry

The foundational insight of word embeddings comes from John Rupert Firth (1957): "You shall know a word by the company it keeps." Words appearing in similar contexts have similar meanings. A model trained to predict context from words will learn to represent semantically similar words with similar vectors.

**Word2Vec** trains a simple neural network on one of two objectives: predict a center word from its surrounding context (CBOW), or predict surrounding words from a center word (skip-gram). The trained vectors for similar words cluster together in the embedding space, and the geometry encodes semantic relationships.

The remarkable result: vector arithmetic reflects semantic relationships. The vector for "king" minus the vector for "man" plus the vector for "woman" points toward the vector for "queen" — not because anyone programmed this, but because these relationships are encoded in the co-occurrence statistics of natural language.

Word2Vec's limitation: each word has one vector regardless of context. "Bank" gets the same representation in "river bank" and "investment bank." BERT and GPT produce **contextual embeddings** — different representations for the same word in different sentences, because the meaning changes with context.

---

## Section 3 — The Attention Mechanism: Focusing on What Matters

Self-attention is the operation that enables contextual representation: for each token, compute a weighted combination of all tokens' representations, where the weights reflect relevance.

The computation uses three learned projections for each token: a **query** Q ("what am I looking for?"), a **key** K ("what do I contain?"), and a **value** V ("what do I contribute?").

The attention weight between token i as query and token j as key:
**a(i,j) = softmax(QᵢKⱼᵀ / √d_k)**

The output for token i:
**Output_i = Σⱼ a(i,j) × Vⱼ**

The scaling by √d_k prevents dot products from growing too large in high dimensions, which would push the softmax into regions of near-zero gradient.

For the sentence "The animal didn't cross the street because it was too tired":
- When computing the representation for "it," the attention weights should be high for "animal" and low for "street" — because "animal" is what "it" refers to in this context
- These weights are learned from data, not programmed
- Different attention heads can specialize: one head resolves pronoun references; another tracks subject-verb agreement; a third identifies named entities

**Multi-head attention** runs this computation in parallel with h different learned projections, then concatenates and linearly projects the results. Each head learns to attend to different relationship types, providing complementary information.

---

## Section 4 — The Transformer Architecture

A complete transformer encoder layer has two sublayers:

**Multi-head self-attention sublayer:**
Output = LayerNorm(x + MultiHeadAttention(x, x, x))

**Feed-forward sublayer:**
Output = LayerNorm(x + FFN(x))

where FFN is a two-layer MLP applied independently at each position.

The **residual connections** (the "+ x" terms) are critical: they allow gradients to flow directly from any layer's output back to any earlier layer, enabling stable training of deep transformer stacks. Without them, 12-layer BERT would not train reliably.

**Positional encoding** is added to token embeddings before the first layer. Self-attention is permutation-invariant — it produces the same output regardless of token order. Without positional information, the model cannot distinguish "The cat chased the dog" from "The dog chased the cat." Fixed sinusoidal or learned positional embeddings encode each position, allowing the model to use position as part of its representations.

The complete architecture stacks multiple such layers (12 in BERT-base, 96 in GPT-3) with increasing layers building increasingly abstract representations.

---

## Section 5 — BERT and GPT: Two Philosophies

The transformer architecture admits two primary configurations, each suited to different tasks.

**BERT (Bidirectional Encoder Representations from Transformers)** is an encoder-only transformer pre-trained on two objectives: Masked Language Modeling (predict randomly masked tokens from both directions simultaneously) and Next Sentence Prediction (predict whether two sentences are consecutive in text). Because BERT sees both left and right context simultaneously, it develops genuinely bidirectional representations — "bank" in "river bank" gets a very different representation from "bank" in "investment bank."

BERT is fine-tuned for tasks requiring understanding of a fixed input: classification, NER, question answering, semantic similarity. It cannot generate text autoregressively — it processes the entire input at once.

**GPT (Generative Pre-trained Transformer)** is a decoder-only transformer pre-trained to predict the next token, using a causal attention mask that prevents each token from attending to future tokens. This autoregressive objective aligns perfectly with text generation: at inference time, GPT generates one token at a time, each conditioned on all previous tokens.

GPT fine-tunes naturally for generation, summarization, translation, and in-context learning. It is less suited for tasks requiring bidirectional understanding.

| Property | BERT | GPT |
|---|---|---|
| Attention | Bidirectional (full) | Causal (left-to-right) |
| Pre-training | Masked LM + NSP | Next-token prediction |
| Output | Contextual embeddings | Generated tokens |
| Best for | Classification, NER, QA | Generation, summarization |
| Cannot do | Long text generation | Bidirectional understanding |

---

## Section 6 — Hugging Face and spaCy: NLP in Practice

The Hugging Face Transformers library has democratized access to state-of-the-art NLP. Tasks that once required months of training are now three lines of Python.

```python
from transformers import pipeline

# Sentiment analysis — fine-tuned BERT
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier([
    "The drug trial results are extremely promising.",
    "Side effects were severe and the study was halted.",
    "The procedure was completed without complications.",
])
# Expected:
# [{'label': 'POSITIVE', 'score': 0.9991},
#  {'label': 'NEGATIVE', 'score': 0.9882},
#  {'label': 'POSITIVE', 'score': 0.8923}]

# Named Entity Recognition — clinical note
ner = pipeline("ner", aggregation_strategy="simple")
clinical_note = ("Dr. Chen at Mass General identified that Patient John Smith "
                 "presented with elevated troponin on March 15, 2024.")
entities = ner(clinical_note)
# Expected:
# [{'entity_group': 'PER', 'word': 'Chen', 'score': 0.9834},
#  {'entity_group': 'ORG', 'word': 'Mass General', 'score': 0.9712},
#  {'entity_group': 'PER', 'word': 'John Smith', 'score': 0.9901},
#  {'entity_group': 'DAT', 'word': 'March 15, 2024', 'score': 0.9756}]

# Zero-shot classification — no task-specific training needed
classifier_zsl = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")
result = classifier_zsl(
    "Patient has chest pain radiating to left arm and diaphoresis.",
    candidate_labels=["urgent emergency", "routine follow-up",
                      "medication adjustment", "diagnostic workup"]
)
# Expected:
# {'labels': ['urgent emergency', 'diagnostic workup', ...],
#  'scores': [0.8234, 0.1123, ...]}
```

**spaCy** provides industrial-strength, production-ready NLP for preprocessing:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Dr. Martinez prescribed amoxicillin 500mg to Alice at Boston Children's.")
for ent in doc.ents:
    print(f"[{ent.label_}] {ent.text}")

# Expected:
# [PERSON] Dr. Martinez
# [PERSON] Alice
# [ORG] Boston Children's
# (Medical entities like drugs require a medical model: en_core_sci_sm)

# Token-level analysis
for token in doc[:5]:
    print(f"{token.text:<15} {token.pos_:<8} {token.dep_:<12} {token.lemma_}")
# Expected:
# Dr.             PROPN    compound     Dr.
# Martinez        PROPN    nsubj        Martinez
# prescribed      VERB     ROOT         prescribe
# amoxicillin     NOUN     dobj         amoxicillin
# 500mg           NOUN     appos        500mg
```

---

## Section 7 — The Ethics of Language AI

### Hallucination and Epistemic Risk

Large language models produce fluent, confident text that is regularly factually incorrect. They cite papers that do not exist, attribute quotes to people who never said them, and describe facts about the world that bear no relationship to reality. This is not a bug awaiting a fix — it is a structural consequence of training systems to produce text that sounds like human text, rather than text verified to be true.

The epistemic risks of deploying hallucinating systems in high-stakes contexts — medical information, legal advice, financial guidance — are severe. Systems that sound authoritative and are wrong, at scale, can cause serious harm. The appropriate response is not to ban the technology but to design carefully around the limitation: build verification pipelines, communicate uncertainty explicitly, and ensure human judgment remains meaningfully in the loop for consequential decisions.

### Bias at Scale

Language models learn from text produced by humans in an unequal world. They reproduce and amplify the biases encoded in that text — gendered associations, racial stereotypes, cultural assumptions. A model fine-tuned on historical hiring records will encode the gender and racial patterns of those decisions. What makes this particularly concerning is scale: a biased human affects the people they personally interact with; a biased AI system deployed to millions affects millions.

### Intellectual Property

Large language models are trained on billions of words of text produced by writers, journalists, programmers, scientists, and artists who did not consent to this use. Whether this constitutes fair use (a legal question being actively litigated) and whether it is ethically acceptable (a normative question without settled answer) are among the most important unresolved questions in contemporary AI ethics.

---

## Section 8 — IAAIS Integration: The Language Module

This week you add the **IAAIS Language Module** — the natural language interface that makes all IAAIS capabilities accessible through text.

The module has three responsibilities:

**Intent classification:** Determine which IAAIS module the user is asking to invoke. "What do you know about Patient Alice?" routes to the Knowledge Base. "Is this result unusual?" routes to the Uncertainty Module and Pattern Recognizer. "What should we do next?" routes to the Planner. A fine-tuned BERT classifier or zero-shot classification handles this routing.

**Entity extraction:** Identify the domain-specific entities mentioned in the query — patient names, medication names, time ranges, clinical values — using NER. The extracted entities become parameters passed to the invoked module.

**Response generation:** Translate module outputs back into natural language that answers the user's original question. For the current IAAIS milestone, rule-based templates are appropriate; Chapter 13 will add an LLM-based generation layer.

| Chapter | Module | Capability |
|---|---|---|
| Ch 2 | Search Engine | Path planning |
| Ch 3 | Knowledge Base | Structured facts and inference |
| Ch 4 | Planner | Goal-directed action sequences |
| Ch 5 | Uncertainty Module | Calibrated probabilistic beliefs |
| Ch 6 | Classifier | Supervised prediction |
| Ch 7 | Pattern Recognizer | Unsupervised structure discovery |
| Ch 8 | Neural Perception Module | Deep feature extraction |
| Ch 9 | Language Module | Natural language interface |

---

## Hands-On Exploration: Medical Text Analysis Pipeline

### The Activity

Open `hands_on_ch9.ipynb` from the course repository.

**Part 1 — Preprocessing and TF-IDF (15 minutes):** Process 200 de-identified clinical notes through a standard preprocessing pipeline. Build TF-IDF representations and train a Naïve Bayes classifier to assign notes to five diagnostic categories. Compare this to a zero-shot BERT classifier. Where does TF-IDF fail that BERT handles?

**Part 2 — Clinical NER (20 minutes):** Process 50 clinical notes with spaCy and a medical NER model. Extract: patients, physicians, organizations, dates, and quantities (lab values, dosages). Build a structured dictionary from each note. How many notes have ambiguous entity boundaries requiring manual review?

**Part 3 — Intent Classification (20 minutes):** Design a zero-shot intent classifier for the IAAIS Language Module. Define five intent categories appropriate for your domain. Test on 20 sample queries. Identify the two most commonly confused intent pairs — what linguistic features make them ambiguous?

### Reflection Questions

1. The zero-shot classifier requires no task-specific training. The fine-tuned BERT classifier achieves higher accuracy but requires labeled examples. Describe the data collection and annotation process you would need to build a fine-tuned classifier for your domain. Estimate the cost.
2. Your NER model misses several medical entity types (drug names, diagnoses) because they don't appear in standard training data. Describe an annotation scheme and training pipeline for a domain-specific medical NER model.
3. A language model is asked "Is this dose of metformin safe for a patient with severe kidney disease?" It produces a fluent, confident answer that is partially incorrect. What verification pipeline would you build around this system before deploying it in a clinical setting?
4. Your IAAIS Language Module frequently confuses "request for diagnosis support" with "request for treatment recommendation." What linguistic features make these intents ambiguous? How would you collect training data to resolve the confusion?

---

## Case Study: Google's BERT Integration into Search

### The Scale

By 2019, Google Search handled approximately 5.6 billion queries per day. The problem was not finding documents — that had been solved by inverted index systems decades earlier. The problem was *understanding* the query.

A query like "parking on a hill with no curb" requires understanding that "no curb" changes what parking on a hill means — a detail that keyword-matching systems miss but BERT handles naturally. "2019 brazil traveler to usa need a visa" requires understanding that "usa" is the destination (requiring a visa) not the origin. These nuances, invisible to keyword search, are exactly the kind of language understanding that BERT was built to provide.

### The Integration

Google integrated BERT into its search ranking pipeline in October 2019. Rather than replacing the retrieval system (which finds relevant documents) or the ranking model (which orders them), BERT augmented the query understanding stage — helping the system interpret what the user actually intended before retrieval began.

The result: an improvement in search quality on 10% of all English queries — approximately 560 million queries per day. Google described it as one of the most significant improvements in search quality in five years.

### The Lesson

The Google BERT integration illustrates a principle that recurs throughout production AI: the most impactful applications are often not the most dramatic ones. BERT-powered search did not change what search engines *do* — it changed how well they understand what users *want*. And because BERT was used to understand queries rather than generate responses, the hallucination risk was mitigated: the content came from verified web pages, not from the model.

Grounded language understanding — using language models to interpret intent, then retrieving or reasoning from verified sources — is the deployment pattern that makes LLMs safe in high-stakes applications. It is the pattern underlying every successful clinical NLP deployment and the foundation of Retrieval-Augmented Generation (Chapter 13).

---

## Chapter Summary

We began with "Attention Is All You Need" and eight researchers who believed that attending to relevant context was sufficient to build the most powerful language processing system ever created. They were right.

The NLP pipeline gave us the foundation: tokenization, normalization, subword vocabulary, and the progression from bag of words through TF-IDF to dense embeddings. Word2Vec showed that meaning can be encoded as geometry — semantic relationships emerging as vector arithmetic from co-occurrence statistics.

Self-attention formalized the notion of contextual representation: each token's meaning depends on all other tokens, weighted by relevance. Multi-head attention allows different heads to specialize in different relationship types simultaneously. Positional encoding preserves sequence order without sacrificing the parallel computation that makes transformers fast.

BERT and GPT showed that the same architecture, with different training objectives, produces systems optimized for understanding and generation respectively — and that the right choice depends entirely on the downstream task. Hugging Face and spaCy made both accessible with minimal engineering effort.

Ethics reminded us that language models are mirrors of their training data — reproducing and amplifying the biases, errors, and value judgments encoded in the text they learned from. Hallucination, bias, and intellectual property concerns are not peripheral; they are central to any responsible deployment of language AI.

In Chapter 10, we turn to computer vision — the domain where deep learning first proved itself transformative and where the consequences of success and failure are most immediately visible.

---

## Discussion Questions

1. **The understanding question:** GPT-4 produces fluent, useful responses to complex questions. A cognitive scientist argues this is sophisticated pattern matching without comprehension. Does this distinction matter for practical AI deployment? In what high-stakes contexts would it matter?
2. **Static vs. contextual embeddings:** Word2Vec assigns one vector per word; BERT assigns different vectors to the same word in different contexts. Give three examples where static embeddings fail but contextual embeddings succeed. Give one example where static embeddings might be preferred.
3. **Scaling and emergence:** GPT-3's in-context few-shot learning appeared as an emergent capability at scale — not directly trained for. What other emergent capabilities have been documented? Does the phenomenon of emergence change how we should think about AI safety?
4. **Hallucination and design:** A clinical LLM answers 94% of drug dosing questions correctly and confidently hallucinates the remaining 6%. Users cannot distinguish correct from incorrect answers. Design a system around this model that makes it safe to deploy in a clinical setting.
5. **Multilingual fairness:** Large language models are predominantly trained on English text, with other languages dramatically underrepresented relative to their number of speakers. What are the equity implications of this imbalance? Who has an obligation to address it?
6. **The consent problem:** A model trained on 800 billion words of internet text can reproduce passages from many of those texts and can generate text in the style of specific authors whose work it trained on. What consent framework should govern this use of human-generated text?
7. **spaCy vs. transformer models:** spaCy processes thousands of documents per second on a CPU; a BERT-based NER model processes tens per second on a GPU. Describe a real application where the throughput difference determines which approach is deployable.
8. **Your IAAIS Language Module:** Define five intent categories for your IAAIS domain. For each: give two example queries that should trigger it, describe which IAAIS module it routes to, and identify one way the intent classifier could fail and what the consequence would be.

---

## Further Reading

### Foundational Papers

Vaswani, A., et al. (2017). Attention is all you need. *Advances in NeurIPS*, 30. The transformer paper.

Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv:1810.04805*.

Mikolov, T., et al. (2013). Distributed representations of words and phrases and their compositionality. *Advances in NeurIPS*. Word2Vec.

### Practical NLP

Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). Available at web.stanford.edu/~jurafsky/slp3/. The definitive NLP textbook.

Hugging Face Documentation. docs.huggingface.co. The authoritative guide to the Transformers library.

### Ethics

Bender, E. M., et al. (2021). On the dangers of stochastic parrots. *FAccT 2021*. The canonical critical analysis of large language model risks.

Weidinger, L., et al. (2021). Ethical and social risks of harm from language models. *arXiv:2112.04359*.

---

*— End of Chapter 9 —*
