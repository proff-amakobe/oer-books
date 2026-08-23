# Chapter 3: The Language of Thought

**Knowledge Representation, Logic, and Machine Reasoning**

*CSC5350 · Artificial Intelligence*

---

## Opening Narrative

### Thirty Years to Teach a Machine What a Child Knows by Three

In 1984, a philosopher named Doug Lenat began one of the most ambitious projects in the history of AI: an attempt to encode into a computer system all the knowledge an adult human needs to read a newspaper. He called the project Cyc — short for encyclopedia.

Lenat's team spent the next three decades entering facts into Cyc's knowledge base by hand. Not just facts about history or science — the obvious things — but the invisible background knowledge that humans carry so effortlessly they forget it is there. Facts like: a person cannot be in two places at once. Knives are typically made of metal, not flowers. Dead people do not attend meetings. If you put something in a box, the box becomes heavier.

By 2014, Cyc contained approximately 25 million facts and 1.5 million rules. It could answer questions that required surprisingly sophisticated reasoning. It could also fail on questions that a three-year-old would answer without hesitation.

The Cyc project illuminated what is now called the **knowledge representation problem**: encoding human knowledge in a form that machines can reason with is far harder than encoding facts. The facts are almost the easy part. The challenge is capturing the *structure* of knowledge — what relates to what, what implies what, what is possible and what is not — in a formal language expressive enough to support reasoning but constrained enough to prevent errors.

This chapter is about that formal language. It is about propositional logic, first-order logic, and the knowledge bases that give AI systems the structured understanding they need to do more than pattern-match. The story of Cyc also reminds us that knowledge representation and knowledge acquisition are inseparable: you cannot build a knowledge base without understanding the domain deeply enough to represent it faithfully.

> **"Knowledge is not merely a collection of facts. It is a structure — a web of relationships, constraints, and implications — and the challenge of knowledge representation is capturing that structure in a language a machine can use."**

---

## Learning Objectives

After completing this chapter, you will be able to:

1. Explain the knowledge representation problem and describe why it is harder than simply storing facts.
2. Define propositional logic, construct truth tables, and identify tautologies and contradictions.
3. Use inference rules — Modus Ponens, Modus Tollens, resolution — to derive new facts from existing ones.
4. Express complex knowledge in First-Order Logic using constants, predicates, functions, quantifiers, and nested structures.
5. Apply forward and backward chaining to a first-order knowledge base and trace the inference steps.
6. Compare the expressiveness and tractability of propositional logic, first-order logic, and their practical subsets.
7. Describe description logics, OWL, and semantic web standards as practical knowledge representation formalisms.
8. Explain the closed-world assumption and open-world assumption and identify when each is appropriate.
9. Build the IAAIS Knowledge Base — a structured repository of domain facts and rules that supports logical inference.

---

## Key Terminology

| Term | Plain-Language Definition |
|---|---|
| **Knowledge Base (KB)** | A structured collection of facts and rules representing an agent's knowledge about a domain. Distinguished from a database by its support for inference — deriving new knowledge from existing knowledge. |
| **Proposition** | A statement that is either true or false. "It is raining" is a proposition. "Is it raining?" is not. |
| **Propositional Logic** | A formal language using propositions and logical connectives (AND, OR, NOT, IMPLIES, IFF) to express compound statements. Supports mechanical inference but cannot express quantified statements. |
| **Connectives** | Logical operators combining propositions: ∧ (AND), ∨ (OR), ¬ (NOT), → (IMPLIES), ↔ (IFF). |
| **Inference** | Deriving new sentences that are entailed by existing sentences. A sentence α is entailed by KB if α is true in every model where KB is true. |
| **Modus Ponens** | The basic inference rule: from P and P → Q, conclude Q. If it is raining and rain causes wet streets, conclude wet streets. |
| **Resolution** | A complete inference procedure for propositional and first-order logic. Derives the empty clause (contradiction) if the negation of the goal follows from the knowledge base. |
| **First-Order Logic (FOL)** | An extension of propositional logic with *objects*, *predicates* (properties and relations), *functions*, and *quantifiers* (∀ for all, ∃ there exists). Dramatically more expressive than propositional logic. |
| **Constant** | A symbol referring to a specific object in the world. John, Penicillin, and Paris are constants. |
| **Predicate** | A symbol expressing a property of or relationship between objects. Doctor(John), Treats(Penicillin, Pneumonia). |
| **Variable** | A placeholder for objects, used with quantifiers. ∀x Person(x) → Mortal(x) — for all x, if x is a person then x is mortal. |
| **Universal Quantifier (∀)** | "For all." ∀x Person(x) → Mortal(x) means every person is mortal. |
| **Existential Quantifier (∃)** | "There exists." ∃x Doctor(x) means at least one doctor exists. |
| **Unification** | The process of finding a substitution for variables that makes two expressions identical. The mechanism that enables first-order inference. |
| **Entailment** | KB ⊨ α means that α is true in every world consistent with KB. Entailment is the semantic relationship that inference algorithms approximate. |
| **Soundness** | An inference algorithm is sound if every conclusion it derives is entailed by the knowledge base — it never produces wrong answers. |
| **Completeness** | An inference algorithm is complete if it can derive every entailed sentence — it never misses correct answers. Resolution is both sound and complete for first-order logic. |
| **Closed-World Assumption** | Facts not in the knowledge base are assumed to be false. Used in databases and Prolog. If Alice's height is not in the KB, conclude Alice has no height. |
| **Open-World Assumption** | Absence of information does not imply falsehood — the world may simply not be known. Used in OWL and semantic web systems. If Alice's height is not in the KB, it is simply unknown. |
| **Description Logic** | A family of logic-based knowledge representation formalisms balancing expressiveness and computational tractability. The formal foundation of OWL. |
| **Forward Chaining** | Inference by repeatedly applying rules to known facts, deriving new facts until the goal is reached. Data-driven. |
| **Backward Chaining** | Inference by working backward from the goal, identifying which rules could prove it, and recursively proving their conditions. Goal-driven. |
| **SPARQL** | The standard query language for RDF knowledge graphs — analogous to SQL for relational databases. |

---

## Section 1 — Why Knowledge Representation Is Hard

A fact is not the same as knowledge. "Aspirin reduces fever" is a fact. But an intelligent medical system needs to know that aspirin is a medication, that medications are taken in doses, that some patients cannot take aspirin (children under 12 due to Reye's syndrome risk, patients on blood thinners), that reducing fever is a treatment goal, that fever is a symptom, that symptoms have causes, that causes should be addressed and not just symptoms suppressed. The knowledge behind a single fact is a complex web.

Three challenges make knowledge representation difficult:

**The knowledge acquisition bottleneck:** Extracting knowledge from human experts and encoding it formally is slow, expensive, and error-prone. Experts often cannot articulate the rules they follow — their expertise is tacit, honed through practice, not explicitly representable. Cyc's thirty-year encoding effort is the extreme expression of this challenge.

**The frame problem:** When the world changes, which other beliefs need updating? If John moves from Boston to New York, he no longer lives in Boston. The knowledge base must update appropriately — but which other facts about John's location, commute, and life need revisiting? Formally specifying what does *not* change is as challenging as specifying what does.

**The qualification problem:** Rules have exceptions, and exceptions have exceptions. "Birds can fly" is a useful rule that fails for penguins, ostriches, injured birds, birds in cages, and birds in outer space. Representing all qualifications explicitly produces a rule base of unmanageable complexity; omitting them produces a system that makes errors in edge cases.

---

## Section 2 — Propositional Logic: The Foundation

Propositional logic is the simplest formal language capable of supporting mechanical inference. Its building blocks are **atomic propositions** — statements that are either true or false — combined using logical connectives.

The five connectives have precise, formal meanings given by truth tables:

| P | Q | ¬P | P ∧ Q | P ∨ Q | P → Q | P ↔ Q |
|---|---|---|---|---|---|---|
| T | T | F | T | T | T | T |
| T | F | F | F | T | F | F |
| F | T | T | F | T | T | F |
| F | F | T | F | F | T | T |

Notice that P → Q is false *only* when P is true and Q is false. This captures the meaning of implication: "if it is raining, then the street is wet" is only violated when it is raining and the street is dry.

A **tautology** is a sentence that is true in every possible assignment of truth values — for example, P ∨ ¬P ("it is raining or it is not raining"). A **contradiction** is a sentence that is false in every assignment — P ∧ ¬P ("it is raining and it is not raining"). Tautologies encode necessary truths; contradictions signal a logical inconsistency in the knowledge base.

### The Core Inference Rules

**Modus Ponens:** From P and P → Q, conclude Q.
"Alice has a fever AND Fever → possible_infection, therefore: possible_infection"

**Modus Tollens:** From ¬Q and P → Q, conclude ¬P.
"The street is not wet AND Rain → wet_street, therefore: it is not raining"

**Resolution:** The most powerful inference rule for automated reasoning. From P ∨ Q and ¬P ∨ R, conclude Q ∨ R. Resolution is refutation-complete: to prove α, add ¬α to the knowledge base and use resolution to derive the empty clause (contradiction).

Propositional logic's limitation is expressive power. We cannot say "all patients with bacterial pneumonia should receive antibiotics" without explicitly listing every patient. For this, we need first-order logic.

---

## Section 3 — First-Order Logic: Expressing the Structure of the World

First-order logic (FOL) extends propositional logic with the ability to talk about *objects*, their *properties*, and *relations* between them — and to make statements about *all* or *some* objects without naming them individually.

### The Vocabulary of FOL

**Constants** name specific objects: `Alice`, `Penicillin`, `BostonMedicalCenter`.

**Predicates** express properties and relations:
- `HasFever(Alice)` — Alice has a fever
- `Treats(Penicillin, BacterialPneumonia)` — penicillin treats bacterial pneumonia
- `WorksAt(Alice, BostonMedicalCenter)` — Alice works at Boston Medical Center

**Functions** map objects to objects: `MotherOf(Alice)`, `CurrentMedication(Patient23)`.

**Variables** stand in for unspecified objects, bound by quantifiers:
- `∀x Patient(x) → EligibleForScreening(x)` — every patient is eligible for screening
- `∃x Physician(x) ∧ OnCall(x)` — there exists a physician who is on call

### Writing Knowledge in FOL

```
# A fragment of a clinical knowledge base in FOL
# Each sentence is a fact or rule the system can reason with.

# Facts about specific individuals
Patient(alice)
HasSymptom(alice, fever)
HasSymptom(alice, cough)
HasLabResult(alice, gram_negative_rods)
HasAllergy(alice, penicillin)

# General medical rules
∀p ∀s HasSymptom(p, fever) ∧ HasSymptom(p, cough) → PossibleInfection(p)

∀p HasLabResult(p, gram_negative_rods) ∧ PossibleInfection(p)
    → SuspectOrganism(p, gram_negative_bacteria)

∀p SuspectOrganism(p, gram_negative_bacteria)
    → Recommend(p, beta_lactam_antibiotic)

∀p ∀d Recommend(p, d) ∧ HasAllergy(p, penicillin) ∧ BetaLactam(d)
    → Contraindicated(p, d) ∧ Recommend(p, carbapenem)

# Taxonomy facts
BetaLactam(amoxicillin)
BetaLactam(piperacillin_tazobactam)
Carbapenem(meropenem)
```

**Forward chaining inference trace:**
```
Start: {Patient(alice), HasSymptom(alice, fever), HasSymptom(alice, cough),
        HasLabResult(alice, gram_negative_rods), HasAllergy(alice, penicillin)}

Step 1: HasSymptom(alice, fever) ∧ HasSymptom(alice, cough)
        → PossibleInfection(alice)                        [Rule 1]

Step 2: HasLabResult(alice, gram_negative_rods) ∧ PossibleInfection(alice)
        → SuspectOrganism(alice, gram_negative_bacteria)  [Rule 2]

Step 3: SuspectOrganism(alice, gram_negative_bacteria)
        → Recommend(alice, beta_lactam_antibiotic)        [Rule 3]

Step 4: Recommend(alice, beta_lactam_antibiotic) ∧ HasAllergy(alice, penicillin)
        ∧ BetaLactam(beta_lactam_antibiotic)
        → Contraindicated(alice, beta_lactam_antibiotic)
        ∧ Recommend(alice, carbapenem)                    [Rule 4]

Final conclusions: Recommend(alice, carbapenem), Contraindicated(alice, beta_lactam)
Explanation: Fever + cough → possible infection; gram-negative rods → suspect GNB;
             GNB → beta-lactam; but penicillin allergy → switch to carbapenem.
```

### Forward vs. Backward Chaining

**Forward chaining** starts with known facts and applies rules repeatedly, asserting new facts until no more rules fire (quiescence) or the goal is derived. It is natural for monitoring and alert systems: as new facts arrive, the system derives their implications automatically.

**Backward chaining** starts with the goal and asks which rules could prove it, recursively proving each rule's conditions. It is natural for diagnostic and query systems: given a goal (what should we recommend for Alice?), find the rules that could derive it and verify their conditions.

The choice depends on the application. Monitoring systems (detect sepsis early) naturally forward chain — new sensor readings flow forward to derived alerts. Diagnostic queries (why should we recommend meropenem?) naturally backward chain — work from the recommendation back to supporting evidence.

---

## Section 4 — Limits of Logic and Practical Formalisms

### The Expressiveness-Tractability Tradeoff

More expressive logic enables more knowledge to be represented — but makes inference harder. Full first-order logic is **semi-decidable**: a sound and complete inference procedure exists, but it may not terminate when the answer is "not entailed." In practice, most deployed knowledge-based systems use restricted subsets of FOL that trade expressiveness for tractability.

**Horn clauses** — implications with a single positive literal in the consequent — support the efficient inference of Prolog and many production rule systems. Forward and backward chaining on Horn clause knowledge bases run in polynomial time.

**Description logics** offer carefully designed subsets of FOL that balance expressiveness with decidability. They form the formal foundation of OWL (Web Ontology Language) and underlie SNOMED CT, DrugBank, and other large medical ontologies.

### The Closed-World and Open-World Assumptions

When a fact is not in the knowledge base, what should the system conclude?

Under the **closed-world assumption (CWA)**, used in databases and Prolog, the absence of a fact implies it is false. If Alice's blood type is not in the database, the system concludes Alice has no blood type — or that the query fails. This is appropriate when the knowledge base is intended to be complete.

Under the **open-world assumption (OWA)**, used in OWL and semantic web systems, the absence of a fact means only that it is unknown, not that it is false. If Alice's blood type is not in the knowledge base, the system makes no conclusion — it simply lacks information. This is appropriate for knowledge bases that are always incomplete representations of the world.

The choice matters practically. A drug interaction checker using CWA concludes "no interaction" for any drug pair not in its database. One using OWA concludes "unknown interaction." For patient safety, the OWA is clearly safer — but also produces more "unknown" alerts that practitioners may ignore.

---

## Section 5 — Knowledge Graphs: Logic at Scale

Modern practical knowledge representation often takes the form of **knowledge graphs**: large structured repositories where nodes represent entities and edges represent typed relationships between them.

The semantic web provides standard languages for knowledge graphs:

**RDF (Resource Description Framework)** represents every fact as a *triple*: subject-predicate-object. `(:alice :hasSymptom :fever)`, `(:penicillin :rdf:type :Antibiotic)`, `(:meropenem :treats :Pseudomonas)`. Every entity has a globally unique URI, enabling facts from different sources to be linked.

**OWL (Web Ontology Language)** extends RDF with description logic expressiveness: class hierarchy, property restrictions, cardinality constraints, and automated classification. An OWL reasoner can infer that meropenem is an Antibiotic (from its class membership) even if the fact was never explicitly asserted — because Antibiotic is a superclass of Carbapenem, and meropenem is a Carbapenem.

**SPARQL** queries knowledge graphs with a syntax analogous to SQL:

```sparql
# Find all medications that treat gram-negative infections
# and are safe for penicillin-allergic patients

SELECT ?medication WHERE {
  ?medication :treats :gram_negative_infection .
  ?medication :rdf:type ?class .
  ?class :safeInPenicillinAllergy true .
}
```

The largest deployed knowledge graphs demonstrate the scale this representation enables. SNOMED CT contains 350,000+ clinical concepts with precise relationships. The Google Knowledge Graph contains hundreds of billions of facts supporting search and voice assistant queries. Wikidata has 100 million+ entities linked across languages and domains.

---

## Section 6 — IAAIS Integration: The Knowledge Base

This week you build the **IAAIS Knowledge Base** — a structured store of domain facts and rules that supports logical inference, query answering, and explanation.

The Knowledge Base is the memory and reasoning layer of IAAIS. Every other module writes to it and reads from it. The Search Engine finds paths; the Knowledge Base knows what those paths mean. The Classifier predicts; the Knowledge Base stores what has been observed and concluded. The Expert Module reasons; the Knowledge Base is where its conclusions persist.

**Design decisions this week:**
- Which logical formalism is right for your domain? Horn clauses for tractable inference? OWL for rich ontological reasoning?
- Closed-world or open-world assumption? What are the safety implications of each choice in your domain?
- How will you represent uncertainty? (Chapter 5 will introduce probabilistic extensions.)
- How will the Knowledge Base explain its conclusions? What format should explanations take for your domain's users?

| Chapter | Module | Capability |
|---|---|---|
| Ch 2 | Search Engine | Path planning |
| Ch 3 | Knowledge Base | Structured facts and logical inference |

---

## Hands-On Exploration: Building a Domain Knowledge Base

### The Activity

Open `hands_on_ch3.ipynb` from the course repository.

**Part 1 — Knowledge Elicitation (15 minutes):** Choose a domain for your IAAIS project. Write 20 facts about your domain in natural language, then translate each into FOL notation. Identify which facts are atomic (cannot be derived from others) and which are rules (can be derived).

**Part 2 — Inference (20 minutes):** Using the provided `KnowledgeBase` class (supporting FOL with backward chaining), add your facts and rules. Write three queries that require inference — not just fact lookup. Trace the backward chaining steps for one query manually.

**Part 3 — Knowledge Graph (20 minutes):** Represent the same knowledge as an RDF knowledge graph using the `rdflib` library. Express three facts as triples. Write a SPARQL query that retrieves information requiring at least one inference step (e.g., all instances of a class through inheritance).

### Reflection Questions

1. Which was harder: expressing your domain knowledge as FOL rules or as RDF triples? What does this tell you about the appropriateness of each formalism for your domain?
2. Identify one piece of domain knowledge that you could not express cleanly in FOL. What additional expressiveness would you need? (Temporal reasoning? Probabilistic statements? Defaults and exceptions?)
3. The closed-world assumption means "not known = false." In your domain, give one example where this assumption is safe, and one where it could cause harm.
4. If your IAAIS Knowledge Base is updated by the Classifier (adding predicted facts) and the Expert Module (adding derived conclusions), what consistency problems might arise? How would you detect and resolve them?

---

## Case Study: Cyc — Thirty Years of Encoding Common Sense

### The Vision

When Doug Lenat launched Cyc in 1984, the vision was ambitious and specific: encode the background knowledge that allows humans to understand language. Not world history or scientific facts — the invisible common sense that underlies communication. If someone tells you "I saw the man with the telescope," you know several things without being told: that the man and the telescope are distinct objects, that the speaker's eyes were involved in the seeing, that the man was probably not transparent, and that this sentence is ambiguous (did you use the telescope to see him, or was he holding it?). Where does this knowledge come from?

Lenat's answer: we must encode it, explicitly, one fact at a time.

### The Reality

By 2014, Cyc contained approximately 25 million facts organized into thousands of microtheories — context-specific knowledge modules that could be activated depending on the domain of discourse. The system could answer questions requiring genuinely sophisticated reasoning: questions about hypothetical situations, questions requiring temporal reasoning, questions about the beliefs and intentions of agents.

It could also fail spectacularly on questions that should be simple. The brittleness of explicitly encoded knowledge is its most reliable characteristic: every edge case the encoders did not anticipate produces a gap.

### The Lesson

Cyc's legacy is not failure. Its 30 years of encoding produced genuine insights about the structure of common sense — which facts are load-bearing (relied upon by many others) and which are islands, which domains yield to explicit encoding and which resist it. The knowledge representation problems Cyc encountered — the frame problem, the qualification problem, the open-world problem — are now central to the field's self-understanding.

The deeper lesson is about the relationship between knowledge representation and knowledge acquisition. You cannot represent what you do not understand. The encoding process forces a precision about domain knowledge that informal description obscures. In this sense, the effort to build a knowledge base is itself a form of discovery — about the domain, and about the limits of formal representation.

Modern approaches combine symbolic knowledge bases with the learned representations of neural networks — recognizing that some knowledge must be explicitly encoded, while other knowledge is best learned from data. This neuro-symbolic synthesis, which we explore in Chapter 12, is where the field is moving.

---

## Chapter Summary

We began this chapter with Doug Lenat's thirty-year attempt to encode common sense — a project that revealed, more than any other, both the depth of the knowledge representation problem and the difficulty of solving it.

Propositional logic gave us the foundation: atomic propositions, logical connectives, truth tables, tautologies, and the inference rules — Modus Ponens, Modus Tollens, resolution — that allow mechanical derivation of new truths from existing ones.

First-order logic extended the foundation to a language expressive enough for real domain knowledge: objects, predicates, functions, and quantifiers that allow us to make statements about all or some entities without naming them individually. Forward and backward chaining showed how inference operates on FOL knowledge bases, and when each direction is natural.

The expressiveness-tractability tradeoff revealed why no single logical formalism dominates: more expressive formalisms support richer knowledge but harder inference. Horn clauses, description logics, and OWL represent different points on this spectrum, each appropriate for different application contexts.

Knowledge graphs — RDF, OWL, SPARQL — showed how logic-based representation scales to production systems with hundreds of millions of facts, powering search engines, clinical systems, and enterprise applications.

In Chapter 4, we move from representing knowledge about the world to *acting* in it — from knowing what is true to planning how to achieve goals.

---

## Discussion Questions

1. **The Wittgenstein problem:** The philosopher Wittgenstein argued that the meaning of words comes from their use in language games, not from formal definitions. How does this challenge the knowledge representation enterprise? Can formal logic capture meaning, or only structure?
2. **Ontological commitment:** When you design a knowledge base, every representational choice commits you to a particular view of the world — what objects exist, what properties they have, how they relate. Describe two domain-specific ontological choices you would make for your IAAIS Knowledge Base and explain the tradeoffs in each choice.
3. **The qualification problem in practice:** The rule "antibiotics treat bacterial infections" has many qualifications. List five qualifications that matter clinically and explain how you would represent them in FOL. At what point does the rule become so qualified as to be useless?
4. **Open-world and safety:** A drug interaction checker using OWA reports "unknown" for any interaction not in its database. A CWA system reports "no known interaction." For patient safety, which assumption is better? What are the practical consequences of each choice for physician workflow?
5. **Knowledge graphs and data integration:** A hospital has three knowledge bases: a medication database, a patient record system, and a clinical guidelines repository. How would you use RDF to integrate these into a single queryable knowledge graph? What alignment problems would you encounter?
6. **Learning vs. encoding:** Modern neural language models can answer many knowledge base queries without any explicit knowledge representation. Does this make formal knowledge representation obsolete? In what domains does explicit representation remain essential?
7. **Provenance and trust:** A knowledge base assertion might come from a clinical trial, a textbook, a case report, or a drug company's marketing material. How would you represent the provenance and trustworthiness of knowledge base facts? What inference rules should treat uncertain facts differently from well-established ones?
8. **Your IAAIS Knowledge Base:** Define the five most important facts your IAAIS domain requires, and the three most important inference rules. Identify one fact that is hard to represent in FOL and explain why.

---

## Further Reading

### Foundational Logic

Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapters 7–10. Authoritative coverage of propositional and first-order logic.

Genesereth, M. R., & Nilsson, N. J. (1987). *Logical Foundations of Artificial Intelligence*. Morgan Kaufmann. Rigorous treatment of knowledge representation and inference.

### Knowledge Graphs and Ontologies

Hitzler, P., et al. (2012). *OWL 2 Web Ontology Language Primer* (2nd ed.). W3C Recommendation. Available at w3.org.

Noy, N. F., & McGuinness, D. L. (2001). *Ontology Development 101: A Guide to Creating Your First Ontology*. Stanford Knowledge Systems Laboratory. A practical, accessible introduction.

### Cyc and Common Sense

Lenat, D. B., & Guha, R. V. (1990). *Building Large Knowledge-Based Systems: Representation and Inference in the Cyc Project*. Addison-Wesley. The primary account of the Cyc project.

Davis, E., & Marcus, G. (2015). Commonsense reasoning and commonsense knowledge in artificial intelligence. *Communications of the ACM*, 58(9), 92–103. Accessible overview of why common sense is hard.

---

*— End of Chapter 3 —*
