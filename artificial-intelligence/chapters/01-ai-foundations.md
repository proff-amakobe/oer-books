# Introduction to Artificial Intelligence

*CSC5350 · Artificial Intelligence · Chapter 1*

*Foundations, History, and Humanity's Search for Intelligence*

---

## Opening Narrative

### The Move That Shook the World

On the evening of March 9, 2016, in a glass-walled conference room in Seoul, South Korea, a professional Go player named Lee Sedol settled into his chair, adjusted his white stones, and prepared to play a game that — by every rational measure — he expected to win.

Go is not chess. Chess has roughly 20 possible moves at any given moment; Go can have more than 200. The total number of legal board positions in Go exceeds the number of atoms in the observable universe. For decades, this mathematical immensity had served as a kind of philosophical fortress: human intuition, honed through years of apprenticeship and pattern recognition built across millions of games, was assumed to be irreducible. A machine could not feel the weight of a stone. A machine could not sense the shape of a game, the harmony or tension of a position, the ineffable quality that masters called *aji* — the lingering possibility, the potential that sleeps inside a configuration. Machines could play chess. Machines could not play Go.

Lee Sedol was one of the best Go players who had ever lived. He had won eighteen world championships. When Google DeepMind announced that its program, AlphaGo, would challenge him to a five-game match, he predicted publicly that he would win 5-0. He was generous enough to consider that he might lose one game.

AlphaGo won the first game. Then the second. Then the third. The match was over before it had officially begun.

But it was the 37th move of the second game that the AI research community would study and debate for years afterward. AlphaGo played a move that no professional human player would have considered. In the language of Go, it was an unusual shoulder hit on the fifth line — a move that expert commentators watching the live stream initially dismissed as a mistake, an error produced by a system that did not truly understand the game. Then they looked more carefully. The move was not a mistake. It was a stroke of what the only word available was: genius. A human professional, watching on stream, stood up and walked out of the room to collect himself.

Lee Sedol later said: *"I thought AlphaGo was based on probability calculation and that it was merely a machine. But when I saw this move, I changed my mind. Surely AlphaGo is creative."*

He did not mean this as a compliment to a tool. He meant it as a recognition of something he had not expected to find: a mind that played differently from any human mind — not worse, not merely faster — but differently. It had discovered a form of insight that humans had not mapped.

Whether AlphaGo was "intelligent" in any meaningful philosophical sense is a question we will spend much of this course examining. What is unambiguous is this: something changed in Seoul in March 2016. A threshold was crossed. And the question it left hanging in the air — over the conference room, over the AI research world, over all of us — was the same question that has driven this field for seven decades:

***What does it actually mean for a machine to be intelligent — and how close are we to finding out?***

That question is where this course begins. It is also, in a very real sense, where all of artificial intelligence begins.

---

## Learning Objectives

### What You Will Be Able to Do

After completing this chapter, you will be able to:

- Explain the relationship between artificial intelligence, machine learning, and deep learning — and describe what distinguishes each level.
- Describe the historical evolution of AI, tracing the journey from symbolic systems and expert programs through machine learning to modern generative AI.
- Define intelligent agents, rational behavior, and the role of environments, perception, and action in agent-based systems.
- Compare symbolic reasoning systems with data-driven learning systems across multiple dimensions — and reason about when each approach is appropriate.
- Explain the impact of AI across healthcare, cybersecurity, transportation, education, and creative industries, and connect each to underlying technical principles.
- Analyze the ethical and societal tensions embedded in real-world AI systems, and reason about them constructively rather than dismissively.
- Set up a working AI development environment using Python, Jupyter Notebook, and the key libraries you will use throughout the semester.
- Articulate the design philosophy for the semester-long capstone project: the Intelligent Adaptive AI System (IAAIS).

---

## Key Terminology

### The Language of Artificial Intelligence

Every field has its vocabulary. The terms below are not a list to memorize before reading — they are a reference to return to as you encounter these ideas in context. What follows is a conceptual map, not a glossary. The terrain makes more sense once you have walked it.

| Term | Plain-Language Definition |
|------|--------------------------|
| **Artificial Intelligence** | The broad field concerned with building computational systems that exhibit behavior we would consider intelligent if a human performed it. This includes reasoning, learning, perception, planning, language understanding, and decision-making. |
| **Machine Learning** | A subset of AI in which systems learn patterns from data rather than following hand-coded rules. Instead of being programmed with explicit instructions, ML systems are shown many examples and discover the underlying structure themselves. |
| **Deep Learning** | A subset of machine learning that uses neural networks with many layers to learn hierarchical representations of data. Deep learning excels on unstructured data — images, audio, language — where manual feature design is impractical. |
| **Generative AI** | AI systems capable of producing new content — text, images, audio, code, video — that resembles human-created output. Examples include large language models, image diffusion models, and music generation systems. |
| **Symbolic AI** | An approach to AI that represents knowledge as symbols and rules, and reasons by manipulating those symbols according to logical principles. Expert systems are the canonical example. Symbolic AI dominated the field from the 1950s through the 1980s. |
| **Intelligent Agent** | Any system that perceives its environment through sensors and acts upon it through actuators in pursuit of a goal. Agents can be physical (robots) or software-based (digital assistants, game-playing programs). The defining characteristic is purposeful interaction with an environment. |
| **Rationality** | Acting to maximize the achievement of one's goal given available information. A rational agent does not require perfect information — it makes the best possible decision given what it knows and can sense. |
| **PEAS** | A framework for specifying an agent's design: **P**erformance measure (how success is defined), **E**nvironment (where the agent operates), **A**ctuators (how it acts), and **S**ensors (how it perceives). Every intelligent agent can be analyzed through this lens. |
| **Search** | In AI, the process of exploring a space of possible states or actions to find a path from an initial state to a goal state. Search underlies navigation, game playing, planning, and many other AI capabilities. |
| **Expert System** | An AI program that encodes the knowledge of a human expert in a specific domain as a set of IF-THEN rules, then applies those rules to solve problems in that domain. MYCIN (medical diagnosis) and XCON (computer configuration) are historical examples. |
| **Knowledge Representation** | The study of how to formally encode facts, relationships, and rules about the world so that an AI system can reason with them. Includes logic, semantic networks, ontologies, and frames. |
| **Natural Language Processing (NLP)** | The subfield of AI concerned with enabling computers to understand, interpret, and generate human language. Modern NLP is dominated by transformer-based language models. |
| **Transformer** | A neural network architecture introduced in 2017, based on an attention mechanism that allows the model to consider relationships between any two parts of its input simultaneously. Transformers power most state-of-the-art language and vision systems today. |
| **Large Language Model (LLM)** | A transformer-based model trained on massive text corpora to predict and generate language. LLMs exhibit broad capabilities — writing, coding, reasoning, translation — without being explicitly programmed for each. |
| **Narrow AI** | AI systems designed to excel at one specific task or domain. Every AI system deployed today is narrow. AlphaGo plays Go; it cannot drive a car or write a sentence. |
| **Artificial General Intelligence (AGI)** | A hypothetical AI system with the ability to perform any intellectual task that a human can — with comparable flexibility and generalization. AGI does not yet exist. |
| **Turing Test** | A test proposed by Alan Turing in 1950 in which a human evaluator converses (via text) with a human and a machine. If the evaluator cannot reliably distinguish the machine from the human, the machine is said to have passed. A benchmark of conversational intelligence, though not universally accepted. |
| **Bias (AI)** | Systematic errors or unfairness in AI system outputs that arise from biased training data, flawed model design, or problematic deployment contexts. Bias can reflect and amplify existing social inequities. |
| **Hallucination** | A phenomenon in which a generative AI system produces confident but factually incorrect output. The system "believes" what it says because it is optimizing for plausible-sounding language, not factual accuracy. |
| **Reinforcement Learning (RL)** | A learning paradigm in which an agent learns by taking actions in an environment and receiving rewards or penalties based on outcomes. AlphaGo used RL to improve beyond human-level play. No labeled training data is required — the agent discovers effective behavior through trial and error. |
| **AI Winter** | A period of reduced funding, diminished optimism, and slowed progress in AI research. Two major AI Winters occurred: the mid-1970s and the late 1980s to early 1990s. Each followed a period of excessive optimism that outpaced actual capability. |

---

## Section 1 — What Is Artificial Intelligence?

Let us start, as all honest inquiries must, with an admission: the term "artificial intelligence" has never had a clean definition. It was coined at a summer workshop in Dartmouth in 1956, and every decade since, researchers have debated, revised, and contested what it actually means. This ambiguity is not a failure of the field. It reflects the genuine difficulty of the underlying question.

Intelligence, after all, is not a binary property. It is not a switch that is either on or off. When we say that a system is intelligent, we usually mean something like: it responds adaptively and purposefully to its situation in ways that require something we would recognize as reasoning, learning, or judgment. But this definition is loose enough to include thermostats (which respond adaptively to temperature) and tight enough to exclude most software programs (which respond only to what they were explicitly programmed for).

The founders of the field tried to sidestep this philosophical quicksand with a practical definition. Artificial intelligence, in the version articulated by John McCarthy at Dartmouth, was the science and engineering of making intelligent machines. Not machines that understand, not machines that feel — machines that *act* intelligently. This shift from the philosophy of mind to engineering pragmatism was deliberate. It allowed the field to make progress without resolving the hard questions.

That pragmatic tradition continues today. When practitioners talk about AI, they almost always mean one of four things:

### Four Ways to Think About AI

**AI as reasoning and logic** — The oldest framing: intelligence is essentially the ability to manipulate symbols according to rules. If you can represent knowledge formally and write programs that reason with it, you have intelligence — or at least a compelling simulation of it. This view produced expert systems, theorem provers, and planning algorithms. It is still alive and important today.

**AI as rational action** — Intelligence means always taking the action most likely to achieve your goals given what you know. This view, developed rigorously in the work of Stuart Russell and Peter Norvig, makes no claim about whether a system "understands" anything. It asks only: is it acting well? This framing connects AI to decision theory, economics, and operations research.

**AI as perception and pattern recognition** — Intelligence is fundamentally about making sense of a rich and ambiguous world — seeing faces in photographs, understanding spoken words, parsing the meaning of a sentence. This view drove the computer vision and NLP revolutions and, ultimately, deep learning.

**AI as generation** — Truly intelligent systems should be able to create — to produce text, images, music, and code that are indistinguishable from human creation. Generative AI represents this view in its purest form.

These four framings are not competing definitions. They are four angles of view onto a genuinely multifaceted phenomenon. A sophisticated AI system today — one of the large language models that has drawn so much attention and controversy — incorporates elements of all four: it reasons about language, acts to produce helpful responses, recognizes patterns across its vast training data, and generates novel output.

### The Nested Circles: AI, ML, Deep Learning, and Generative AI

You will hear deep learning described as a subset of machine learning, which is a subset of AI. You will hear generative AI described as a capability that can be built using deep learning. This layered relationship is accurate, but the word "subset" can mislead. It suggests replacement — as if each inner circle superseded the outer one. A better image is nested circles, each adding a specific kind of capability.

- **Artificial Intelligence** (outermost): The broadest aspiration. Everything in the field — rule-based systems, expert programs, search algorithms, planning systems, machine learning models — lives in this circle.
- **Machine Learning** (inside AI): The specific idea that intelligent systems should learn from data rather than following explicit rules. ML includes many distinct approaches — decision trees, random forests, support vector machines, gradient boosting — each with its own strengths.
- **Deep Learning** (inside ML): Learning using neural networks with many layers. Deep learning is not just a more powerful version of ML; it is a qualitatively different approach to representing knowledge, one that discovers its own features from raw data rather than relying on human-designed feature engineering.
- **Generative AI** (inside or alongside deep learning): Systems designed not just to classify or predict, but to produce — large language models like GPT-4, image generation systems like Stable Diffusion, and music and video generation systems.

> *Each circle does not replace the one outside it. It extends what is possible within it. A generative AI system is still a deep learning system. A deep learning system is still a machine learning system. And all of them are still artificial intelligence.*

### Narrow AI, AGI, and the Limits of Current Systems

Every AI system deployed in the world today — including the most capable large language models — is **narrow AI**. It is designed to excel at a specific class of tasks, and its capability does not generalize across arbitrary domains the way human intelligence does. AlphaGo is superhuman at Go and cannot write a sentence. GPT-4 can write sophisticated prose and cannot reliably perform long chains of arithmetic.

**Artificial general intelligence** — a system with human-comparable flexibility and generalization across arbitrary tasks — remains a theoretical aspiration. Whether it is achievable, and what "achievable" even means in this context, is one of the most contested questions in the field.

Beyond AGI, some researchers discuss **superintelligence**: systems that would surpass human cognitive capability in every domain. The ethical and existential implications of superintelligence — if it is ever achieved — are among the most important questions in AI safety research.

---

## Section 2 — A History of Dreams, Winters, and Breakthroughs

The history of artificial intelligence is not a clean arc of progress. It is a story of dramatic oscillation — visionary ideas, genuine breakthroughs, crushing disappointments, abandoned laboratories, and then, against all expectation, vindication.

### The Intellectual Foundations: Before AI Had a Name (1936–1955)

The story arguably begins not with a computer but with a question. In 1936, a twenty-four-year-old mathematician named Alan Turing published a paper that defined, for the first time, what it meant to compute. His imaginary "Turing machine" — a theoretical device that could read and write symbols on an infinite tape according to a finite set of rules — proved that computation was a formal, mechanical process. The implications were staggering: if thought could be reduced to symbol manipulation, and symbol manipulation was computable, then thought itself might be computable.

Turing returned to this question in 1950, in a paper titled "Computing Machinery and Intelligence." He opened with a question that has echoed through the field ever since: *"Can machines think?"* He then, with characteristic pragmatism, proposed setting that unanswerable question aside in favor of a testable substitute: the imitation game, now known as the **Turing Test**.

Two years before Turing's 1950 paper, Norbert Wiener published *Cybernetics*, introducing the idea that feedback and control — the mechanisms by which systems regulate themselves in response to their environment — were the unifying principles of both biological and mechanical intelligence.

### The Founding Moment: Dartmouth (1956)

Artificial intelligence was formally named at a summer workshop at Dartmouth College in 1956, organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. The proposal that launched the workshop was audaciously confident:

> *"We propose that a 2-month, 10-man study of artificial intelligence be carried out... The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it."*

What followed was a period of infectious optimism. Herbert Simon and Allen Newell built the General Problem Solver. Simon predicted in 1965 that "machines will be capable, within twenty years, of doing any work a man can do." He was wrong — but in an interesting way.

### Symbolic AI and Its Limits (1956–1974)

The dominant approach of early AI was symbolic. This approach produced genuine achievements:

- **LISP**, one of the oldest high-level programming languages, was created to support symbolic AI.
- Programs were built that could play checkers, prove mathematical theorems, and understand natural language in limited domains.
- **ELIZA**, written by Joseph Weizenbaum at MIT in the mid-1960s, simulated a psychotherapist using simple pattern-matching rules. Users who knew it was a program reported feeling genuinely understood.

ELIZA's most important lesson: the appearance of intelligence and the reality of intelligence are dangerously easy to conflate. ELIZA understood nothing. It matched patterns and reflected them back.

The limits of symbolic AI became increasingly apparent through the late 1960s and early 1970s. The fundamental problem was the **knowledge representation problem**: encoding the vast background of common-sense knowledge humans take for granted proved far harder than anyone anticipated.

### The First Winter (1974–1982)

In 1973, Sir James Lighthill published a report concluding that AI had failed to achieve its advertised goals and that the prospects for meaningful progress were poor. Funding evaporated. Positions were eliminated. The field contracted sharply.

The first AI Winter was not a total freeze — it was a pruning. The community had overpromised. The gap between what symbolic AI could do in constrained laboratory settings and what it could do in the real world was vast. The winter forced a more sober accounting.

### The Expert Systems Era and the Second Winter (1982–1993)

The thaw came from an unexpected direction: business. In the early 1980s, **expert systems** demonstrated that narrow, well-defined domains could be addressed effectively with symbolic AI. **XCON**, developed at Carnegie Mellon for Digital Equipment Corporation, configured computer systems from customer orders and was reportedly saving DEC millions of dollars per year by the mid-1980s.

The commercial excitement drove a wave of investment. The Japanese government launched the **Fifth Generation Computer Project**, a billion-dollar initiative. But expert systems had fatal flaws:

- They required painstaking knowledge engineering for every new domain.
- They were brittle — rules that worked 95% of the time would fail mysteriously on the other 5%.
- They could not learn.

By the late 1980s and early 1990s, the gap between expectation and capability had opened again. A second, milder AI Winter set in.

### The Machine Learning Revolution (1990s–2010)

Even as expert systems faded, a quieter revolution was underway. Researchers began making steady progress on systems that **learned from data** rather than following hand-coded rules. Key advances included:

- **Support Vector Machines** by Vladimir Vapnik and colleagues — elegant classification achieving state-of-the-art performance.
- **Random forests**, **gradient boosting**, and **Bayesian methods** all made significant advances.
- The internet was making data abundant in unprecedented ways.
- The **backpropagation algorithm**, popularized by David Rumelhart and Geoffrey Hinton in the mid-1980s, kept neural networks alive through the margins.

### The Deep Learning Era (2012–Present)

The moment that ended the long marginalization of neural networks can be dated with unusual precision: **September 30, 2012**.

On that day, the ImageNet Large Scale Visual Recognition Challenge results were announced. The leading traditional approaches achieved error rates around 26%. Then came **AlexNet** — a deep convolutional neural network from Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton — with an error rate of **15.3%**.

Eleven percentage points. This was not a close victory. It was a demonstration of a different order of capability. The deep learning era had begun.

What followed was an extraordinary acceleration:

| Year | Milestone |
|------|-----------|
| 2012 | AlexNet wins ImageNet by 11 percentage points |
| 2014 | Ian Goodfellow introduces Generative Adversarial Networks |
| 2016 | AlphaGo defeats Lee Sedol 4-1 |
| 2017 | Transformer architecture introduced ("Attention Is All You Need") |
| 2020 | AlphaFold 2 effectively solves protein structure prediction |
| 2020 | GPT-3 demonstrates emergent capabilities at scale |

We are living through the middle chapters of this story.

---

## Section 3 — Intelligent Agents and Rational Systems

Before we can build intelligent systems, we need a precise vocabulary for talking about what we want them to do. The most useful such vocabulary comes from the concept of the **intelligent agent**.

### What Is an Agent?

An agent is any entity that **perceives its environment** and **takes actions in pursuit of a goal**. The definition is deliberately broad. By this definition, a thermostat is an agent — it perceives temperature and acts by switching heating or cooling on or off. So is a chess program. So is a self-driving car. So is a recommendation algorithm. So is a human being.

What distinguishes an **intelligent** agent from a simple reactive system is the relationship between what it perceives and what it does. A thermostat responds mechanically to a single measurement. An intelligent agent considers its current situation — including what it has perceived over time, what it knows about the world, and what goals it is trying to achieve — and selects actions that it has good reason to believe will advance those goals. **Rationality** is the key concept.

### PEAS: A Framework for Designing Agents

Every agent can be specified through four dimensions:

| PEAS Component | Definition | Example: Self-Driving Car |
|----------------|------------|--------------------------|
| **P**erformance | What does success look like? | Speed, safety, legal compliance, passenger comfort |
| **E**nvironment | Where does the agent operate? | Physical road, traffic, weather, pedestrians |
| **A**ctuators | How does the agent act? | Steering, acceleration, braking |
| **S**ensors | How does the agent perceive? | Cameras, radar, lidar, GPS |

**Environments vary along several key dimensions:**

- **Observable vs. partially observable** — Chess is fully observable; driving is not.
- **Deterministic vs. stochastic** — Actions have certain vs. uncertain outcomes.
- **Static vs. dynamic** — The environment changes while the agent deliberates.
- **Discrete vs. continuous** — Finite vs. infinite possible states and actions.

Running any AI system through PEAS analysis before you build it is one of the most valuable habits a practitioner can develop. Many AI failures in the real world trace back to a mismatch between the performance measure that was optimized and the performance measure that actually matters.

### Rationality and Its Limits

A **rational agent** takes the action most likely to maximize its performance measure given its current knowledge and perceptions. Rationality is not omniscience — a rational agent can still make mistakes if it lacks crucial information. Rationality is not perfection — it is doing the best possible with what is available.

Many of the most important technical problems in AI are, at their core, problems of rational decision-making under uncertainty: how to represent incomplete knowledge, how to reason about probability, how to plan over long time horizons when outcomes are unpredictable.

---

## Section 4 — Symbolic AI vs. Learning-Based AI: A Defining Tension

No tension in AI has been more persistent, more productive, or more misunderstood than the divide between symbolic AI and learning-based AI.

### The Symbolic Vision

Symbolic AI rests on a powerful intuition: *intelligence is reasoning, reasoning is symbol manipulation, and therefore intelligence can be built by designing systems that manipulate symbols correctly.*

**Strengths:**
- **Interpretable** — you can read the rules and understand why the system made a decision.
- **Reliable** within its domain — if the rules are correct, the system applies them correctly.
- **Requires no training data** — knowledge comes from human experts.
- **Can reason** about novel situations using logical inference.

**Weaknesses:**
- **Knowledge acquisition bottleneck** — extracting and encoding expert knowledge is slow, expensive, and error-prone.
- **Brittleness** — rule-based systems fail unpredictably on unanticipated inputs.
- **Common-sense knowledge problem** — human reasoning relies on vast background knowledge that is too obvious to state and too enormous to enumerate.

### The Learning-Based Vision

Learning-based AI starts from a different intuition: *perhaps we don't need to explicitly encode what intelligence is — perhaps we can build systems that discover it themselves, given enough examples.*

In machine learning, the algorithm discovers the parameters that make the model perform well on training examples. Deep learning goes further: not only does the algorithm learn the parameters, it also learns the **features** — the relevant aspects of raw data to pay attention to.

**Strengths:**
- Handles the messiness of real-world data.
- Improves with more data.
- Does not require laborious knowledge engineering.

**Weaknesses:**
- Requires large amounts of labeled training data.
- **Opaque** — we cannot easily read a neural network's parameters and understand why it makes a given prediction.
- Can fail in surprising ways when test data differs from training data (**distribution shift**).

### Where Symbolic AI Still Matters

It would be a mistake to read deep learning's success as a verdict against symbolic AI. Symbolic methods remain essential in many contexts:

- **Interpretability requirements** — medical decision support, financial risk assessment, and criminal justice applications often require human-readable explanations.
- **Small data regimes** — Bayesian inference and probabilistic graphical models outperform data-hungry deep learning when examples are scarce.
- **Well-specified rule domains** — constraint satisfaction, formal verification, and automated theorem proving are uniquely suited to symbolic methods.

The most sophisticated AI systems today increasingly combine both paradigms — a direction sometimes called **neuro-symbolic AI**.

> *The question is not which approach is correct. The question is which approach is right for the problem in front of you — and increasingly, how the two can be combined.*

### The Evolutionary Arc

| Era | Paradigm | Why It Emerged | Why It Gave Way |
|-----|----------|----------------|-----------------|
| 1950s–1970s | Symbolic AI | Compute was expensive; data was scarce | Knowledge acquisition bottleneck; brittleness |
| 1980s–1990s | Expert Systems | Demonstrated commercial value | Could not learn; fragile; expensive to build |
| 1990s–2010 | Machine Learning | Data became abundant; compute more accessible | Plateaued without deep feature learning |
| 2012–present | Deep Learning | Data vast; compute cheap; GPU revolution | Still active; opacity and data hunger remain challenges |
| 2017–present | Generative AI | Transformer architecture; scale | Still unfolding |

---

## Section 5 — AI in the Real World

### Healthcare and Medicine

Deep learning systems can now detect diabetic retinopathy, a leading cause of blindness, from retinal photographs with accuracy matching board-certified specialists. AI-assisted pathology systems can identify cancer cells in tissue slides faster and, in some studies, more accurately than experienced pathologists.

**Ethical dimensions:** When an AI system recommends or withholds a diagnosis, who bears responsibility if it is wrong? Documented disparities in AI system performance across race, gender, and age have already emerged in dermatology, radiology, and emergency medicine. Technical capability and equitable deployment are separate problems, and solving the first does not automatically solve the second.

### Agriculture and Food Security

Deep learning systems trained on satellite imagery and drone photographs can detect crop diseases, predict yield, identify weed infestations, and optimize irrigation — with significant implications for a world that must feed eight billion people while adapting to climate disruption.

**Ethical dimensions:** Technology developed for large industrial farms may not be accessible to smallholder farmers in low-resource settings — the farmers for whom improved efficiency would be most consequential.

### Transportation and Autonomous Systems

Self-driving vehicles must perceive a complex, dynamic environment through multiple sensor modalities; plan routes and maneuvers in real time; and balance competing objectives — speed, comfort, legal compliance, and the safety of all road users.

**Ethical dimensions:** Who bears liability when an autonomous vehicle is involved in an accident? How should regulators evaluate safety before allowing deployment at scale? The distribution of risk across thousands of ordinary driving decisions matters more than hypothetical edge cases.

### Cybersecurity

Recurrent neural networks and transformer models are now routinely used to detect anomalous behavior in network traffic, identify malicious code, and flag security incidents.

**Ethical dimensions:** The dual-use problem is severe and permanent. The same capabilities that defend systems can attack them. The landscape is a continuous arms race, and the technology itself is neutral.

### Education

AI tutoring systems that adapt instruction to individual student pace and prior knowledge have shown genuine promise. Large language models are enabling new forms of personalized writing feedback and on-demand explanation.

**Ethical dimensions:** AI systems trained on historical educational data will encode historical inequities. Over-reliance on AI assistance may impair the development of underlying skills. AI evaluation of student work raises fairness and accuracy concerns.

### Creative Industries

Generative image models can produce photorealistic portraits and illustrations from text descriptions. Language models can write stories, poems, screenplays, and code. Music generation systems can compose in the style of any artist whose work appeared in their training data.

**Ethical dimensions:** If a generative model produces an image in the style of a living artist, using training data that included that artist's work without consent, is that theft? What does authorship mean for a work produced in collaboration between a human and a generative system? These debates are unfolding in courts and legislatures in real time.

---

## Section 6 — The Ethics of Intelligence

Ethics is not a chapter to be appended to an AI textbook. It is woven into every design decision, every deployment choice, every choice about what data to collect, what objective to optimize, and what tradeoffs to accept.

### Bias and Fairness

AI systems learn from data produced by human societies. Human societies are not fair. When the data reflects historical patterns of discrimination, AI systems trained on that data will reproduce and often amplify those patterns. A facial recognition system trained predominantly on lighter-skinned faces will perform less accurately on darker-skinned faces.

Fairness is not a technical property that can be added as an afterthought. It requires deliberate attention at every stage: in data collection, in the choice of training objective, in evaluation across demographic groups, and in the deployment context. Different mathematical definitions of fairness are often **mutually incompatible** — making the fairness of an AI system inescapably a value judgment.

### Hallucination and Epistemic Risk

Large language models produce fluent, confident-sounding text. They also, regularly, produce fluent, confident-sounding text that is **false**. This phenomenon — **hallucination** — is not a bug that will be engineered away; it is a structural consequence of how these systems are trained, optimizing for text that sounds plausible rather than text that is true.

The appropriate response is not to ban the technology but to design carefully around its limitations: to build in verification, to communicate uncertainty honestly, and to ensure that human judgment remains meaningfully in the loop for consequential decisions.

### Labor Displacement and Economic Disruption

AI-driven automation increasingly affects cognitive work rather than only physical work. The outcome will depend heavily on policy choices that have not yet been made. Honest practitioners should resist both the techno-optimist claim that everything will work out fine and the techno-pessimist claim that mass unemployment is inevitable.

### Privacy and Surveillance

Facial recognition, gait recognition, behavioral profiling, and predictive policing all use AI capabilities to monitor, track, and categorize people at a scale that would have been impossible a decade ago. Democratic societies must address the relationship between technology and power through law, policy, and deliberate collective choice.

### Autonomy, Accountability, and the Responsibility Gap

When an AI system causes harm — a wrong diagnosis, an unjust content moderation decision, a vehicle accident — who is responsible? The developers? The company? The regulator? The user? As AI systems become more capable and autonomous, this **responsibility gap** grows.

### AI Alignment

A community of researchers is focused on ensuring that very capable AI systems pursue goals that are aligned with human values and interests. The **alignment problem** is difficult precisely because specifying human values in a form that a system can be trained on is hard. A system given a poorly specified objective and highly capable of pursuing it can cause catastrophic harm even without malicious intent.

> *Every AI system embeds value judgments — about what matters, about who benefits, about what tradeoffs are acceptable. Those judgments are not neutral, even when they are implicit. Part of becoming a thoughtful AI practitioner is learning to see those judgments clearly, and to make them deliberately.*

---

## Section 7 — Introducing the Semester Project: IAAIS

Throughout this course, you will build something: a single, evolving intelligent system that grows in capability each week. By the final week of the semester, you will have built a functioning **Intelligent Adaptive AI System — IAAIS**.

### What IAAIS Will Become

At full capability, IAAIS will:

- **Reason symbolically** using logical inference and knowledge representation
- **Search intelligently** using classical and heuristic search algorithms
- **Classify and predict** using machine learning models trained on real data
- **Process language** using modern transformer-based NLP tools
- **Connect to generative AI** capabilities through API integration
- Do all of this within a coherent, documented architecture you will design, build, test, and present

### This Week: The Design Philosophy Document

Before writing a line of code, the most important work is thinking. Your task this week is to write a **one-to-two-page IAAIS Design Philosophy Document** addressing four questions:

1. **Purpose** — What problem should IAAIS solve? For whom? What capability should it provide that does not exist today, or that is currently inaccessible to the people who need it?

2. **Intelligence goals** — What kinds of intelligence should IAAIS exhibit? Should it reason logically? Search spaces? Learn from feedback? Generate language? Be specific about the capabilities you aspire to build.

3. **Stakeholders** — Who will use this system? Who else might be affected by it — with or without their knowledge or consent? Whose interests might it serve, and whose might it neglect?

4. **Risks** — What could go wrong? What harms could this system enable, even if unintentionally? What safeguards should be built in from the beginning?

This document is a living artifact. You will return to it every week as your technical understanding grows. By Week 16, it will be both a design record and a reflection on how understanding changes what we build.

---

## Section 8 — Setting Up Your Development Environment

A working development environment is the foundation for all hands-on work ahead. Take the time to set this up correctly now — dependency conflicts and environment errors are almost entirely preventable.

### Step 1: Python

Download and install **Python 3.10 or later** from [python.org](https://python.org). During installation, ensure Python is added to your system PATH. Verify:

```bash
python --version
```

You should see a version number of 3.10 or higher.

### Step 2: Virtual Environment

Create and activate a virtual environment for your project:

```bash
# Create the environment
python -m venv iaais_env

# Activate (macOS / Linux)
source iaais_env/bin/activate

# Activate (Windows)
iaais_env\Scripts\activate
```

Once activated, your terminal prompt will change to show the environment name.

### Step 3: Core Dependencies

```bash
pip install jupyter scikit-learn numpy pandas matplotlib seaborn
pip install networkx spacy nltk transformers torch
pip install openai streamlit python-dotenv
```

Download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

### Step 4: Jupyter Notebook

Launch Jupyter with:

```bash
jupyter notebook
```

This opens a browser window showing the Jupyter interface. Navigate to your project directory and create a new notebook to verify the installation.

### Step 5: Version Control

Initialize a Git repository and connect it to GitHub:

```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/iaais.git
```

Commit your environment setup files (`requirements.txt` and a `README`) as your first commit. The habit of committing early and often is worth establishing from day one.

### Step 6: Verification

Run the provided **`setup_check.ipynb`** from the course repository. This notebook checks all dependencies, runs a quick test of each major library, and confirms the basic pipeline works end to end.

---

## Hands-On Exploration

### Can a Machine Be Intelligent? A Comparative Experiment

#### The Activity

Open the notebook `hands_on_ch1.ipynb` from the course repository. It contains three AI interactions: a rule-based chatbot, a simple machine learning classifier, and an interface to a modern large language model. You will interact with all three and reflect carefully on what you observe.

#### Part 1: The Rule-Based Chatbot (15 minutes)

The notebook includes a simple chatbot implemented using pattern matching — the same basic approach used by ELIZA in 1966. It responds to inputs based on hand-coded rules:
- IF the user says "hello" THEN respond "Hi there."
- IF the user's input contains "sad" THEN ask "What's making you feel that way?"

Interact with the chatbot for at least five minutes. Try to understand how it works. Then try to **break it** — give it inputs its rules do not anticipate. Document your observations: when does it seem intelligent? When does it fail?

#### Part 2: The Machine Learning Classifier (15 minutes)

The notebook includes a text sentiment classifier trained on movie reviews using a bag-of-words approach and a Naive Bayes classifier. Give it a variety of inputs:

- Obviously positive sentences
- Obviously negative sentences
- Ambiguous sentences
- Sentences with unusual syntax
- Sarcastic sentences

Record the classifier's confidence scores. When does it get things right? When does it fail? Does its failure mode look different from the rule-based chatbot's?

#### Part 3: The Large Language Model (15 minutes)

The notebook includes an interface to a modern LLM. Ask it complex questions. Ask it to reason through a problem. Ask it something factual that you know the answer to, then ask something you suspect it might get wrong.

Pay attention to confidence and fluency: does fluent, confident language feel like intelligence? Does it feel the same when the output is wrong?

#### Reflection Questions

Write a **250- to 350-word reflection** in the notebook addressing the following:

1. How did the three systems differ in the kinds of intelligence they seemed to exhibit? How did they differ in their failure modes?
2. When did each system seem most intelligent? What was actually happening at those moments?
3. Is there a point at which the simulation of intelligence and the reality of intelligence become indistinguishable? Does it matter?
4. Turing proposed that we evaluate machine intelligence by whether we can tell it apart from human intelligence in conversation. Based on your experiment, do you think this is a good test? What does it miss?

---

## Case Study

### AlphaGo vs. Lee Sedol: What a Board Game Taught Us About Intelligence

#### The Problem

Go has been played for more than 2,500 years. In that time, it accumulated a tradition of mastery built on pattern recognition, strategic intuition, and aesthetic sensibility. For decades, Go was considered the last major game where human beings had a decisive and permanent advantage over machines. Chess had fallen to Deep Blue in 1997, but Go's vastly larger search space seemed to place it beyond the reach of brute-force computation. The conventional wisdom was that human-level Go was at least a decade away.

#### The Technical Solution

AlphaGo, developed by DeepMind, combined three innovations:

**1. Deep neural networks for position evaluation**
Networks trained on millions of recorded human games that learned to assess the value of a board position through pattern recognition — seeing the board the way a strong human player sees it: holistically, with attention to local patterns and global implications.

**2. Monte Carlo Tree Search (MCTS)**
A search algorithm that explores the game tree probabilistically — sampling promising lines of play deeply rather than exploring everything shallowly. The neural networks were used to guide the search, focusing computation on the most fruitful parts of the tree.

**3. Reinforcement learning through self-play**
AlphaGo played versions of itself millions of times, receiving rewards for winning and penalties for losing, and gradually updating its networks toward moves that led to victory. This allowed AlphaGo to move beyond patterns it had learned from human games — to discover strategies humans had never played and never recorded.

The combination was decisive. AlphaGo defeated Lee Sedol **4-1**. A subsequent version, **AlphaGo Zero**, trained entirely through self-play with no human game data, became substantially stronger than AlphaGo in three days and surpassed all previous versions within forty days.

#### Move 37 and the Question of Creativity

The move that most captivated observers — the fifth-line shoulder hit in game two — was not a move that appeared in AlphaGo's human training data. AlphaGo discovered it through self-play: a move that human professionals had considered and rejected as unlikely to be good, but that AlphaGo determined was, in this specific position, excellent.

Fan Hui, the European Go champion, watched that move live and later said: *"It's not a human move. I've never seen a human play this move. So beautiful."*

Lee Sedol, who lost the game in which the move appeared, won game four — the only game a human beat AlphaGo in the match. His winning move in that game was itself described as a move of remarkable creativity: a response that AlphaGo had not anticipated.

#### Lessons for AI Practitioners

**What AlphaGo demonstrated:**
- When you combine sufficient compute, sufficient data, a well-specified objective, and a powerful learning algorithm, capabilities emerge that were not designed and were not predicted.
- In domains with clear feedback signals, self-play reinforcement learning can discover knowledge that humans have not yet discovered.

**What AlphaGo's limitations reveal:**
- AlphaGo can play Go. It cannot transfer what it has learned to chess, or to any other task. This is the defining characteristic of narrow AI.
- Impressive domain-specific performance does not generalize.

**The ethical dimensions:**
- The resources required to build AlphaGo are available only to well-funded research organizations.
- The concentration of the most powerful AI capabilities in a small number of institutions raises governance questions that extend well beyond any individual application.

---

## Chapter Summary

### What We Have Learned

We began this chapter in a conference room in Seoul, watching a machine do something that a professional human player called creative. We end it with the conceptual foundations needed to understand how — and whether — that description makes sense.

**Artificial intelligence** is not a single technology. It is a family of approaches to building systems that exhibit intelligent behavior: reasoning, learning, perceiving, planning, generating, and deciding. The boundaries between AI, machine learning, deep learning, and generative AI are real but nested — each inner circle extends what is possible within the outer one without replacing it.

**The history of AI** is a story of dramatic oscillation between optimism and disappointment, driven by the recurring gap between what the field promised and what it could deliver. Symbolic AI captured real insight but foundered on the knowledge acquisition bottleneck. Expert systems demonstrated commercial value in narrow domains but could not scale or learn. Machine learning introduced learning from data. Deep learning demonstrated that with the right architecture, sufficient data, and sufficient compute, the field could exceed human performance on tasks previously considered uniquely human. Generative AI has pushed these capabilities further, into the territory of creation.

**Intelligent agents** are the conceptual framework that unifies these approaches: any system that perceives its environment and acts to achieve goals. The PEAS framework is a practical tool for designing and analyzing AI systems. Rationality is the standard against which agent behavior is measured: not perfection, but optimal action given available information.

**The tension between symbolic and learning-based AI** is not resolved. It is productive. The most sophisticated AI systems of the near future will likely combine both paradigms.

**Ethics** is not a chapter appended to AI. It is woven through every design decision, every deployment choice, every choice about data, objective, and tradeoff. Bias, hallucination, privacy, accountability, and alignment are not peripheral concerns — they are central to what it means to build AI systems responsibly.

In Chapter 2, we will open the hood of problem-solving in AI. We will discover that many of the most important AI capabilities — game playing, route planning, puzzle solving, logical inference — reduce at their core to a common formalism: the systematic search through a space of possible states. The terrain ahead is rich, and we have just begun to explore it.

---

## Discussion Questions

### Questions for Reflection and Debate

These questions have no single correct answers. They are designed to be debated, refined, and revisited as your understanding deepens throughout the course.

**1. The Turing Test and Its Limits**
Alan Turing proposed that a machine should be considered intelligent if a human evaluator cannot reliably distinguish its conversational responses from a human's. Modern large language models can pass this test in many contexts. Does that mean they are intelligent? What does "intelligent" actually require — and is the inability to be distinguished from a human a sufficient condition, a necessary condition, or neither?

**2. Creativity and AlphaGo**
Lee Sedol called AlphaGo's 37th move "creative." Fan Hui called it "not a human move." Are these two descriptions compatible? Can a system produce genuinely creative output without understanding what it has created? Is there a meaningful difference between discovering a good move through reinforcement learning and discovering it through human insight?

**3. The AI Winters and Scientific Optimism**
Neural network research survived two periods of widespread dismissal because a small community kept working despite the field's contraction. What does this pattern tell us about how scientific progress actually happens? Are there ideas being dismissed today that will be vindicated in ten years? How would you distinguish genuine progress from fashionable hype?

**4. The Knowledge Representation Problem**
Symbolic AI struggled with encoding common-sense knowledge. Deep learning sidesteps this problem by learning representations from data, but the knowledge it learns is distributed, implicit, and uninterpretable. Which approach produces more trustworthy systems? Is trust the right criterion, or should we prioritize accuracy, fairness, interpretability, or something else?

**5. The Ethics of Training Data**
Large language models and image generation systems are trained on text and images from the internet, including works created by millions of writers, artists, photographers, and musicians who did not consent to their use as training data. What ethical framework should govern this practice? Does the social benefit of capable AI systems justify the use of unconsented training data? Who should have the right to decide?

**6. Narrow AI and the Illusion of Understanding**
Every AI system deployed today is narrow — it excels at specific tasks and fails entirely outside them. Yet these systems often seem to understand in ways that feel general. Is the appearance of general understanding evidence of genuine general capability, or is it a more sophisticated version of ELIZA's pattern matching? How would you distinguish the two?

**7. The Responsibility Gap**
When an AI system causes harm — a wrong medical diagnosis, a discriminatory loan decision, a car accident — who is responsible? The developers? The company? The regulator? The user? What governance structures are adequate to the accountability challenge of sophisticated AI? What would a responsible AI deployment framework look like?

**8. Your Own IAAIS**
Think about the IAAIS you are beginning to design. What problem do you most want it to solve? Who benefits from that solution — and who might be harmed or excluded? What would it mean for your system to fail? How would you know? What safeguards would you build in before you knew whether you needed them?

---

## Further Reading

### Going Deeper

The following resources are organized by theme.

#### Foundations and History

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. — The definitive textbook of the field; Chapters 1 and 2 provide authoritative historical and conceptual foundations.
- Turing, A. M. (1950). Computing machinery and intelligence. *Mind, 59*(236), 433–460. — The original paper. Read it in full — it is more nuanced and more surprising than its reputation suggests.
- McCarthy, J., Minsky, M. L., Rochester, N., & Shannon, C. E. (1955). *A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence.* — The founding document of the field.
- Crevier, D. (1993). *AI: The Tumultuous History of the Search for Artificial Intelligence.* Basic Books. — A readable and historically detailed account through the first expert systems era.

#### Intelligent Agents and Classical AI

- Russell, S., & Norvig, P. (2020). Chapters 2–6 cover intelligent agents, search, and classical AI in depth.
- Nilsson, N. J. (2010). *The Quest for Artificial Intelligence: A History of Ideas and Achievements.* Cambridge University Press. — Freely available online; provides the most thorough historical survey of the field available.

#### Deep Learning and the Modern Era

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. — Available free at [deeplearningbook.org](https://deeplearningbook.org). The standard reference for deep learning theory and practice.
- Metz, C. (2021). *Genius Makers: The Mavericks Who Brought AI to Google, Facebook, and the World.* Dutton. — Narrative journalism that tells the human story of the deep learning revolution.

#### AlphaGo and Reinforcement Learning

- Silver, D., Huang, A., Maddison, C. J., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature, 529*, 484–489. — The original AlphaGo paper; readable without deep technical background.
- *AlphaGo* (2017). Documentary film directed by Greg Kohs. — An emotionally engaging account of the Lee Sedol match. Widely available on streaming platforms.

#### Ethics and Society

- O'Neil, C. (2016). *Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy.* Crown. — Essential context for understanding how algorithmic systems can cause harm at scale.
- Noble, S. U. (2018). *Algorithms of Oppression: How Search Engines Reinforce Racism.* NYU Press. — A rigorous examination of how bias is embedded in search and recommendation systems.
- Floridi, L., et al. (2018). AI4People — An ethical framework for a good AI society. *Minds and Machines, 28*, 689–707. — A principled framework for AI ethics.
- Bostrom, N. (2014). *Superintelligence: Paths, Dangers, Strategies.* Oxford University Press. — The most influential treatment of long-term AI risk; read alongside critical responses for a balanced view.

#### Philosophy of Mind and Intelligence

- Searle, J. (1980). Minds, brains, and programs. *Behavioral and Brain Sciences, 3*(3), 417–424. — The famous Chinese Room argument against the view that computational symbol manipulation constitutes understanding.
- Hofstadter, D. R. (1979). *Gödel, Escher, Bach: An Eternal Golden Braid.* Basic Books. — One of the most remarkable books ever written about mind, intelligence, and self-reference.

---

*— End of Chapter 1 —*
