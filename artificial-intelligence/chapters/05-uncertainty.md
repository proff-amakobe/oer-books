# Chapter 5: Reasoning Under Uncertainty

**Probability, Bayesian Networks, and Calibrated Belief**

*CSC5350 · Artificial Intelligence*

---

## Opening Narrative

### The Doctor Who Changed Medicine With a Formula

In the 1960s, a cardiologist named Amos Tversky and a psychologist named Daniel Kahneman ran a series of experiments that revealed something disturbing about human reasoning under uncertainty: we are bad at it. Systematically, predictably, measurably bad. We overweight recent information and underweight base rates. We confuse the probability of a symptom given a disease with the probability of a disease given a symptom. We see patterns in random data and miss patterns in structured data.

The physician who orders a cancer test for a patient with a 1% prior probability of having the disease, receives a positive result from a test with 95% sensitivity and 90% specificity, and concludes the patient probably has cancer — is making a mistake. The probability of cancer given the positive test is not 95%. It is approximately 8.7%. Most of the positive tests are false positives, because the disease is so rare that even the small fraction of false positives among healthy patients vastly outnumbers the true positives among sick ones.

Bayes' theorem, published posthumously by Reverend Thomas Bayes in 1763, provides the exact calculation that escapes human intuition. It is not a statistical trick. It is the formal statement of how rational agents should update their beliefs in response to new evidence. And it is the foundation of a family of AI methods — Bayesian networks, hidden Markov models, particle filters — that give AI systems the ability to reason under uncertainty with calibrated, principled confidence.

> **"An agent that ignores base rates is not being intuitive — it is being wrong. Probability theory is not a constraint on natural reasoning; it is the formalization of what natural reasoning should be, but often is not."**

---

## Learning Objectives

After completing this chapter, you will be able to:

1. Apply Bayes' theorem to update beliefs given evidence, including in the presence of misleading base rates.
2. Construct and interpret joint probability distributions and use marginalization and conditioning to answer probabilistic queries.
3. Describe Bayesian networks as compact representations of joint distributions and explain how conditional independence enables their efficient construction.
4. Apply exact and approximate inference algorithms to Bayesian networks.
5. Implement a Naïve Bayes classifier and explain why the independence assumption is both wrong and often sufficient.
6. Describe Hidden Markov Models and explain the three fundamental HMM algorithms: evaluation, decoding, and learning.
7. Explain the particle filter as an approximate inference method for continuous state spaces.
8. Identify the situations where probabilistic reasoning is preferable to logical reasoning, and vice versa.
9. Build the IAAIS Uncertainty Module — a probabilistic reasoning component that maintains calibrated beliefs about domain states.

---

## Key Terminology

| Term | Plain-Language Definition |
|---|---|
| **Probability** | A measure of belief or frequency, between 0 and 1. P(A) = 1 means certainty; P(A) = 0 means impossibility; P(A) = 0.5 means maximum uncertainty. |
| **Joint Probability** | P(A, B) — the probability that both A and B are true simultaneously. The joint distribution over all variables specifies everything about a probabilistic system. |
| **Marginal Probability** | P(A) — the probability of A regardless of the value of other variables. Computed by summing the joint probability over all values of the other variables. |
| **Conditional Probability** | P(A\|B) — the probability of A given that B is known to be true. P(A\|B) = P(A, B) / P(B). |
| **Bayes' Theorem** | P(H\|E) = P(E\|H) × P(H) / P(E). Updates the prior probability P(H) of a hypothesis to the posterior P(H\|E) after observing evidence E. |
| **Prior** | P(H) — belief in a hypothesis before seeing evidence. The starting point for Bayesian updating. |
| **Likelihood** | P(E\|H) — the probability of observing the evidence if the hypothesis is true. How well the hypothesis explains the evidence. |
| **Posterior** | P(H\|E) — belief in the hypothesis after seeing evidence. What Bayes' theorem computes. |
| **Base Rate** | The prior probability of a condition in the relevant population. Ignoring base rates is one of the most common errors in probabilistic reasoning. |
| **Conditional Independence** | A is conditionally independent of B given C — written A ⊥ B \| C — if knowing C makes A and B irrelevant to each other. The foundation of Bayesian network efficiency. |
| **Bayesian Network** | A directed acyclic graph where nodes represent random variables and edges represent direct probabilistic dependencies. Each node has a conditional probability table given its parents. |
| **Conditional Probability Table (CPT)** | A table specifying P(node \| parents) for every combination of parent values. The local specification of a variable's distribution in a Bayesian network. |
| **Inference (Probabilistic)** | Computing a probability query P(X \| evidence) from a probabilistic model. Can be exact (enumeration, variable elimination) or approximate (sampling). |
| **Naïve Bayes** | A classification model assuming all features are conditionally independent given the class. Despite this (usually wrong) assumption, performs well on text classification, medical diagnosis, and spam detection. |
| **Hidden Markov Model (HMM)** | A probabilistic model where the system transitions between hidden states, with each state producing observable outputs. Used for speech recognition, protein sequence analysis, and temporal pattern recognition. |
| **Emission Probability** | P(observation \| hidden state) — the probability of observing a particular output given the current hidden state. |
| **Transition Probability** | P(next state \| current state) — the probability of moving from one hidden state to another. |
| **Viterbi Algorithm** | Dynamic programming algorithm for HMMs that finds the most likely sequence of hidden states given a sequence of observations. |
| **Particle Filter** | An approximate inference algorithm for systems with continuous states. Maintains a set of weighted samples (particles) that approximate the probability distribution over states. |
| **Calibration** | A model is well-calibrated if its stated probabilities match empirical frequencies — when it says 70% confident, it is right 70% of the time. Calibration is essential for high-stakes decision support. |

---

## Section 1 — Why Uncertainty Is Unavoidable

Classical AI — logic, STRIPS planning, expert system rules — assumes that the agent has complete and certain knowledge of the world. In practice, agents face uncertainty from multiple sources simultaneously.

**Sensor noise:** Measurements are imperfect. A thermometer reads 37.9°C, but the patient's true temperature could be anywhere from 37.4 to 38.4°C depending on measurement error. A medical imaging system sees patterns in pixel data; the underlying tissue structure is inferred, not directly observed.

**Partial observability:** The agent cannot observe everything relevant. A physician cannot directly observe bacterial cultures growing in a patient's bloodstream — they must infer the presence of bacteria from symptoms, lab results, and prior probabilities.

**Stochastic actions:** The same action, in the same state, does not always produce the same outcome. Medication A cures 70% of patients with condition X; the other 30% do not respond. The planning formalism of Chapter 4 cannot represent this — STRIPS assumes deterministic outcomes.

**Model approximation:** Every model is wrong. The probabilities in a Bayesian network are estimated from data and expert judgment, not perfect measurements. Reasoning under uncertainty requires acknowledging the uncertainty in the model itself.

The formal language for reasoning under all these sources of uncertainty is probability theory — the only mathematically consistent framework for representing and manipulating degrees of belief.

---

## Section 2 — The Foundations of Probabilistic Reasoning

### The Three Rules of Probability

Three rules, and their consequences, underlie all of probabilistic AI.

**Product rule:** P(A, B) = P(A\|B) × P(B) = P(B\|A) × P(A). The joint probability equals the conditional times the marginal. This rule is how Bayesian networks factorize the joint distribution.

**Sum rule:** P(A) = Σ_B P(A, B). The marginal probability of A is obtained by summing the joint probability over all values of B — "integrating out" or "marginalizing" B. This is how we answer queries about single variables from joint models.

**Bayes' theorem** follows directly from the product rule. Rewriting P(A\|B) = P(B\|A) × P(A) / P(B) gives us the most important equation in probabilistic AI:

**P(H\|E) = P(E\|H) × P(H) / P(E)**

The prior P(H) represents our belief before seeing evidence. The likelihood P(E\|H) represents how probable the evidence is if the hypothesis holds. The posterior P(H\|E) represents our updated belief after seeing the evidence. The normalizing constant P(E) ensures probabilities sum to 1.

### The Cancer Screening Example

A screening test for a rare cancer has sensitivity 0.95 (true positive rate) and specificity 0.90 (true negative rate). The cancer's prevalence in the screened population is 0.01 (1%).

```
Applying Bayes' theorem:
  Prior: P(Cancer) = 0.01

  P(Test+ | Cancer) = 0.95  (sensitivity)
  P(Test+ | No Cancer) = 0.10  (1 - specificity)

  P(Test+) = P(Test+|Cancer)×P(Cancer) + P(Test+|NoCancer)×P(NoCancer)
           = 0.95 × 0.01 + 0.10 × 0.99
           = 0.0095 + 0.099 = 0.1085

  P(Cancer | Test+) = P(Test+|Cancer) × P(Cancer) / P(Test+)
                    = 0.95 × 0.01 / 0.1085
                    = 0.0095 / 0.1085
                    ≈ 0.0876  →  8.76%
```

A positive test on a population with 1% prevalence gives only an 8.76% probability of cancer — despite a 95% sensitive test. The vast majority of positive tests are false positives from the large healthy population. This is the **base rate neglect** failure mode: focusing on the test's sensitivity while ignoring the rarity of the condition.

This calculation has direct clinical implications. Screening programs for rare conditions need careful analysis of the Bayesian consequences of positive results — because the actions triggered by positive tests (follow-up procedures, patient anxiety, costs) must be weighed against the low posterior probability of true disease.

---

## Section 3 — Bayesian Networks: Compact Models of Uncertain Worlds

A full joint distribution over n binary variables requires 2^n numbers — 2^30 ≈ 10^9 for just 30 variables. Bayesian networks make probabilistic reasoning tractable by exploiting **conditional independence**: when two variables are independent given a third, they need not be explicitly modeled together.

### Structure and Semantics

A Bayesian network is a directed acyclic graph where:
- Each node represents a random variable
- Each edge X → Y represents a direct probabilistic influence of X on Y
- Each node has a **conditional probability table (CPT)** giving P(node\|parents)

The joint probability of any complete assignment of values to all variables factorizes as the product of each variable's CPT value:

**P(X₁, X₂, ..., Xₙ) = Π P(Xᵢ | Parents(Xᵢ))**

This factorization is the key: instead of specifying 2^n numbers, we specify only the CPTs — one per variable, with size exponential only in the number of *parents*, not all variables. For networks with sparse parent relationships (as most real domains have), this is a dramatic compression.

```
# A small clinical Bayesian network for sepsis diagnosis
#
# Structure (arrows indicate direct causal influence):
#
#  Immunocompromised → InfectionPresent → BacteriaInBlood
#  InfectionPresent  → Fever
#  Fever             → WBC_Elevated
#  BacteriaInBlood   → BloodCulturePositive
#  InfectionPresent  → BloodCulturePositive

# Conditional probability tables (representative values):

P(Immunocompromised = True) = 0.15

P(InfectionPresent = T | Immunocompromised = T) = 0.40
P(InfectionPresent = T | Immunocompromised = F) = 0.10

P(Fever = T | InfectionPresent = T) = 0.85
P(Fever = T | InfectionPresent = F) = 0.05

P(WBC_Elevated = T | Fever = T) = 0.80
P(WBC_Elevated = T | Fever = F) = 0.20

P(BloodCulturePositive = T | BacteriaInBlood = T) = 0.90
P(BloodCulturePositive = T | BacteriaInBlood = F) = 0.05

# Query: P(InfectionPresent | Fever=T, WBC_Elevated=T)
# Exact inference via variable elimination gives ≈ 0.72
# The probability of infection given both symptoms is 72%.
```

### Inference in Bayesian Networks

**Exact inference** computes query probabilities precisely. **Variable elimination** processes variables in an order that minimizes intermediate computation, eliminating variables one by one through marginalization. **Belief propagation** (message passing) efficiently distributes information through the network for tree-structured and polytree networks.

For complex, densely connected networks, exact inference is NP-hard. **Approximate inference** through sampling (MCMC, likelihood weighting, rejection sampling) provides accurate estimates when exact computation is infeasible.

---

## Section 4 — Naïve Bayes: Simplicity That Works

The **Naïve Bayes** classifier applies Bayes' theorem to classification by assuming that all features are conditionally independent given the class:

**P(class\|features) ∝ P(class) × Π P(feature_i\|class)**

This independence assumption is almost always wrong — the presence of fever and the presence of elevated white blood cells are not independent given infection status. Yet Naïve Bayes works remarkably well in practice. Why?

Because classification only requires ordering classes by probability, not computing exact probabilities. As long as the independence assumption doesn't systematically mislead the ordering — as long as the feature correlations affect all classes similarly — the classification boundary may still be correct even when the probabilities are miscalibrated.

Naïve Bayes classifiers are fast to train (linear in the number of training examples), fast to predict (linear in the number of features), and require surprisingly little data. They remain competitive with much more sophisticated classifiers on text classification and spam filtering — domains where the feature space is large but the independence assumption is approximately satisfied.

---

## Section 5 — Hidden Markov Models: Reasoning Through Time

A **Hidden Markov Model** represents a system that transitions between unobservable hidden states, with each state producing observable outputs according to emission probabilities. The "Markov" property ensures the next hidden state depends only on the current hidden state.

HMMs answer three fundamental questions:

**Evaluation:** What is the probability of an observed sequence given the model? This is solved by the forward algorithm, which computes the probability of each observation sequence by summing over all possible hidden state sequences.

**Decoding:** What is the most likely sequence of hidden states that produced the observed sequence? This is solved by the Viterbi algorithm — dynamic programming that efficiently finds the maximum-probability path through the state sequence.

**Learning:** Given a set of observations (but not the hidden states), how should the model parameters (transition and emission probabilities) be adjusted to maximize the probability of the observations? This is solved by the Baum-Welch algorithm (a special case of expectation-maximization).

Applications of HMMs pervade modern AI. **Speech recognition** models speech as a sequence of phoneme states (hidden) producing acoustic observations (visible). **Gene finding** models DNA as a sequence of coding and non-coding states (hidden) producing nucleotide sequences (visible). **Financial modeling** represents market regimes (hidden) producing returns and volatility (visible).

---

## Section 6 — Particle Filters: Uncertainty in Continuous Space

When the state space is continuous — a robot's position in a room, a patient's blood glucose level over time — exact Bayesian inference becomes intractable. **Particle filters** provide an approximate solution: represent the probability distribution as a set of weighted samples (particles), each representing a hypothesis about the current state.

At each time step, the particle filter performs three operations:

**Prediction:** Each particle is propagated forward through the transition model with added noise, representing the uncertainty in how the state evolves.

**Update:** Each particle is weighted by the likelihood of the current observation given that particle's state hypothesis. Particles inconsistent with the observation get low weight.

**Resampling:** Particles are resampled proportional to their weights, concentrating the particle population in regions of high probability.

The result is a set of particles that approximately tracks the evolving probability distribution over the hidden state — even in continuous spaces where exact computation is impossible.

Robot localization — determining a robot's position from sensor readings — is the canonical application. A robot begins with uniform uncertainty over its location (particles spread across the entire floor plan). As it moves and receives sensor readings, the particle cloud concentrates around the true location.

---

## Section 7 — IAAIS Integration: The Uncertainty Module

This week you add the **IAAIS Uncertainty Module** — a probabilistic reasoning component that maintains calibrated beliefs about states that cannot be directly observed.

The Uncertainty Module connects to the Knowledge Base: it reads deterministic facts and updates them with probabilistic beliefs. When the Classifier (Chapter 6) produces a prediction with confidence 0.73, the Uncertainty Module stores not just the prediction but the full probability distribution. When the Sensor Module receives noisy measurements, the Uncertainty Module maintains the posterior distribution over the true value.

The module's three core functions:

**Belief update:** Given new evidence, apply Bayes' theorem to update the posterior distribution. For categorical variables, use exact Bayesian updating. For continuous variables or complex dependency structures, use particle filtering.

**Uncertainty propagation:** When the Planner (Chapter 4) requires expected values for planning under uncertainty, the Uncertainty Module provides them — computing expected utilities by weighting outcomes by their probabilities.

**Calibration reporting:** For high-stakes decisions, the module reports not just the most probable conclusion but the full posterior distribution and the strength of the evidence — enabling clinicians or users to weigh uncertainty explicitly.

| Chapter | Module | Capability |
|---|---|---|
| Ch 2 | Search Engine | Path planning |
| Ch 3 | Knowledge Base | Structured facts and inference |
| Ch 4 | Planner | Goal-directed action sequences |
| Ch 5 | Uncertainty Module | Calibrated probabilistic beliefs |

---

## Hands-On Exploration: Bayesian Diagnosis

### The Activity

Open `hands_on_ch5.ipynb` from the course repository.

**Part 1 — Bayes' Theorem Calculator (15 minutes):** Implement a Bayes' theorem calculator. Use it to analyze three medical screening scenarios with different sensitivity, specificity, and prevalence values. Plot a graph showing how posterior probability varies with prevalence for a fixed test performance. At what prevalence does screening become informative (posterior > 50%)?

**Part 2 — Naïve Bayes Classifier (20 minutes):** Train a Naïve Bayes classifier on the provided symptom dataset (patients described by 15 binary symptoms; labels are diagnostic categories). Evaluate accuracy, precision, and recall on a held-out test set. Then examine the learned conditional probabilities — do they match your domain intuitions? Which features are most discriminative?

**Part 3 — Bayesian Network Inference (20 minutes):** Using the `pgmpy` library, construct the sepsis Bayesian network from Section 3. Compute: P(InfectionPresent\|Fever=T, WBC=T), P(InfectionPresent\|Fever=T, WBC=T, BloodCulture=T). How does adding the blood culture result change the posterior? At what blood culture sensitivity would you change the treatment decision?

### Reflection Questions

1. In the cancer screening example, a positive result gives only 8.76% probability of cancer. Would you disclose this to a patient before testing? How would you explain it? What obligation does a physician have to reason probabilistically on a patient's behalf?
2. The Naïve Bayes classifier assumes feature independence. For your domain, identify two features that are clearly not independent. How would you modify the model to capture this dependency, and what would be the cost in data requirements and computation?
3. A Bayesian network requires specifying conditional probability tables. Where do these numbers come from? Describe two different elicitation methods and their tradeoffs.
4. Your IAAIS Uncertainty Module must communicate uncertainty to users. How would you present a posterior probability of 0.67 for a medical diagnosis to (a) a physician, (b) a patient, (c) a hospital administrator reviewing outcomes? Should the same number be presented differently to different audiences?

---

## Case Study: Bayesian Networks in Medical Diagnosis — PathFinder

### The System

PathFinder was a Bayesian network for diagnosing lymph node pathology, developed at Stanford Medical School in the late 1980s by David Heckerman and colleagues. It contained 60 diseases, 130 symptoms and test results, and approximately 8,000 conditional probability estimates — each elicited through careful interviews with pathology experts.

### The Validation

To evaluate PathFinder, Heckerman ran an experiment that is now famous in AI circles. He gave 25 difficult lymph node cases to PathFinder and to Dr. Hanna Kouprine, a world-renowned expert in lymph node pathology from Germany. PathFinder matched or outperformed Dr. Kouprine on 24 of 25 cases.

The result was striking — but its interpretation requires care. PathFinder was not more knowledgeable than Dr. Kouprine. It could not examine a specimen directly. It could not notice the subtle visual features that experts detect. What it could do was maintain probability distributions over 60 diseases simultaneously, correctly apply Bayes' theorem to combine evidence from 130 features, and avoid the cognitive biases (anchoring, availability heuristic, base rate neglect) that affect even the best human diagnosticians.

### The Lesson

PathFinder illustrated a recurring theme in AI-assisted diagnosis: AI systems can outperform experts not by knowing more, but by *computing more carefully*. The bottleneck in complex probabilistic reasoning is often not knowledge but computation — specifically, the human inability to simultaneously track dozens of hypotheses and correctly weight evidence for each.

The lesson is also cautionary. PathFinder's conditional probabilities were elicited from a small number of experts at one institution. When deployed in different settings with different patient populations and different prior probabilities, the model needed recalibration. A Bayesian network is a model — and like all models, it is wrong in proportion to how well it was validated on the deployment population.

---

## Chapter Summary

We began this chapter with a calculation that eludes human intuition: how to correctly update beliefs about disease probability when a test comes back positive. Bayes' theorem provides the exact answer, and AI systems that apply it correctly can outperform human reasoning on exactly the class of problems where human cognition consistently fails.

Probability theory gave us the foundational rules: the product rule expressing how joint probabilities decompose, the sum rule enabling marginalization, and Bayes' theorem providing the update equation from prior through likelihood to posterior. The cancer screening example showed how dramatically the base rate changes the meaning of a positive test result.

Bayesian networks gave us the ability to represent complex uncertain domains compactly, exploiting conditional independence to avoid the exponential explosion of full joint distributions. Naïve Bayes showed that a simplifying assumption (feature independence) can be simultaneously wrong and practically effective.

Hidden Markov Models extended probabilistic reasoning to temporal sequences — modeling systems that transition through hidden states over time, with applications in speech recognition, genomics, and financial modeling. Particle filters extended this to continuous state spaces through approximate sampling-based inference.

In Chapter 6, we move from reasoning about uncertainty to *learning from data* — using the observed patterns in labeled examples to build classifiers that can predict on new cases.

---

## Discussion Questions

1. **Base rates and clinical intuition:** Experienced physicians often do not consciously apply Bayes' theorem, yet some develop accurate intuitions about posterior probabilities. How might this happen? And why might those intuitions fail in novel situations or with unfamiliar tests?
2. **Prior selection:** Bayesian inference requires a prior. Where should priors come from for a clinical AI system? What are the risks of using a prior derived from one hospital's patient population when deploying at another?
3. **Calibration vs. discrimination:** A model can be well-calibrated (probabilities match frequencies) but poor at distinguishing cases (AUC ≈ 0.6). It can also discriminate well (AUC ≈ 0.9) but be badly miscalibrated. Which property matters more for a clinical decision support system, and why?
4. **Approximate inference trade-offs:** Variable elimination gives exact answers but is NP-hard. Particle filters give approximate answers but scale to large, continuous state spaces. Design a scenario where the inaccuracy of particle filter approximation could cause harm — and describe what safeguards you would implement.
5. **HMMs and temporal reasoning:** Design an HMM for a patient monitoring scenario in your domain. What are the hidden states? What are the observable outputs? How would you estimate the transition and emission probabilities from historical data?
6. **Uncertainty communication:** A Bayesian network reports P(Sepsis\|evidence) = 0.73 for a patient. How should this be displayed to the nurse at the bedside? To the attending physician? To the family? Is there a level of posterior probability below which the system should stay silent?
7. **Independence assumptions in practice:** The Naïve Bayes assumption fails when features are correlated. In natural language processing, the words in a document are clearly not independent. Yet Naïve Bayes is still competitive for text classification. How do you reconcile this?
8. **Your IAAIS Uncertainty Module:** Identify three uncertain quantities in your IAAIS domain that require probabilistic representation. For each: describe the prior, the evidence that would update the belief, and the action triggered by different posterior values.

---

## Further Reading

### Probability and Bayes

Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann. The foundational text — introduced Bayesian networks to AI.

Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press. The definitive exposition of Bayesian probability theory.

### Bayesian Networks

Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press. The comprehensive modern reference.

### Applications

Heckerman, D., Breese, J., & Rommelse, K. (1995). Decision-theoretic troubleshooting. *Communications of the ACM*, 38(3), 49–57. PathFinder and applied Bayesian networks in medicine.

---

*— End of Chapter 5 —*
