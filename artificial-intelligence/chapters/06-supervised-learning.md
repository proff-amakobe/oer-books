# Learning From Examples

**Supervised Machine Learning — From Data to Decisions**

*CSC5350 · Artificial Intelligence*

---

## Opening Narrative

### The Million-Dollar Mistake That Changed Everything

In October 2006, Netflix offered a prize of one million dollars to anyone who could improve its movie recommendation algorithm by 10%. The company released 100 million ratings from 480,000 customers across 17,000 movies — stripped of identifying information — and waited to see what the machine learning community could do.

What followed over the next three years was an extraordinary experiment in applied supervised learning. Teams from around the world tried every technique available: matrix factorization, nearest neighbors, gradient boosting, neural networks, and increasingly exotic ensembles. The winning submission, submitted in 2009, combined over 100 different algorithms — models that individually achieved 2-4% improvements, blended into an ensemble that crossed the 10% threshold.

The Netflix Prize transformed machine learning from an academic discipline into an industrial one. It demonstrated three things that are now axiomatic: that sufficiently large labeled datasets enable surprising performance from relatively simple models; that ensemble methods consistently outperform their components; and that the gap between theory and practice is bridged by careful evaluation on held-out data, not by mathematical elegance.

But the prize also concealed a cautionary note. When Netflix examined the winning algorithm more carefully, they concluded it was too computationally expensive to deploy at scale. The model that won the competition was never used in production. The gap between benchmark performance and real-world utility would become one of the defining tensions of applied machine learning.

> **"A model that achieves 94% accuracy on a benchmark and fails in deployment is not a success. A model that achieves 87% accuracy and actually improves user experience is. The evaluation matters as much as the algorithm."**

---

## Learning Objectives

After completing this chapter, you will be able to:

1. Define the supervised learning problem formally and distinguish classification from regression.
2. Explain the bias-variance tradeoff and use it to diagnose underfitting and overfitting.
3. Implement decision tree learning and explain how information gain guides split selection.
4. Describe random forests and explain how bagging and feature randomization produce ensembles superior to their components.
5. Explain gradient boosting and describe why sequential error correction outperforms parallel averaging.
6. Design a model evaluation pipeline: train/validation/test splits, cross-validation, appropriate metrics.
7. Interpret precision, recall, F1, AUC-ROC, and confusion matrices and select the appropriate metric for a given application.
8. Apply feature engineering, normalization, and handling of missing values to real datasets.
9. Build the IAAIS Classifier — a supervised learning component that predicts domain-specific labels from structured features.

---

## Key Terminology

| Term | Plain-Language Definition |
|---|---|
| **Supervised Learning** | Learning a mapping from inputs to outputs using labeled training examples — pairs of (input, correct output). The supervision comes from the labels provided by human experts. |
| **Classification** | Supervised learning where the output is a discrete category label. Binary classification (two classes); multi-class (more than two). |
| **Regression** | Supervised learning where the output is a continuous numeric value. Predicting a patient's blood pressure, a stock's price, or a building's energy consumption. |
| **Training Set** | Labeled data used to fit the model's parameters. The model is explicitly optimized on this data. |
| **Validation Set** | Held-out data used to tune hyperparameters and select the best model. Not used during training. |
| **Test Set** | Held-out data used only once, after all modeling decisions are made, to estimate real-world performance. Using it for model selection invalidates its estimate. |
| **Bias** | The error from incorrect assumptions in the learning algorithm. High bias models underfit — they are too simple to capture the true patterns. |
| **Variance** | The error from sensitivity to fluctuations in the training set. High variance models overfit — they memorize training data and fail to generalize. |
| **Overfitting** | When a model learns the noise in training data rather than the underlying pattern, achieving low training error but high test error. The primary failure mode of powerful models. |
| **Underfitting** | When a model is too simple to capture the patterns in the data, achieving high error on both training and test sets. |
| **Regularization** | Techniques that reduce overfitting by penalizing model complexity. L1 (Lasso) promotes sparsity; L2 (Ridge) penalizes large weights; dropout randomly deactivates neurons during training. |
| **Cross-Validation** | Evaluating a model by training on multiple different training/validation splits and averaging the results. More reliable than a single split, especially on small datasets. |
| **Decision Tree** | A hierarchical sequence of if-then-else questions about feature values, leading to a classification or regression prediction at each leaf. Highly interpretable. |
| **Information Gain** | The reduction in entropy achieved by splitting on a given feature. The measure used by most decision tree algorithms to select the best split at each node. |
| **Entropy** | A measure of impurity or uncertainty in a set. A perfectly pure set (all one class) has entropy 0; a maximally uncertain set (equal class distribution) has maximum entropy. |
| **Bagging** | Bootstrap Aggregating. Train multiple models on random subsamples of the training data (with replacement) and average their predictions. Reduces variance without increasing bias. |
| **Random Forest** | An ensemble of decision trees, each trained on a random subsample of data and a random subset of features. The combination of tree diversity and averaging produces models highly robust to overfitting. |
| **Boosting** | An ensemble method that trains models sequentially, with each new model focusing on the examples the previous models got wrong. Reduces bias as well as variance. |
| **Gradient Boosting** | Boosting where each new model fits the negative gradient of the loss function on the current ensemble's residuals. XGBoost and LightGBM are highly optimized implementations. |
| **Precision** | Of all instances predicted positive, what fraction were truly positive? Measures the purity of positive predictions. |
| **Recall** | Of all truly positive instances, what fraction were correctly predicted positive? Measures the completeness of positive predictions. |
| **F1 Score** | The harmonic mean of precision and recall: 2×(P×R)/(P+R). Balances both measures; useful when classes are imbalanced. |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve. Measures overall classifier performance across all thresholds. AUC=1 is perfect; AUC=0.5 is random. |
| **Feature Engineering** | The process of transforming raw data into features that machine learning algorithms can use effectively. Often more impactful than algorithm selection. |

---

## Section 1 — The Supervised Learning Framework

Supervised learning is the most commercially deployed branch of machine learning. Every spam filter, fraud detector, medical diagnosis assistant, and credit scoring system at scale is a supervised learning model.

The formal setup: we have a training set of n examples {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)} where each xᵢ is a feature vector and each yᵢ is the corresponding label. We want to learn a function f such that f(x) ≈ y for new, unseen examples — not just the training examples.

This last requirement — generalization to unseen examples — is what makes supervised learning non-trivial. Any model complex enough can memorize the training set exactly; the challenge is capturing the underlying pattern without capturing the noise.

### The Bias-Variance Tradeoff

Every model lives on a spectrum between two failure modes. Consider the problem of predicting whether a patient will be readmitted to hospital within 30 days.

A model with **high bias** — say, a rule that predicts readmission only if age > 75 — is too simple. It misses patterns involving diagnosis, medications, prior hospitalizations, and social factors. It has high training error and high test error: it underfits.

A model with **high variance** — say, a decision tree with no depth limit that memorizes every training patient — is too specific to the training data. It perfectly predicts training patients but fails on new ones because it has learned idiosyncrasies (this 68-year-old with these exact lab values was readmitted) rather than generalizable patterns (patients discharged with unresolved infections tend to return). It has low training error but high test error: it overfits.

The goal is the Goldilocks model: enough complexity to capture the true pattern, not so much that it captures noise. Regularization, cross-validation, and ensemble methods are the tools for finding this sweet spot.

---

## Section 2 — Decision Trees: Interpretable Learning

A decision tree learns a hierarchical series of questions about features that progressively narrow the prediction. Starting at the root, the tree asks: "Is the patient's age above 65?" If yes, go right and ask "Was there a prior hospitalization in the last 6 months?" The tree continues until a leaf node is reached, which provides the prediction.

### Growing a Tree: Information Gain

The key question in building a decision tree is: which feature to split on at each node? The answer comes from **information gain** — the reduction in entropy (uncertainty) achieved by the split.

**Entropy** measures impurity:
H(S) = -Σ p_i log₂(p_i)

A set of 100 patients, 50 of whom were readmitted and 50 of whom were not, has maximum entropy: H = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1 bit. A perfectly pure set (all readmitted or none) has entropy 0.

**Information gain** of splitting set S on feature A:
IG(S, A) = H(S) - Σ (|S_v| / |S|) × H(S_v)

where S_v is the subset where feature A has value v. The algorithm greedily selects the feature with the highest information gain at each node.

Trees grown to full depth overfit severely. **Pruning** — cutting back branches that do not improve validation set performance — is essential. Alternatively, depth limits or minimum-sample constraints prevent the tree from specializing too finely.

Decision trees have one irreplaceable advantage: interpretability. A physician can follow a decision tree's reasoning step by step. In regulated environments where decisions must be explained, this is not a minor advantage — it may be a legal requirement.

---

## Section 3 — Ensemble Methods: Strength in Diversity

The central insight of ensemble methods: diverse models that fail differently can be combined into a stronger model. If three classifiers each have 10% error rate but their errors occur on different examples, a majority vote achieves much lower error than any individual.

### Random Forests: Parallel Diversity

**Random forests** build many decision trees, each on a random bootstrap sample of the training data (with replacement) and using a random subset of features at each split. The final prediction is the majority vote (classification) or average (regression) of all trees.

Two sources of diversity prevent the trees from all making the same mistakes:

**Bagging:** Each tree sees a different sample of the data. Examples not selected for a tree's training set form its "out-of-bag" set, enabling free validation without a separate validation split.

**Feature randomness:** At each split, only a random subset of features is considered. This forces different trees to find different features useful, creating genuine diversity.

The result is an ensemble that is dramatically more robust to overfitting than a single deep tree while often outperforming any single model. Random forests are among the most reliable algorithms in practice: they rarely catastrophically fail, require minimal hyperparameter tuning, provide built-in feature importance estimates, and handle missing values and mixed feature types gracefully.

### Gradient Boosting: Sequential Error Correction

While random forests build trees in parallel, **gradient boosting** builds them sequentially, with each tree correcting the errors of the ensemble so far.

The key idea: fit a new tree to the *residuals* — the differences between the current predictions and the true labels. Add the new tree to the ensemble with a small learning rate (step size). Repeat. Each iteration reduces the ensemble's error on the training data.

Framed as gradient descent, each tree fits the negative gradient of the loss function. For squared error regression, the gradient is exactly the residual. For classification with log loss, the gradient is a function of predicted probabilities and true labels. This framing enables gradient boosting to optimize almost any differentiable loss function by simply computing its gradient.

Modern implementations — **XGBoost**, **LightGBM**, **CatBoost** — add regularization, efficient tree construction, and parallelization. They consistently achieve state-of-the-art performance on structured (tabular) data and dominate competitions like Kaggle. The dominant practical wisdom: for tabular data, try gradient boosting first.

---

## Section 4 — Model Evaluation: Measuring What Matters

A model is only as valuable as its evaluation is honest. The most common mistake in applied machine learning is optimizing for the wrong metric — or computing the right metric incorrectly.

### Splits and Cross-Validation

The golden rule: the test set must never influence any decision. Once data is used to tune a model — even to decide between two algorithms — it is no longer an honest test.

**Hold-out validation:** Split data into 70% training, 15% validation, 15% test. Use training to fit models, validation to select hyperparameters, and test only once for the final estimate.

**K-fold cross-validation:** For small datasets where a single validation split is too noisy, partition the data into K folds. Train K models, each time holding out one fold as validation. Average the K validation scores. This uses all data for training (across folds) while providing a more stable performance estimate.

**Nested cross-validation:** For complete hyperparameter optimization with limited data, use an outer loop for test evaluation and an inner loop for hyperparameter selection. Computationally expensive but unbiased.

### Choosing the Right Metric

The choice of metric encodes your values about which errors are costly.

For **imbalanced classes** (e.g., fraud detection where 99.9% of transactions are legitimate), accuracy is misleading: a model that predicts "not fraud" for every transaction achieves 99.9% accuracy while being useless. Use F1 score, precision-recall AUC, or cost-sensitive metrics instead.

For **asymmetric costs** (e.g., medical screening where false negatives are far more costly than false positives), tune the classification threshold. Moving the threshold lower than 0.5 reduces false negatives at the cost of more false positives. The operating point should be chosen based on the relative costs.

| Task | Inappropriate metric | Better metric |
|---|---|---|
| Rare disease screening | Accuracy | Recall, F1 |
| Fraud detection | Accuracy | Precision-Recall AUC |
| Patient risk stratification | AUC-ROC | Decision curve analysis |
| Readmission prediction | F1 | Net benefit at decision threshold |

---

## Section 5 — Feature Engineering: Where Expertise Becomes Signal

Machine learning algorithms learn patterns from features — the numerical representations of raw data. The quality of features often matters more than the choice of algorithm.

**Normalization:** Many algorithms are sensitive to the scale of features. StandardScaler (zero mean, unit variance) or MinMaxScaler (scale to [0,1]) prevents features with large numeric ranges from dominating features with small ranges.

**Categorical encoding:** Algorithms require numbers. Ordinal categories (low/medium/high) can be encoded as integers. Nominal categories (blood type A/B/AB/O) require one-hot encoding — a separate binary feature for each category — to avoid implying ordering.

**Missing values:** Missing data is the norm, not the exception. Simple strategies: impute with mean, median, or mode. Better strategies: use a model to predict missing values, or treat missingness itself as a feature (the fact that a value is missing may be informative).

**Domain-specific features:** The features that best discriminate often require domain knowledge. A readmission risk model benefits enormously from a "number of prior hospitalizations in the last 12 months" feature — obvious to a clinician, invisible to an algorithm looking at raw admission records. This is where practitioners add most of their value: translating domain expertise into signal.

---

## Section 6 — IAAIS Integration: The Classifier

This week you add the **IAAIS Classifier** — a supervised learning component that maps structured feature vectors to domain-specific predictions.

The Classifier connects to the Knowledge Base (reading facts to use as features), the Uncertainty Module (reporting probabilities rather than hard labels), and the Expert Module (providing probabilistic outputs that expert rules can refine). It also writes predictions back to the Knowledge Base, where they become facts available to all other modules.

**Design decisions this week:**
- What is the classification task? (readmission, diagnosis, anomaly, intent?)
- What features are available, and how should they be engineered?
- How will you handle class imbalance in your domain?
- Which metric should be optimized, and why?
- What level of interpretability does the deployment context require?

| Chapter | Module | Capability |
|---|---|---|
| Ch 2 | Search Engine | Path planning |
| Ch 3 | Knowledge Base | Structured facts and inference |
| Ch 4 | Planner | Goal-directed action sequences |
| Ch 5 | Uncertainty Module | Calibrated probabilistic beliefs |
| Ch 6 | Classifier | Supervised prediction |

---

## Hands-On Exploration: Building a Readmission Risk Classifier

### The Activity

Open `hands_on_ch6.ipynb` from the course repository. The notebook contains a de-identified dataset of 1,500 hospital discharge records, each described by 25 features (demographics, diagnoses, procedures, lab values, prior visits) and a 30-day readmission label.

**Part 1 — Exploratory Analysis (15 minutes):** Compute class balance, missing value rates, and feature distributions. Identify the three features most correlated with readmission. Plot the distribution of these features for readmitted vs. non-readmitted patients.

**Part 2 — Modeling Pipeline (25 minutes):** Build and compare: (a) Logistic Regression baseline, (b) Random Forest, (c) Gradient Boosting (XGBoost). Use 5-fold cross-validation for evaluation. Report accuracy, precision, recall, F1, and AUC-ROC for each. Plot the ROC curves on the same axes.

**Part 3 — Feature Importance and Threshold Tuning (15 minutes):** Extract feature importance from your best model. Are the most important features clinically plausible? Select the operating threshold that maximizes recall while maintaining at least 60% precision (simulating a screening scenario where missing readmissions is costly). How many additional interventions would this threshold trigger?

### Reflection Questions

1. If 85% of patients are not readmitted, what accuracy would a model that always predicts "not readmitted" achieve? Why is this useless, and what metric would better capture the model's value?
2. Your gradient boosting model outperforms your logistic regression by 4% AUC. A physician asks you to explain why the model predicted high risk for a specific patient. How would you answer?
3. Suppose you deploy the readmission model and discover it performs 8% worse on patients over 85 than on younger patients. What could cause this, and what would you do?
4. The 30-day readmission label in your training data reflects past clinical decisions as much as patient health. A patient who received a follow-up call from a care coordinator was less likely to be readmitted — but this intervention doesn't appear in the features. How does this bias your model, and how would you address it?

---

## Case Study: The Netflix Prize — What a Competition Taught Machine Learning

### The Benchmark Trap

The Netflix Prize created one of the most valuable labeled datasets ever assembled for machine learning. It also created one of the field's most important cautionary tales.

The winning Pragmatic Chaos ensemble combined the outputs of 107 different algorithms, including matrix factorization variants, neighborhood-based methods, and restricted Boltzmann machines. On the benchmark — predicting ratings from the released dataset — it achieved a 10.06% improvement over Netflix's existing algorithm, crossing the 1% threshold that unlocked the million-dollar prize.

Netflix never deployed it. The engineering complexity of maintaining 107 models in production, combined with the latency requirements of real-time recommendation, made the winning approach impractical. Netflix instead implemented a much simpler model that achieved perhaps a 7% improvement — and ran reliably on millions of users.

### The Gap Between Competition and Reality

The Netflix Prize illustrates what is now called the **benchmark trap**: optimizing for a fixed metric on a fixed dataset can produce models that are excellent benchmarks but poor products. Real deployment requires:

- **Computational efficiency:** Models that run in milliseconds under production load
- **Robustness to distribution shift:** Performance when users and content change over time
- **Cold-start handling:** What to do for new users or new content with no ratings
- **Explainability:** Why is this movie being recommended?
- **Fairness:** Are some user groups receiving systematically worse recommendations?

The gap between the million-dollar benchmark model and the deployed model is a version of the gap between validation accuracy and real-world utility that appears in every production ML system. Closing that gap requires engineering as much as machine learning.

### The Legacy

Despite never being used, the Netflix Prize transformed machine learning in three ways. It demonstrated that collaborative filtering with latent factors could achieve dramatically better recommendations than content-based or nearest-neighbor methods. It pioneered the use of large-scale benchmark datasets as community-wide research drivers. And it revealed that ensemble methods — combining many diverse models — almost always outperform any individual model, a finding that has proven robust across decades of subsequent research.

---

## Chapter Summary

We began this chapter with the Netflix Prize — a million-dollar competition that transformed machine learning into an industrial discipline and revealed, at its conclusion, the gap between benchmark performance and deployment reality.

The supervised learning framework gave us the formal setup: labeled training examples, a learning algorithm, and the critical requirement of generalization — performing well on examples not seen during training. The bias-variance tradeoff gave us the diagnostic framework: underfitting models are too simple; overfitting models are too specific; the goal is the appropriate level of complexity for the available data.

Decision trees gave us interpretable learning through information gain and recursive splitting — the baseline of supervised learning for tabular data, with the crucial property that their reasoning can be followed and explained. Random forests showed how bagging and feature randomness create diverse ensembles that dramatically outperform individual trees. Gradient boosting showed how sequential error correction can reduce both bias and variance simultaneously.

Model evaluation demanded honest discipline: separate test sets, appropriate metrics for the application, and resistance to the many ways practitioners inadvertently cheat themselves into overly optimistic estimates. Feature engineering demonstrated that domain knowledge encoded as well-chosen features often matters more than algorithm selection.

In Chapter 7, we remove the labels — and ask whether machines can discover structure in data without being told what to look for.

---

## Discussion Questions

1. **The benchmark trap:** Design an evaluation protocol for a medical AI system that avoids the Netflix Prize failure — ensuring that benchmark performance predicts deployment performance. What data would you need, what metrics would you compute, and what validation steps would precede deployment?
2. **Fairness and metrics:** A readmission risk model achieves 82% AUC overall. It achieves 85% AUC for patients under 65 and 74% AUC for patients over 80. Should the model be deployed? What options do you have to address the disparity?
3. **Interpretability vs. performance:** Your gradient boosting model outperforms logistic regression by 6% AUC on readmission prediction. The hospital's physicians say they will not use a model they cannot understand. How would you address this? Is there a middle ground?
4. **The leakage problem:** After deploying your readmission model, you discover that "number of discharge medications" was highly predictive — and that this feature is partially determined by the physician's expectation of readmission risk. How does this feedback loop affect your model's validity?
5. **Sample efficiency:** You have 5,000 labeled examples for training. Describe how you would choose between logistic regression (low variance, high bias), random forest (moderate both), and gradient boosting (higher variance, lower bias) for this scenario.
6. **Threshold selection:** A model predicts sepsis risk with AUC 0.88. Setting the threshold at 0.3 catches 92% of sepsis cases but triggers alerts for 40% of all patients. Setting it at 0.7 triggers alerts for only 12% of patients but misses 35% of sepsis cases. Which threshold is better? Who should make this decision?
7. **Data labeling as a bottleneck:** Supervised learning requires labeled data. For your IAAIS domain, estimate the cost (time, money, expert effort) of labeling 1,000 training examples. What strategies would you use to reduce this cost while maintaining label quality?
8. **Your IAAIS Classifier:** Define the classification task for your IAAIS Classifier. What are the features? What is the label? What metric should be optimized? Sketch the feature engineering pipeline you would apply to raw data from your domain.

---

## Further Reading

### Textbooks

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Available free at web.stanford.edu/~hastie/ElemStatLearn/. The comprehensive theoretical reference.

Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Authoritative Bayesian treatment.

### Ensemble Methods

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. The original random forest paper.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*. XGBoost — the dominant tabular learning algorithm.

### Applied ML

Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly. Practical feature engineering across domains.

---

*— End of Chapter 6 —*
