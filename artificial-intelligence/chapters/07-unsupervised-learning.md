# Chapter 7: Finding Structure in the Dark

**Unsupervised Learning, Clustering, and Anomaly Detection**

*CSC5350 · Artificial Intelligence*

---

## Opening Narrative

### The Map That Drew Itself

In 2003, a team of neuroscientists at Stanford connected a hundred electrodes to a rat's hippocampus — the brain region responsible for spatial memory — and recorded the electrical activity as the rat explored a maze. They expected to see neurons associated with specific places: place cells, first discovered by John O'Keefe in 1971, that fire when an animal is at a specific location.

What they found was richer than that. Without any label telling the neurons "this is position (3,4) in the maze," without any supervisor providing correct answers, the neurons had spontaneously organized themselves into a map. Each neuron had a receptive field — a specific region of the maze where it fired preferentially. And the arrangement of these fields, across the hundred recorded neurons, encoded the geometry of the maze with surprising fidelity.

The hippocampus was performing unsupervised learning. From raw sensory experience — proprioception, visual flow, landmark recognition — it was discovering the hidden structure of space. No labels, no supervision, no explicit objective. Just the organization that emerges when you look for patterns in experience.

This is what unsupervised learning does computationally: it looks for structure — clusters, dimensions, patterns, anomalies — in data without being told what to look for. The structure it finds may be more relevant, more surprising, and more useful than any structure a human would have thought to encode in labels.

> **"Supervised learning finds what you tell it to look for. Unsupervised learning finds what is actually there."**

---

## Learning Objectives

After completing this chapter, you will be able to:

1. Describe the unsupervised learning paradigm and distinguish it from supervised learning.
2. Implement K-means clustering and analyze its convergence, sensitivity to initialization, and appropriate use cases.
3. Describe hierarchical clustering and explain when dendrograms reveal useful structure.
4. Explain DBSCAN and describe why density-based clustering handles non-convex clusters and noise better than K-means.
5. Apply Principal Component Analysis to reduce dimensionality while preserving variance, and interpret principal components.
6. Describe t-SNE and UMAP and explain why they are appropriate for visualization but not for distance-preserving embedding.
7. Design and implement an anomaly detection system using isolation forests and autoencoders.
8. Apply unsupervised pre-training to improve supervised learning with limited labeled data.
9. Build the IAAIS Pattern Recognizer — an unsupervised module that discovers structure in unlabeled domain data.

---

## Key Terminology

| Term | Plain-Language Definition |
|---|---|
| **Unsupervised Learning** | Learning patterns, structure, or representations from data without labeled examples. The algorithm must find structure that is intrinsic to the data. |
| **Clustering** | Partitioning data into groups (clusters) such that items within a cluster are more similar to each other than to items in other clusters. The clusters are not specified in advance. |
| **K-Means** | A clustering algorithm that partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids to be the mean of assigned points. |
| **Centroid** | The mean position of all points assigned to a cluster. The centroid of a K-means cluster is the "center of gravity" of the cluster. |
| **Inertia (Within-Cluster Sum of Squares)** | The sum of squared distances from each point to its cluster's centroid. K-means minimizes inertia. Lower inertia means tighter clusters. |
| **Elbow Method** | A heuristic for choosing K by plotting inertia vs. K and looking for the "elbow" — the K beyond which additional clusters produce diminishing inertia reduction. |
| **Silhouette Score** | A measure of clustering quality: how similar a point is to its own cluster compared to other clusters. Ranges from -1 (wrong cluster) to 1 (perfect cluster). |
| **Hierarchical Clustering** | Builds a tree-like hierarchy of clusters by either merging small clusters into larger ones (agglomerative) or splitting large clusters into smaller ones (divisive). |
| **Dendrogram** | A tree diagram showing the sequence of merges or splits in hierarchical clustering. The height at which two clusters merge indicates their distance. |
| **Linkage** | The method for computing distance between clusters: single (minimum pairwise distance), complete (maximum), average, or Ward (minimize variance increase). |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise. Groups points in dense regions; marks isolated points as noise. Does not require specifying K; handles arbitrary cluster shapes. |
| **Core Point** | In DBSCAN, a point with at least MinPts neighbors within radius ε. Core points are the "centers" of dense regions. |
| **Border Point** | In DBSCAN, a non-core point within ε of a core point. On the edge of a dense region. |
| **Noise Point** | In DBSCAN, a point that is neither a core point nor a border point. Treated as an outlier — not assigned to any cluster. |
| **Principal Component Analysis (PCA)** | A dimensionality reduction technique that finds the directions (principal components) of maximum variance in the data and projects the data onto these directions. |
| **Principal Component** | A direction in the original feature space along which the data has maximum variance. Each subsequent PC is orthogonal to all previous PCs. |
| **Explained Variance Ratio** | The fraction of total variance captured by each principal component. Used to decide how many components to retain. |
| **t-SNE** | t-Distributed Stochastic Neighbor Embedding. A nonlinear dimensionality reduction technique optimized for visualizing high-dimensional data in 2D or 3D. Preserves local structure but not global distances. |
| **UMAP** | Uniform Manifold Approximation and Projection. Similar to t-SNE but faster and better at preserving both local and global structure. |
| **Autoencoder** | A neural network trained to reconstruct its input by passing it through a bottleneck — a compressed latent representation. The bottleneck forces the network to capture the most important structure. |
| **Anomaly Detection** | Identifying observations that deviate significantly from the patterns learned from normal data. Also called outlier detection or novelty detection. |
| **Isolation Forest** | An anomaly detection algorithm that isolates anomalies by building random trees: anomalies are isolated with fewer splits than normal points. |
| **Reconstruction Error** | In autoencoder-based anomaly detection: the difference between an input and its reconstruction. High reconstruction error indicates an anomaly — the autoencoder fails to reconstruct unusual patterns. |

---

## Section 1 — Learning Without Labels

Supervised learning requires labeled examples — which means it requires the labor of human annotators who must examine each example and assign the correct label. For many real-world problems, labeling is impossible at scale: there are no ground truth labels for "what documents are thematically related," "which customers have similar purchasing patterns," or "which network packets represent normal behavior."

Unsupervised learning sidesteps the labeling bottleneck. Instead of learning a mapping from inputs to predefined outputs, it looks for *intrinsic* structure in the data — patterns that exist whether or not a human has named them.

This capability is valuable in three settings:

**Exploration:** Before building a supervised model, unsupervised analysis reveals the data's structure — which features co-vary, which examples are similar, which subgroups exist, which examples are unusual. This understanding informs every subsequent decision.

**Representation learning:** The compressed representations learned by autoencoders or PCA can serve as features for downstream supervised models — particularly useful when labeled data is scarce but unlabeled data is abundant.

**Anomaly detection:** By learning what "normal" looks like from unlabeled data, anomaly detection systems identify deviations without ever being shown what anomalies look like.

---

## Section 2 — K-Means: The Geometry of Grouping

K-means is the simplest and most widely used clustering algorithm. Its objective is to partition n data points into K clusters such that the total distance from each point to its cluster's centroid (mean) is minimized.

The algorithm alternates between two steps:

**Assignment:** Assign each point to the nearest centroid (by Euclidean distance).

**Update:** Recompute each centroid as the mean of all points assigned to it.

This alternation continues until assignments stop changing — convergence is guaranteed but to a local, not necessarily global, optimum. Different initializations produce different solutions; running K-means multiple times with different random initializations and keeping the best result is standard practice.

```
# K-Means in action: clustering patient vital signs
# Features: [systolic_bp, heart_rate, respiratory_rate, temperature, spo2]
# n=500 ICU patient records, K=3 clusters

Iteration  1: Centroids updated — inertia = 8,421
Iteration  5: Centroids updated — inertia = 5,234
Iteration 10: Centroids updated — inertia = 4,891
Iteration 15: Converged — inertia = 4,782

Cluster 0 (n=187): "Stable" — low BP variability, HR 72, SpO2 98%
  → Centroid: [122, 72, 14, 37.2, 98.1]

Cluster 1 (n=156): "Cardiovascular instability" — high systolic variability
  → Centroid: [158, 98, 18, 37.8, 96.2]

Cluster 2 (n=157): "Respiratory compromise" — low SpO2, high RR
  → Centroid: [110, 88, 24, 38.1, 91.4]

Silhouette score: 0.634 (reasonable cluster separation)
```

### Choosing K

K-means requires specifying K in advance — often the hardest decision. The **elbow method** plots inertia vs. K: the optimal K is where adding more clusters produces sharply diminishing returns (the "elbow" of the curve). The **silhouette score** provides a more principled metric: it measures how similar each point is to its own cluster compared to the nearest other cluster, averaging to give an overall quality score.

Neither method gives a definitively correct K — the "right" number of clusters depends on what the clusters will be used for. Exploratory analysis may call for more granular clusters than a downstream classification task would.

### Limitations of K-Means

K-means works best when clusters are roughly spherical, similar in size, and well-separated. It fails on clusters that are non-convex (crescent-shaped, concentric rings), very different in density, or embedded in noise. For these cases, density-based methods are more appropriate.

---

## Section 3 — Hierarchical Clustering and DBSCAN

### Hierarchical Clustering

**Agglomerative hierarchical clustering** starts with each point as its own cluster and repeatedly merges the two closest clusters until all points form a single cluster. The result is a **dendrogram** — a tree showing the merge sequence. Cutting the dendrogram at a chosen height gives any desired number of clusters.

The **linkage criterion** determines how cluster distance is measured. Ward linkage minimizes the increase in total within-cluster variance at each merge and tends to produce compact, similarly-sized clusters. Single linkage (minimum pairwise distance) can create chain-like clusters; complete linkage (maximum pairwise distance) tends toward compact, spherical clusters.

The dendrogram's value lies in visualization: it reveals the hierarchical structure of similarity at multiple resolutions simultaneously. Pharmaceutical researchers use hierarchical clustering to identify compound families; evolutionary biologists use it to construct phylogenetic trees; text miners use it to identify topic hierarchies.

### DBSCAN: Clusters as Dense Regions

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** defines clusters as dense regions of points separated by sparse regions. It requires no specification of K and naturally identifies outliers as noise.

DBSCAN has two parameters: **ε** (the neighborhood radius) and **MinPts** (the minimum number of neighbors to be a core point). A point is:

- A **core point** if it has ≥ MinPts neighbors within ε
- A **border point** if it is within ε of a core point but has fewer than MinPts neighbors
- A **noise point** if it is neither a core nor a border point

Clusters grow outward from core points: all points reachable from a core point (through chains of core points) belong to the same cluster. Points not belonging to any cluster are noise.

DBSCAN's advantages: discovers clusters of arbitrary shape, requires no K, explicitly identifies outliers. Its disadvantages: sensitive to ε and MinPts choices, struggles with clusters of very different densities.

---

## Section 4 — Dimensionality Reduction: Seeing What Matters

High-dimensional data is both computationally expensive and conceptually opaque. Dimensionality reduction addresses both problems: it compresses data into fewer dimensions while preserving the most important structure.

### Principal Component Analysis

**PCA** finds the directions of maximum variance in the data — the dimensions along which the data spreads most widely. These directions, called **principal components**, are orthogonal (uncorrelated) and ordered by how much variance each captures.

The first principal component is the direction along which the data varies most. The second PC is the direction of maximum remaining variance, orthogonal to the first. And so on. By projecting data onto the top K PCs, we obtain a K-dimensional representation that preserves as much variance as possible.

Two interpretations of PCA are complementary. As a compression technique, PCA reduces data from n dimensions to K dimensions with minimum information loss — the K dimensions that matter most. As a feature decorrelation technique, PCA removes linear correlations between features, which can improve the performance of algorithms sensitive to correlated inputs.

```
# PCA on a medical imaging dataset
# Original features: 1,024 pixel intensities (32×32 image)
# Target: compact representation for clustering and classification

PCA Components    Cumulative Variance Explained
First PC:         31.2%
First 5 PCs:      61.4%
First 10 PCs:     78.9%
First 20 PCs:     91.2%
First 50 PCs:     98.1%
First 100 PCs:    99.8%

# 95% of variance captured by just 36 components
# (out of 1,024 original dimensions)
# Downstream classifier trained on 36-dim PCA features:
# accuracy 91.4% vs 89.2% on raw features — and 28× faster
```

### t-SNE and UMAP: Visualization of Complex Structure

PCA is linear — it finds linear directions of variance. For data with complex nonlinear structure (clusters that are not linearly separable, manifolds in high-dimensional space), nonlinear methods are needed.

**t-SNE** places each data point in 2D or 3D such that points that are nearby in high-dimensional space remain nearby in the 2D projection. It does this by minimizing the difference between probability distributions over neighbor relationships in high and low dimensions. The result is a visualization where clusters in high-dimensional space appear as distinct groups.

Critical caveats: t-SNE does not preserve global distances (the distance between clusters in the t-SNE plot is meaningless) and is non-deterministic (different runs produce different plots). It is a visualization tool, not an embedding suitable for downstream tasks. **UMAP** addresses some of these limitations: it is faster, better at preserving global structure, and produces reproducible results.

---

## Section 5 — Anomaly Detection: Finding the Unusual

Anomaly detection identifies observations that deviate significantly from normal patterns. Unlike classification, it does not require labeled examples of anomalies — which are often rare, novel, or deliberately disguised. Instead, anomaly detectors learn what "normal" looks like and flag deviations.

### Isolation Forest

**Isolation forests** exploit a simple insight: anomalies are rare and different. In a random decision tree, anomalies are isolated (separated from all other points) with fewer splits than normal points, because they occupy sparse regions of the feature space.

An isolation forest builds many such random trees and averages the number of splits required to isolate each point. Short average path length indicates an anomaly; long average path length indicates a normal point.

Isolation forests are efficient (linear time), require no labeled anomaly examples, and handle high-dimensional data well. They are one of the most reliable anomaly detection baselines.

### Autoencoder-Based Anomaly Detection

**Autoencoders** are neural networks trained to reconstruct their input through a bottleneck. The encoder compresses the input into a low-dimensional latent representation; the decoder reconstructs the input from this representation. When trained on normal data, the autoencoder learns to compress and reconstruct normal patterns efficiently.

For anomaly detection: present a new data point to the trained autoencoder and compute the **reconstruction error** — the difference between the input and its reconstruction. Normal points that resemble the training data are reconstructed well (low error). Anomalies, which do not resemble the training data, cannot be reconstructed from the bottleneck that was only trained on normal patterns: they have high reconstruction error.

This approach is particularly effective for high-dimensional data (images, time series, text) where classical anomaly detection methods fail and where a small reconstruction error is a meaningful signal.

```
Autoencoder anomaly detection on hospital equipment sensor data:
Training: 10,000 hours of normal operation (unlabeled)
Architecture: 24 sensor inputs → 12 → 6 → 6 → 12 → 24 reconstruction
Training loss: 0.0042 (MSE on training data after 50 epochs)

Threshold selection (95th percentile of training reconstruction error):
  Normal operation: avg error = 0.0038, 95th pctile = 0.0089
  → Detection threshold: 0.0089

Evaluation on 200 labeled test hours:
  True anomalies detected:     47/51   (92% recall)
  False alarms:                 9/149  ( 6% false alarm rate)

Failure mode analysis: missed 4 anomalies were gradual drift
(slow changes that never crossed threshold in any one hour)
```

---

## Section 6 — Unsupervised Pre-Training and Representation Learning

When labeled data is scarce, unsupervised learning provides an alternative path. **Unsupervised pre-training** uses abundant unlabeled data to learn a general-purpose representation, then fine-tunes on limited labeled data for the specific task.

The intuition: the structure discovered in unlabeled data (clusters, principal components, autoencoder representations) captures genuine patterns in the domain. A classifier built on top of these representations starts from a better place than one built from raw features.

This approach has driven major advances in NLP (language models pre-trained on unlabeled text, then fine-tuned for classification) and computer vision (models pre-trained on ImageNet, fine-tuned for specialized tasks). Chapter 13 will examine the most powerful version of this approach: large language models trained on massive unlabeled corpora.

---

## Section 7 — IAAIS Integration: The Pattern Recognizer

This week you add the **IAAIS Pattern Recognizer** — an unsupervised learning module that discovers structure in unlabeled domain data.

The Pattern Recognizer serves three functions in the integrated IAAIS system. First, it segments the domain data into meaningful groups — patients with similar risk profiles, equipment with similar failure modes, users with similar behaviors — without requiring labeled examples of group membership. Second, it flags anomalies: deviations from normal patterns that may warrant attention or further investigation. Third, it provides learned representations that improve the performance of the supervised Classifier (Chapter 6) when labeled data is limited.

| Chapter | Module | Capability |
|---|---|---|
| Ch 2 | Search Engine | Path planning |
| Ch 3 | Knowledge Base | Structured facts and inference |
| Ch 4 | Planner | Goal-directed action sequences |
| Ch 5 | Uncertainty Module | Calibrated probabilistic beliefs |
| Ch 6 | Classifier | Supervised prediction |
| Ch 7 | Pattern Recognizer | Unsupervised structure discovery |

---

## Hands-On Exploration: Patient Phenotyping Without Labels

### The Activity

Open `hands_on_ch7.ipynb` from the course repository. The notebook contains records of 2,000 ICU patients described by 20 clinical features (vital signs, lab values, interventions) — with no diagnosis or outcome labels.

**Part 1 — Clustering (20 minutes):** Apply K-means for K = 2, 3, 4, 5, 6. Plot the elbow curve and silhouette scores. Choose a K and characterize each cluster by the mean values of its most discriminating features. Do the clusters correspond to recognizable clinical phenotypes?

**Part 2 — Visualization (15 minutes):** Apply PCA to reduce to 2 dimensions. Plot the data colored by cluster assignment. Now apply t-SNE and plot again. Compare: does t-SNE reveal structure not visible in PCA? What structure did PCA miss?

**Part 3 — Anomaly Detection (20 minutes):** Train an Isolation Forest on the full dataset. Flag the top 5% of patients by anomaly score. Examine these patients' feature values. What makes them anomalous? Would these patients benefit from clinical attention?

### Reflection Questions

1. You found K clusters in the ICU data. A clinician says these don't correspond to any known diagnostic categories. Does this mean the clustering is wrong, or that it found something diagnostically novel? How would you distinguish between a meaningless clustering and a genuinely new clinical discovery?
2. The t-SNE visualization shows five visually distinct clusters but your K-means with K=3 grouped them differently. What does this reveal about the relationship between visual structure and statistical cluster structure?
3. Anomaly detection identifies the 5% most unusual patients. Some of these may simply have unusual demographics (very elderly, very young) rather than pathological anomalies. How would you design an anomaly detection system that distinguishes genuine clinical anomalies from benign unusual cases?
4. Your Pattern Recognizer discovers patient clusters in unlabeled ICU data. The Classifier (Chapter 6) has limited labeled training data. Describe how you would use the Pattern Recognizer's output to improve the Classifier's performance.

---

## Case Study: Google News and the Emergence of Topics

### The Problem of Scale

Google News aggregates thousands of news articles per hour from hundreds of sources, in dozens of languages. No human editorial team could group these articles into coherent stories: a piece about a White House press briefing, a Reuters wire story about the same event, an analysis piece from the New York Times, and an international reaction story from Le Monde should all appear under a single "story." But their content varies, their vocabulary differs, and their framing is shaped by the perspectives of their authors.

Google News solved this with clustering. Articles are represented as bags of words (with TF-IDF weighting), and clustering algorithms group articles whose word distributions are similar. The clusters that emerge correspond, remarkably reliably, to discrete news stories — not because the algorithm was told what a "news story" is, but because articles about the same event naturally use overlapping vocabulary.

### What Emerged

The clustering approach revealed something valuable: the same event generates a characteristic fingerprint in language, even across sources with very different editorial perspectives. An article about a trade agreement uses certain terms; an article about a political statement about that trade agreement uses a related but distinct set of terms. The clusters naturally separate events from reactions to events, breaking news from analysis, domestic coverage from international coverage.

This emergent structure — never explicitly programmed — allowed Google News to present not just articles but *perspectives*: the same story from multiple viewpoints, grouped by editorial angle as much as by topic. Unsupervised learning had revealed structure that a human designing a news aggregator might not have thought to encode.

### The Limits

Topic clustering for news works because news articles about the same event share vocabulary. It breaks down at the boundaries: two articles can discuss very different events using similar language, or very similar events in very different language. Satire, opinion pieces, and analysis that uses the language of one topic to discuss another create systematic errors.

This limitation is general: unsupervised methods discover statistical structure, not semantic structure. When statistical regularities align with human concepts (events produce shared vocabulary), unsupervised learning works beautifully. When they diverge (satire deliberately disrupts the statistical regularities of sincere speech), it fails. Understanding when this alignment holds — and when it does not — is part of responsible deployment.

---

## Chapter Summary

We began this chapter in a rat's hippocampus, where neurons spontaneously organized themselves into a spatial map without any labels telling them what to do. Unsupervised learning is the computational analog: discovering structure that exists in data without being told what structure to look for.

K-means gave us the foundational clustering algorithm: simple, fast, and effective for spherical clusters in well-structured data. The elbow method and silhouette score provided principled ways to choose K and evaluate cluster quality. Hierarchical clustering gave us the dendrogram — a richer representation showing structure at every resolution simultaneously. DBSCAN provided density-based clustering that handles non-spherical clusters and explicit outlier labeling.

PCA gave us principled dimensionality reduction through directions of maximum variance — compressing high-dimensional data to its essential structure while enabling visualization and improving downstream model efficiency. t-SNE and UMAP extended this to nonlinear structure, enabling visualization of complex high-dimensional patterns at the cost of interpretable distance relationships.

Anomaly detection — through isolation forests for structured data and autoencoders for complex unstructured data — gave us the ability to learn normality from unlabeled data and flag meaningful deviations. The reconstruction error framework is one of the most versatile ideas in applied machine learning.

In Chapter 8, we enter the deep learning era — building neural networks with many layers that learn hierarchical representations automatically, without any feature engineering, from raw data.

---

## Discussion Questions

1. **Cluster validity:** K-means minimizes within-cluster variance, but variance is not always the right objective. Design a scenario where K-means finds statistically tight clusters that are clinically meaningless. What alternative objective would be more appropriate?
2. **The curse of dimensionality:** In high-dimensional spaces, all points become equidistant — distance metrics lose meaning. How does this affect K-means, DBSCAN, and isolation forests? What strategies can partially mitigate this effect?
3. **PCA and domain knowledge:** PCA finds the directions of maximum statistical variance, which may not correspond to the most domain-relevant dimensions. Describe a scenario in your domain where the first principal component captures variance that is not relevant to the task at hand. How would you handle this?
4. **Anomaly detection and fairness:** A hospital deploys an autoencoder anomaly detection system trained on historical patient data. Patients from underrepresented demographic groups may have different typical vital sign patterns — leading the system to flag them as anomalies even when they are clinically normal. How would you detect and address this?
5. **Choosing the right method:** For each of these applications, identify the most appropriate unsupervised learning method and justify: (a) identifying patient subgroups with similar disease progression, (b) reducing 200 clinical features to the most informative 10, (c) detecting unusual equipment sensor readings in real time, (d) visualizing 500 drug compounds for exploratory analysis.
6. **Unsupervised learning and reproducibility:** K-means, t-SNE, and other unsupervised algorithms are non-deterministic. Two researchers running the same algorithm on the same data may get different results. How does this affect scientific reproducibility? What practices would you recommend?
7. **Semi-supervised learning:** You have 10,000 patient records but only 200 with labels. Describe how you would use unsupervised learning to improve a supervised classifier trained on only the 200 labeled examples.
8. **Your IAAIS Pattern Recognizer:** For your domain, identify one clustering task (what groups would you look for?), one dimensionality reduction task (what would you visualize?), and one anomaly detection task (what counts as anomalous?). For each, specify which algorithm you would use and what evaluation strategy would tell you if it worked.

---

## Further Reading

### Clustering

Jain, A. K. (2010). Data clustering: 50 years beyond K-means. *Pattern Recognition Letters*, 31(8), 651–666. A comprehensive retrospective on clustering methods.

Ester, M., et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD 1996*. The original DBSCAN paper.

### Dimensionality Reduction

Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605. The t-SNE paper.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection. *arXiv:1802.03426*.

### Anomaly Detection

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM 2008*. The Isolation Forest paper.

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3). Comprehensive overview.

---

*— End of Chapter 7 —*
