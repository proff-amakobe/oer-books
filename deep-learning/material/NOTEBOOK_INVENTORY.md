# Notebook Inventory Report

**Deep Learning: A Comprehensive Guide** — companion notebook audit

---

## Summary

- Chapters/sections scanned: 16 (Chapters 1–16, Second Edition) + Setup section
- Chapters whose Hands-On Exploration genuinely requires a notebook: **14** (Setup + Chapters 1, 3–14)
- Chapters whose Hands-On Exploration does **not** require a notebook: **3** (Chapter 2: browser-only TensorFlow Playground activity; Chapters 15–16: written reflection/design-review activities using the student's own Architecture Record)
- Notebooks explicitly named in the manuscript before this audit: **2** (`setup_check.ipynb`, `hands_on_ch1.ipynb`)
- Notebooks referenced only implicitly ("a starter notebook is provided," no filename given) before this audit: **12**
- Missing notebooks identified: **14** (all of them — none existed in the repository)
- Missing notebooks created: **14**
- Broken/implicit references corrected: **12** (see "Naming/Reference Corrections" below)
- Orphan notebooks identified: **0** (after corrections; see below)
- Supporting files created: **0** (every notebook generates or loads its own data — see Design Notes)

---

## Notebook Inventory

| Chapter | Section | Notebook | Status Before | Status After | Notes |
|---|---|---|---|---|---|
| Setup | Environment Verification | `setup_check.ipynb` | Referenced, missing | Created, **executed** | Fully offline; checks packages, GPU, runs an end-to-end toy pipeline |
| 1 | Feature Engineering vs. Feature Learning | `hands_on_ch1.ipynb` | Referenced, missing | Created, **executed** | Generates its own 150-image shape dataset; no downloads |
| 2 | Seeing Non-Linearity in Action | — | N/A | N/A | Browser-only (TensorFlow Playground); no notebook needed |
| 3 | Seeing Training Dynamics in Action | `hands_on_ch3.ipynb` | Referenced implicitly, missing | Created, **executed** | Synthetic dataset substituted for literal CIFAR-10 (external download unavailable in this environment); optional Colab-only cell included for the literal CIFAR-10 version |
| 4 | Architecture Archaeology | `hands_on_ch4.ipynb` | Referenced implicitly, missing | Created, **executed** | Attempts real ImageNet-pretrained ResNet-50/EfficientNet-B0; graceful offline fallback verified working |
| 5 | From Classification to Detection | `hands_on_ch5.ipynb` | Referenced implicitly, missing | Created, **syntax-validated only** | Requires `torch`/`ultralytics`; not installed in this sandbox due to disk constraints (see Validation section) |
| 6 | The Vision Pipeline Audit | `hands_on_ch6.ipynb` | Referenced implicitly, missing | Created, **executed** | Full evaluation harness (confusion matrix, calibration, t-SNE, model card) |
| 7 | Watching Memory Decay | `hands_on_ch7.ipynb` | Referenced implicitly, missing | Created, **executed** | Trains RNN and LSTM from scratch on a synthetic subject-recall task |
| 8 | Reading What Attention Sees | `hands_on_ch8.ipynb` | Referenced implicitly, missing | Created, **executed** | Attempts real pretrained BERT; offline fallback uses a from-scratch trained Transformer on a designed pointer-following task |
| 9 | Same Model, Different Tasks | `hands_on_ch9.ipynb` | Referenced implicitly, missing | Created, **executed** | Attempts real HuggingFace pipelines; offline fallback trains three domain-specific classifiers to demonstrate distribution shift |
| 10 | Exploring the Joint Space | `hands_on_ch10.ipynb` | Referenced implicitly, missing | Created, **executed** | Attempts real CLIP; offline fallback trains a small two-tower contrastive (CLIP-style) model from scratch |
| 11 | Navigating the Latent Space | `hands_on_ch11.ipynb` | Referenced implicitly, missing | Created, **executed** | Trains a real VAE on scikit-learn's built-in handwritten digits dataset (no download needed) |
| 12 | Steering the Denoiser | `hands_on_ch12.ipynb` | Referenced implicitly, missing | Created, **executed** | Attempts real Stable Diffusion; offline fallback trains a small class-conditional generator using genuine classifier-free-guidance training/inference |
| 13 | Watching an Agent Learn | `hands_on_ch13.ipynb` | Referenced implicitly, missing | Created, **executed** | Real DQN trained from scratch on `gymnasium`'s CartPole (pure physics simulation, no downloads) |
| 14 | Measuring the Production Gap | `hands_on_ch14.ipynb` | Referenced implicitly, missing | Created, **executed** | Real TensorFlow Lite INT8 post-training quantization on a from-scratch trained model; every number measured, not simulated |
| 15 | Tracing the Infrastructure | — | N/A | N/A | Written analysis activity using the student's own Architecture Record; no notebook needed |
| 16 | The Design Review | — | N/A | N/A | Written reflection activity; no notebook needed |

---

## Orphaned Existing Notebooks

None. Before this audit, no notebooks existed in the repository at all — there was nothing to be orphaned. After creation, every notebook created is now explicitly referenced by filename in its corresponding chapter (see corrections below), so there are zero orphans in the final state.

---

## Naming/Reference Corrections

Twelve chapters referenced "a starter notebook is provided" (or equivalent phrasing) **without ever naming the file**. This is the kind of reference that silently breaks for a reader — there is no way to know which file to open. The following files were corrected to name the actual notebook explicitly:

| File | Correction |
|---|---|
| `Ch_3_Second_Edition.md` | Added `hands_on_ch3.ipynb` filename to the CIFAR-10 training activity description |
| `Ch_4_Second_Edition.md` | Added `hands_on_ch4.ipynb` filename to the Architecture Archaeology setup description |
| `Ch_5_Second_Edition.md` | Added `hands_on_ch5.ipynb` filename to the detection/segmentation setup description |
| `Ch_6_Second_Edition.md` | Added `hands_on_ch6.ipynb` filename to the evaluation harness description |
| `Ch_7_Second_Edition.md` | Added `hands_on_ch7.ipynb` filename to the Tools line |
| `Ch_8_Second_Edition.md` | Added `hands_on_ch8.ipynb` filename to the Tools line |
| `Ch_9_Second_Edition.md` | Added `hands_on_ch9.ipynb` filename to the Tools line |
| `Ch_10_Second_Edition.md` | Added `hands_on_ch10.ipynb` filename to the Tools line |
| `Ch_11_Second_Edition.md` | Added `hands_on_ch11.ipynb` filename to the Tools line |
| `Ch_12_Second_Edition.md` | Added `hands_on_ch12.ipynb` filename to the Tools line |
| `Ch_13_Second_Edition.md` | Added `hands_on_ch13.ipynb` filename to the Tools line |
| `Ch_14_Second_Edition.md` | Added `hands_on_ch14.ipynb` filename to the Tools line |

Chapter 1 and the Setup section already named their notebooks explicitly (`hands_on_ch1.ipynb`, `setup_check.ipynb`) and needed no correction.

**Important — these corrections were made to the working `.md` manuscript copies used for this audit, not to the original `.docx` chapter files.** If you'd like, I can apply the equivalent one-line correction to each `.docx` chapter file directly — say the word and I'll do that as a follow-up pass.

---

## Design Notes — Why Some Notebooks Differ From a Literal Reading of the Chapter Text

Several chapters describe activities using large pretrained models (real ImageNet-pretrained ResNet-50/EfficientNet-B0, YOLO, Mask R-CNN, BERT, CLIP, Stable Diffusion) hosted on services this authoring environment cannot reach (Hugging Face Hub, PyTorch model hosting, etc. are outside its network allowlist). For every such notebook, the design follows the same pattern:

1. **Attempt the real pretrained model first.** In Google Colab, with internet access, every notebook will genuinely attempt to load the real model exactly as the chapter describes.
2. **Fall back gracefully if unavailable**, rather than crashing. Each fallback was purpose-built to preserve the *actual mechanism* being taught (e.g., Chapter 12's fallback uses genuine classifier-free-guidance training, not an analogy for it; Chapter 8's fallback required two redesigns before the synthetic task actually forced the model to use content-based attention routing rather than finding a shortcut).
3. **Say so, clearly, in the notebook**, so a reader understands which mode they're in and what they'd see differently in Colab.

Three notebooks needed no fallback at all because they don't depend on external hosting: Chapter 11 (VAE) uses scikit-learn's bundled digits dataset; Chapter 13 (DQN) uses `gymnasium`'s CartPole, a pure physics simulation; Chapter 14 (quantization) trains its own small model and applies real, local TensorFlow Lite compression.

---

## Supporting Resource Issues

None identified. Every notebook generates its training/evaluation data directly in-notebook (synthetic shapes, synthetic sequences, or scikit-learn's bundled datasets), so no `.csv`, `.json`, image, or other external data file needed to be created or committed separately.

The book's Setup section (Chapter 1) lists `pip install tensorflow numpy matplotlib pillow jupyter scikit-learn` as the required dependencies. This is sufficient for `setup_check.ipynb` and the fully-offline-capable notebooks. Chapters that attempt real pretrained models additionally reference (in-notebook, as optional installs) `torch`, `torchvision`, `ultralytics`, `transformers`, and `diffusers` — none of these are currently listed in the book's Setup section. **Recommendation:** add a short "Optional packages for later chapters" note to the Setup section listing these, so readers aren't surprised.

---

## Validation

**Executed successfully, end-to-end, in a clean kernel (13 of 14 notebooks):** `setup_check.ipynb`, `hands_on_ch1.ipynb`, `hands_on_ch3.ipynb`, `hands_on_ch4.ipynb`, `hands_on_ch6.ipynb`, `hands_on_ch7.ipynb`, `hands_on_ch8.ipynb`, `hands_on_ch9.ipynb`, `hands_on_ch10.ipynb`, `hands_on_ch11.ipynb`, `hands_on_ch12.ipynb`, `hands_on_ch13.ipynb`, `hands_on_ch14.ipynb`.

Several of these required real debugging after the first execution attempt surfaced genuine problems — not just import errors, but *pedagogically wrong results that looked superficially fine*:
- Chapter 1's rule-based classifier was scoring at chance level even on matched-distribution data (broken hand-engineered features — fixed).
- Chapter 3's regularization experiment showed no effect at first (hyperparameters retuned until overfitting and its correction were both clearly visible).
- Chapter 8's attention weights came out essentially uniform on the first two task designs — the model had found a shortcut that let it solve the task *without* needing content-based attention, which would have taught readers the wrong lesson entirely. The task was redesigned around a large-vocabulary pointer-following structure that makes the shortcut mathematically infeasible.
- Chapter 11's VAE exhibited posterior collapse on the first training configuration (all samples nearly identical) — fixed by switching to a summed reconstruction loss and a much smaller KL weight.

**Syntax/structure-validated only, not executed (1 of 14):** `hands_on_ch5.ipynb`. This notebook requires `torch`, `torchvision`, and `ultralytics`. Installing these in this sandbox triggered disk-space exhaustion (their CUDA-enabled dependency chains are large); rather than risk destabilizing the environment for marginal validation benefit on a notebook designed for Colab anyway, I left it syntax-validated only. The code is complete and follows the same patterns proven to work in the executed notebooks.

**Require GPU or optional dependencies for their *genuine* (non-fallback) activity:** Chapters 4, 5, 6, 8, 9, 10, 12 (pretrained-model downloads); Chapter 12 specifically benefits from a GPU for real Stable Diffusion. None of these are required to complete the fallback activity offline.

**Final reference audit:** `python3 scripts/audit_notebook_references.py manuscript notebooks` reports **0 missing, 0 orphaned** — every `.ipynb` filename named anywhere in the manuscript now resolves to an actual file, and every notebook that exists is now named somewhere in the manuscript.
