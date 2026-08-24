# Figure Technical Review

Phase 2.5 review of the eleven numbered figures identified in `figure-inventory.md`. SVG is authoritative; PNG is a 2400-pixel-wide compatibility export. All plotted curves and optimization paths are explicitly conceptual rather than empirical.

| Figure | Chapter | Status | Technical confidence | Source asset | Inserted | Alt text | Outstanding issue |
|---|---:|---|---|---|---|---|---|
| 1.2 | 1 | READY — EDITORIAL REVIEW | High | `assets/figures/fig-1-2-neural-network-timeline.svg` | Yes | Yes | The chapter describes the second AI Winter only as the 1990s and early 2000s; the shaded interval is therefore labeled approximate rather than given false precision. |
| 1.3 | 1 | READY | High | `assets/figures/fig-1-3-hierarchical-feature-learning.svg` | Yes | Yes | None. The four stages reproduce the chapter terminology. |
| 3.1 | 3 | READY — EDITORIAL REVIEW | High | `assets/figures/fig-3-1-loss-landscape.svg` | Yes | Yes | Uses a contour view rather than decorative perspective so multiple basins and paths remain legible; paths are labeled conceptual. |
| 3.3 | 3 | READY | High | `assets/figures/fig-3-3-training-loop.svg` | Yes | Yes | None. The four phases and direction match the chapter. |
| 3.4 | 3 | READY — EDITORIAL REVIEW | Medium–high | `assets/figures/fig-3-4-learning-rate-effects.svg` | Yes | Yes | Curves are qualitative illustrations because no run data or numeric values are supplied. No performance claim is encoded. |
| 3.5 | 3 | READY — EDITORIAL REVIEW | Medium–high | `assets/figures/fig-3-5-batch-strategies.svg` | Yes | Yes | Descent paths are qualitative illustrations, not measured trajectories. |
| 3.6 | 3 | READY — EDITORIAL REVIEW | High | `assets/figures/fig-3-6-training-curves.svg` | Yes | Yes | Curves illustrate the four diagnostic patterns described in the chapter; they are not empirical results. |
| 3.7 | 3 | READY | High | `assets/figures/fig-3-7-dropout.svg` | Yes | Yes | None. Inactive units are crossed out as well as grayed, so the state does not depend on color. |
| 3.8 | 3 | READY | High | `assets/figures/fig-3-8-gradient-pathologies.svg` | Yes | Yes | None. Arrow thickness and labels distinguish shrinking and growing gradients. |
| 3.9 | 3 | TECHNICAL REVIEW REQUIRED | Medium–high | `assets/figures/fig-3-9-learning-rate-schedules.svg` | Yes | Yes | **FIGURE TECHNICAL REVIEW REQUIRED:** the placeholder requests final validation accuracy and claims warmup + cosine is best, but supplies no experiment or values. The figure therefore compares schedule shapes only and makes no accuracy ranking. |
| 3.10 | 3 | READY — EDITORIAL REVIEW | High | `assets/figures/fig-3-10-mipds-training-engine.svg` | Yes | Yes | The diagram is limited to the Week 3 interfaces stated in the chapter; editorial review should confirm that this is the intended definitive MIPDS interface before print lock. |

## Checklist Result

- Arrow directions were checked against the surrounding chapter descriptions.
- Terminology and mathematical symbols match the manuscript.
- No external diagrams, screenshots, or traced assets were used.
- Solid and dashed lines, shapes, text, and position supplement color encoding.
- SVGs use responsive `viewBox` dimensions and readable system-safe sans-serif labels.
- Captions retain the manuscript's educational claims; Figure 3.9 deliberately omits unsupported numeric outcomes.
- Every inserted figure has instructional alt text.
