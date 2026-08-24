# Chapter Reference Audit

This audit reviews the manuscript’s chapter-level Further Reading sections without inventing or externally completing metadata. The entries remain in their original chapters.

## Inventory

- URLs found across the chapters: 83
- Unique URLs: 73
- Central bibliography supplied: none
- Pandoc citation keys in the manuscript: none

## Confirmed Repeated Sources

- Bender et al., “On the Dangers of Stochastic Parrots,” appears across Chapters 8, 9, 10, 11, and 16.
- Bommasani et al., “On the Opportunities and Risks of Foundation Models,” appears in Chapters 9 and 15.
- Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” appears in Chapters 10 and 12.
- Jay Alammar’s “The Illustrated Transformer” appears in Chapters 2 and 8.
- *Deep Learning* by Goodfellow, Bengio, and Courville appears in Chapters 7 and 8.
- TensorFlow Playground is linked more than once in Chapter 2 and appears again in Chapter 3.

Repeated sources are not errors when they support different chapters; they are candidates for future centralized bibliography records.

## Metadata Issues Requiring Editorial Review

- Several web resources provide a title and URL but no explicit author or publication date.
- Some books do not state an edition in the chapter entry.
- Some conference papers use abbreviated venue names while others use full proceedings titles.
- Long author lists are represented inconsistently with escaped ellipses.
- Some arXiv links accompany a conference citation without clearly identifying whether the cited version is the preprint or proceedings version.
- Several entries include extensive reading annotations immediately after the bibliographic information. These are pedagogically useful but should be separated consistently from the citation during a future APA normalization pass.
- Chapter-level reference formatting varies in capitalization, page ranges, venue style, DOI presentation, and terminal punctuation.

## Citation Infrastructure Recommendation

Do not create bibliography records until each source has been verified. When verification is complete, migrate records incrementally into `references.bib`, use stable descriptive citation keys, retain reading annotations as chapter prose, and configure a validated APA 7 CSL supplied by an authoritative source. No empty or fabricated bibliography file is needed at this stage.
