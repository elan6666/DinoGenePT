# Confirmed project lessons

## 2026-09-04: Keep GenePT-Seed independent from downstream cell models

- This repository owns GenePT-compatible text corpora, embedding generation,
  provenance, vector audits, and gene-level evaluation only.
- Do not add cell-model backbones, self-distillation, perturbation training,
  downstream model adapters, or training dashboards to this package. Those
  belong in a separate project that consumes released GenePT-Seed vectors.
- When replacing GenePT's embedding backbone, preserve the selected text,
  gene mapping, comparison universe, normalization, split, classifier, and
  evaluation settings unless a separately named ablation changes them.
