# Research

Research date: 2026-08-28

## Scope

Official GenePT source/data, selected paper evaluation logic, and the Fire
Volcano Agent Plan embedding interface.

## Findings

- The [official GenePT repository](https://github.com/yiqunchen/GenePT) is a
  notebook repository, not a packaged library. Its current documented commit is
  pinned in `DECISIONS.md`.
- [Zenodo v2](https://zenodo.org/records/10833191) contains NCBI-only text,
  NCBI + UniProt text, Ada embeddings, and a gene-protein
  `text-embedding-3-large` artifact in one 574.4 MB archive.
- The official Table 1 notebook uses five-fold stratified evaluation with
  logistic regression and random forest. The Figure 2 GGI notebook uses the
  fixed Gene2vec train/test files and pair embeddings formed from two genes.
- The notebooks contain absolute author-machine paths, making direct execution
  non-portable.
- The Agent Plan alias `doubao-embedding-vision` maps to the current plan model
  and is configured through `https://ark.cn-beijing.volces.com/api/plan/v3`.
  The standard `/api/v3` endpoint is explicitly excluded to avoid extra fees.

## Product implications

- Copy: task definitions, fixed public datasets, estimator families, and seeds.
- Adapt: paths, loading, checkpointing, manifests, and deterministic CLI runs.
- Avoid: notebook state, hidden local files, untracked random state, and claims
  based only on paper table values.
- Differentiate: exact model/text provenance, coverage accounting, resumable
  generation, native and dimension-controlled comparisons, and server receipts.

## Caveats

- Zenodo's prose for the fourth artifact says "summary in 1" while its filename
  and companion combined-text JSON imply gene-protein content. The archive must
  be inspected and key coverage compared before scientific interpretation.
- A multimodal general embedding model is not guaranteed to improve biomedical
  gene semantics; matched downstream evidence decides.

