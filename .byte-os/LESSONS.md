# Confirmed project lessons

## 2026-09-05: Preserve DinoGenePT identity while replacing old model code

- DinoGenePT is the repository, Python package, CLI, and future cell-model
  identity. GenePT-Seed is an embedding condition/subsystem, not the project
  name.
- A request to clean the old DINO/cell-model implementation does not authorize
  renaming or splitting the project. Remove the retired implementation while
  retaining the `DinoGenePT` repository and `dinogenept` package, then record
  the replacement architecture as an unimplemented design.
- Historical result labels and JSON schema fields named `genept-seed` remain
  stable for artifact compatibility; do not rewrite released receipts merely
  to match the package rename.
- When replacing GenePT's embedding backbone, preserve the selected text,
  gene mapping, comparison universe, normalization, split, classifier, and
  evaluation settings unless a separately named ablation changes them.
