# Active project state

- Scope: retain only the GenePT-Seed corpus, embedding, provenance, vector
  audit, and gene-level evaluation toolkit.
- Local source root is `/Users/elan/code/GenePT-Seed`; server materialization
  root remains `/data/yilangliu/GenePT-Seed`.
- `src/genept_seed` is the only packaged Python module. DINO self-distillation,
  cell-model backbones, GEARS/Scouter adapters, training configs, Trackio, and
  architecture ablations have been removed from the current tree.
- NCBI + UniProt Base, GO-EXP, Protein, Reactome/SIGNOR, HPA, GraD-Pert axis
  coverage, checkpoint-safe Ark embedding generation, and matched GGI/property
  evaluations remain supported.
- Existing compact GenePT result receipts under `docs/results` remain tracked;
  raw datasets, embeddings, checkpoints, and full logs remain server-only.
- The GenePT-only local gate passes 52 tests, Ruff, shell syntax, wheel/sdist,
  module CLI, and installed entry-point checks.
- Four pre-cleanup uncommitted files from the retired training/Trackio path are
  recoverable from Git stash
  `backup-before-genept-only-cleanup-2026-09-04`.
