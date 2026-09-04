# Active project state

- Project identity: DinoGenePT. The current implementation temporarily retains
  only the GenePT-Seed corpus, embedding, provenance, vector-audit, and
  gene-level evaluation subsystem while the replacement cell model is designed.
- Local source root is `/Users/elan/code/DinoGenePT`; server materialization
  root is `/data/yilangliu/DinoGenePT`.
- `src/dinogenept` is the packaged Python module. The retired DINO self-distillation,
  cell-model backbones, GEARS/Scouter adapters, training configs, Trackio, and
  architecture ablations have been removed from the current tree.
- NCBI + UniProt Base, GO-EXP, Protein, Reactome/SIGNOR, HPA, GraD-Pert axis
  coverage, checkpoint-safe Ark embedding generation, and matched GGI/property
  evaluations remain supported.
- Existing compact GenePT result receipts under `docs/results` remain tracked;
  raw datasets, embeddings, checkpoints, and full logs remain server-only.
- The knowledge-subsystem local gate passes 52 tests, Ruff, shell syntax, wheel/sdist,
  module CLI, and installed entry-point checks.
- Four pre-cleanup uncommitted files from the retired training/Trackio path are
  recoverable from Git stash
  `backup-before-genept-only-cleanup-2026-09-04`.
