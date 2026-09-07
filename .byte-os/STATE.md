# Active project state

## Active goal (latest user decisions, 2026-09-07)

- Goal mode is active: default native model, 2 full pretraining epochs,
  Adamson and Norman each 10 LoRA fine-tuning epochs; no ablations.
- Pretraining corpus choice delegated; continuous expression required, scGPT
  route excluded. Candidate: 500k-human-cell stratified subset from scPRINT's
  public CELLxGENE Census source, release 2023-12-15; not yet materialized/frozen.
  Check actual source access, study provenance/overlap and dual-GPU throughput.
- Finish dataset selection and model implementation before supplementing
  independent GenePT-Seed views. Reuse previous Ark Agent Plan/keychain/model
  and hash-exact checkpoints; no embedding API calls made in this update.
- Native LoRA and DINO/iBOT/KoLeo/center/EMA primitives added; local original
  suite 60 pass, torch fixtures skipped locally (torch absent). Server CPU:
  8 LoRA/distillation + 5 KDA + 8 metric fixtures pass. KDA chunks match recurrent
  values/gradients. Native default backbone added with four CPU fixture tests;
  12/768 hybrid KDA/MLA + BlockAttnRes + dense SiTU width1. See source ledger
  for reduced MLA dimensions and explicit from-scratch decay initialization.
  Full runner, reconstruction heads/orchestration, data and GPU gates pending.
- Server torch 2.13.0+cu130, cellxgene-census1.18.0 and data dependencies installed
  in own .venv. Never mutate GraD-Pert environment. Both GPUs currently occupied by
  PIDs391010/391014; no DinoGenePT GPU training started.
- Census release 2023-12-15 inventory fetched: 651 datasets. Server artifact
  data/pretraining/census-2023-12-15-inventory.json. Candidate metadata query:
  Tabula Sapiens and cross-tissue immune atlas; not a frozen training manifest.
- Candidate cell metadata command is running in exec session81061; output dir
  data/pretraining/census-two-atlas-candidates-v1, script census_candidate_cells.py.
  Inspect process/output before retrying. No raw expression shards downloaded.
- AGENTS.md contains enduring standards only (native package/source fidelity,
  local-server-GitHub synchronization, safe server use and scientific integrity).
  Campaign sequence and budgets belong here, not in AGENTS.md.
- Ark skill read; macOS Keychain status confirms credential present without
  exposing it. Code defaults match doubao-embedding-vision and Agent Plan
  /api/plan/v3. No API generation yet. After dataset/model completion use the
  ark-keychain ssh-run helper, preserve exact caches and global rate limits.
- Current gate: local 60 tests pass (torch-dependent tests skipped), server
  77 tests pass on CPU; both Ruff and wheel/sdist builds pass, CLI passes.
  Server build explicitly excludes .runtime/data/checkpoints/credentials;
  code sync checksum dry-run clean. User-owned uv.lock remains untouched.

## Active implementation and data update (2026-09-07)

- User selects CellFM pretraining data and CellFM perturbation data first:
  Adamson + Norman. Do not automatically launch the prior five-dataset scope.
- See docs/CELLFM_DATA_PROTOCOL.md: full processed pretraining corpus access
  remains unverified; downstream Zenodo ZIP is not evidence of full availability.
- Official notebook uses norman-1000, simulation seed3, 15 epochs, GEARS emb/frozen.
  Resolve archive-to-notebook preprocessing and splits before formal runs.
- Native evaluation metrics and eight unit tests added locally; eight pass.
  This does not constitute a cell-model implementation or training result.
- Next: corpus provenance/access, CellFM split/axis audit, native model and
  source-parity gates. No GPU jobs launched; do not preempt GraD-Pert jobs.

## Current design delivery (2026-09-07)

- Authoritative replacement-model plan: docs/CELL_DINO_KNOWLEDGE_LOCAL_DESIGN.md
  v2 and docs/ENGINEERING_EVALUATION_DESIGN.md. Older Byte OS experiment/status
  notes describe historical GenePT work, not the current cell-model settings.
- Main: 12/768 single-direction hybrid KDA, short convolution off; LC2/4/8,
  FFN1/4; fixed five-loss objective, no loss ablations now.
- Dual-5090 DDP target; per-rank batch8/accumulation8 is a capacity proposal,
  not a tested fit. Existing GraD-Pert GPU jobs were not interrupted.
- Native source-first package, shared GraD-Pert-aligned five-dataset evaluator,
  performance receipts and metrics-only checkpoint retention are documented.
- No cell-model implementation, new training, dataset download or historical
  artifact cleanup was performed for this design delivery. Next work follows
  the engineering plan's source-ledger/parity/capacity/integration gates.
- Existing untracked uv.lock is user-owned and remains untouched.

## Existing implemented subsystem (historical verification below)

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
