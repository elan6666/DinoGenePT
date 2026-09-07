# Confirmed project lessons

## 2026-09-07: Single-process backward is not a DDP integration test

- The first backbone block created a depth-read query/norm although no history
  exists at that point. Single-process backward passed, but two-rank DDP failed
  with unfinished gradient reduction because those parameters were never used.
- Remove structurally dead parameters instead of hiding this with broad unused
  parameter handling. Assert gradients exist for all active pretraining parameters
  and test multi-rank accumulation across more than one optimizer step.
- Keep forward AND backward inside no_sync on accumulation microsteps. Give
  DataLoader its own generator so an extra iterator creation on resume does not
  perturb the model RNG. Test interrupted vs uninterrupted weights and RNG.

## 2026-09-07: Keep user defaults separate from official reference settings

- The current main experiment is 12 blocks / 768 width, single-direction
  hybrid KDA, short convolution off. CellFM 40/1536 is a reference, not the
  experiment default. Do not promote an assistant suggestion into a user decision.
- Local-count sweep is 2/4/8 and FFN expansion 1/4. Loss weights are fixed;
  do not restore superseded coefficient or loss-removal sweeps.
- Implement native dinogenept modules after reading pinned upstream behavior.
  Do not substitute runtime upstream model calls or guessed implementations.
- Stage-1 locals are shared-head crops; stage-2 locals are independently
  embedded source-only texts with missing sources omitted.

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
## 2026-09-07: Separate durable standards from campaign instructions

- Mistake: put current execution order, training budgets and model hyperparameters
  into AGENTS.md. User clarified that AGENTS.md is for enduring standards.
- Prevention: keep model/data choices, order, epochs and API-generation gates
  in STATE/design/configs. Keep native-package/source-first implementation and
  local-server-GitHub synchronization in AGENTS.md. The runtime restriction is
  on upstream research models, not foundational dependencies such as torch.
