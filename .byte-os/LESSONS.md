# Confirmed project lessons

## 2026-09-08: Label coverage is not identity or pretrained-token coverage

- A complete17730-label master still contains63 multi-label identity groups;
  the current HGNC snapshot cannot resolve1573 labels. Some approved target
  genes have Base text but no matching token in the frozen GeneCompass vocabulary.
- Keep label coverage, HGNC resolution, source availability and pretrained-token
  coverage separate. Audit against the actual dataset-specific extended corpus,
  not only the old master. Never merge processed expression columns or mask
  ambiguous source identity as absent GO; test these cases explicitly.

## 2026-09-07: Explicitly spawn DataLoader workers with NCCL

- Installed PyTorch DDP documentation warns NCCL is not fork-safe. The runner
  previously relied on the platform's default multiprocessing start method.
- Select multiprocessing_context=spawn whenever workers>0. Reopen read-only
  mmap shards in each worker; do not inherit GPU contexts. Verify the complete
  epoch and exact resume tests with a real spawned loader, not only workers=0.

## 2026-09-07: Preserve operation order before relaxing precision gates

- Factoring a triangular solve into separate value/state terms passed FP32
  primitive tests but failed whole-backbone BF16 acceptance. Keep the failed
  receipt and original tolerances; mathematical equivalence is not bitwise
  finite-precision equivalence.
- Batching pair coefficients while retaining residual-before-solve order
  passed the same gate at full2048tokens/60664vocabulary. Confirm token-axis
  and vocabulary scope, not just a256-token fixture. Forward equality still
  does not imply equal optimizer trajectories; report this distinction.

## 2026-09-07: GPU ownership is not just a process-group comparison

- Torch Elastic creates each rank with start_new_session=True (verified in
  installed subprocess_handler.py). A rank may initialize CUDA before its peer
  queries nvidia-smi; different PGIDs do not prove unrelated ownership.
- Only accept peers with our same verified torchrun launcher parent. Arbitrary
  shell siblings and unrelated parents remain forbidden. Test both launch
  styles (torchrun entrypoint and python -m torch.distributed.run), peer races
  and rejection of unrelated jobs. Never relax the guard to same Unix user.
- Single-GPU RNG save/restore must touch only its current device; using
  get_rng_state_all can initialize unwanted contexts on another research GPU.

## 2026-09-07: IO helper draws are not the formal evaluator's control sampler

- Earlier seed42/no-replacement-when-large helper draws tested IO, not GraD-Pert
  method fidelity. The inspected formal sampler uses seed20260824 with a
  dataset/split/condition SHA256-derived128-bit PCG64 key, sampling truth contexts
  and compatible controls with replacement, exactly300 ordered controls.
- Freeze these rows before evaluation; validate the exact regeneration. Match
  DE selection order too: filter non-dropout, first20, remove targets, no refill.
  State metric-method alignment separately from dataset/split comparability.

## 2026-09-07: Distinguish absent genes from missing source annotations

- The source-only corpus helper counted requested genes absent from both input
  corpora in its requested total, but omitted them from available/missing counts.
- Reject requested genes outside the input corpus universe. A mapped gene with
  no GO/Protein/etc suffix is a legitimate missing annotation; a gene not present
  in the source universe is an incomplete materialization, not the same case.
- Training must additionally pin text fingerprints and exact source-only/vector
  receipts; matching dimensions or a filename containing GO is insufficient.

## 2026-09-07: Bound downloads and preserve progress on each retry

- A fresh curl invocation used internal --retry with no initial resume option.
  After its 1800-second transfer timeout, it truncated a ~1.5GB archive prefix.
- Retry in our own loop with --retry 0 and --continue-at - on EVERY attempt.
  Never fall back to opening the partial file with wb after curl failure; urllib
  resumption must validate HTTP206 and the Content-Range start before appending.
- Inspect remote ZIP directory before large downloads when only a few members
  are needed. CellFM target files occupy only the first ~196MB of a 5.3GB stored
  archive. Reusing the prefix plus entry CRC32/local SHA256 is efficient, but
  must never be described as full-archive MD5 verification.

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

- Download concurrency is not a durable speed guarantee: the GeneCompass
  four-worker8MiB range downloader hitHTTP429 after6.87GB. Preserve its
  contiguous prefix and failure log; verify the resume range, then reduce to
  a single streaming request. Never blindly restart the same parallel load.

- Mistake: put current execution order, training budgets and model hyperparameters
  into AGENTS.md. User clarified that AGENTS.md is for enduring standards.
- Prevention: keep model/data choices, order, epochs and API-generation gates
  in STATE/design/configs. Keep native-package/source-first implementation and
  local-server-GitHub synchronization in AGENTS.md. The runtime restriction is
  on upstream research models, not foundational dependencies such as torch.
