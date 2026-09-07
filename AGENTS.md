# DinoGenePT engineering standards

This file contains durable project standards, not a campaign plan. Put current
goals, execution order, datasets, epochs, hyperparameters and ablation choices
in `.byte-os/STATE.md`, design documents and versioned experiment configs.
Keep one-off task ordering and completion gates out of this file, including
the order of data selection, model construction and corpus supplementation.

## Native package and source fidelity

- Maintain our own `dinogenept` Python package and CLI. Preserve the existing
  GenePT knowledge subsystem and historical artifact/schema compatibility.
- Read the actual upstream implementation before reproducing a mechanism.
  Pin source revisions, understand its inputs/masks/normalization, and implement
  the behavior in our package. Document adaptations and test numerical parity.
  Do not guess undocumented behavior or present a proposal as verified code.
- If upstream code or a relevant detail is unavailable, record the uncertainty
  in the design/source ledger. Clearly distinguish our proposed adaptation
  from an upstream implementation; do not claim fidelity without evidence.
- Do not import or wrap upstream research-model implementations as our model:
  CellFM, Kimi, DINO, scGPT, scFoundation, GEARS, Scouter and GraD-Pert are
  source references, not runtime model dependencies. This restriction does
  **not** prohibit foundational libraries such as PyTorch, NumPy, SciPy,
  CUDA/NCCL, HDF5, AnnData or public data-access clients.
- In particular, `import torch` is allowed; delegating our backbone or training
  algorithm to a CellFM or other upstream model package is not. Own the model
  implementation, configuration, training entry points and evaluation wiring.
- Respect upstream licenses and attribution. Do not confuse implementing a
  published mechanism with permission to redistribute all upstream code.
- Organize different models and datasets separately with dataset-specific
  configs and a shared evaluation system. Ablations of one model are config
  variants, not independent model packages. Avoid duplicated evaluation code.

## Local, server and GitHub consistency

- Local Git is the source of truth: `/Users/elan/code/DinoGenePT`.
  Edit code locally, then synchronize to `/data/yilangliu/DinoGenePT` on
  `yilangliu@10.24.1.91`; do not maintain a divergent server-only implementation.
- Use the approved SSH workflow, dry-run file transfers and verify checksums.
  Preserve unrelated changes and server data; do not delete files as a side
  effect of synchronization without checking the exact targets.
- Keep verified source/config/docs changes committed and pushed to the intended
  GitHub branch. Inspect remote/branch/status first; do not force-push, change
  branches, or include unrelated user changes. Report commit and sync status.
- Never synchronize or commit credentials, raw datasets, generated embeddings,
  checkpoints or full logs to GitHub. Raw materializations stay server-side;
  synchronize only code and approved compact non-secret receipts.

## Server execution and performance

- Run data downloads/processing, embeddings, training, inference and benchmarks
  on the server. Local lightweight unit tests and code checks are allowed.
- Inspect active jobs, GPU ownership/utilization/memory and disk before launching.
  Do not kill, pause or compete with unrelated research jobs. Measured free
  memory alone does not establish exclusive GPU availability.
- Instrument training entry points: raw/weighted losses, learning rate,
  optimizer steps, cell/token counts, data/step time, throughput, peak memory
  and total/trainable parameters. Record precision, distributed strategy,
  accumulation, hardware and resolved config.
- Optimize only with numerical and task-semantic checks. Do not silently reduce
  the requested model/data/epoch budget to make a run fit or finish sooner.
- Prefer compact `.pt` checkpoints, sparse/chunked datasets and JSON/CSV scalar
  receipts; avoid default giant PKL prediction dumps. A `.pt` suffix does not
  itself imply compression or pickle-free serialization. Support resumability.

## Provenance, evaluation and secrets

- Preserve gene identifiers, order, preprocessing scale, condition eligibility,
  splits, seeds and reference/control populations across matched comparisons.
  Audit coverage and study/donor/cell overlap; prevent held-out outcome leakage.
- Freeze manifests and source hashes. Missing annotations or undefined metrics
  must have explicit policies/reasons; do not fabricate data or silently delete
  difficult examples. Identical metric names alone do not prove comparability.
- Reuse exact validated caches/checkpoints before recomputing. For API workflows,
  use approved credential helpers and the configured billing endpoint; never
  print keys, dump credential-bearing environments or persist secrets in source,
  logs, command arguments, synchronized files or tmux configuration.
- Verify with `python -m pytest`, lint, CLI smoke checks, packaging and relevant
  integration/numerical/resumption tests. Distinguish implemented, tested,
  running and completed; a smoke run is not a formal experiment result.
- Large/noisy paths: `.venv/`, `.runtime/`, `data/`, `results/`, `checkpoints/`,
  generated arrays/archives, caches and build artifacts.
