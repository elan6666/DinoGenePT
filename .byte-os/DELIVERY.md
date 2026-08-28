# Product

GenePT-Seed: a controlled GenePT variant that keeps the official NCBI + UniProt
gene text and replaces only the embedding backbone with Doubao.

# What Was Delivered

- Checksum-pinned GenePT/Gene2vec server data preparation.
- Resumable Ark embedding generation with safe Agent Plan endpoint validation.
- Safe batch/concurrency pacing derived from live 400/429 evidence.
- Matched GGI evaluation for paper Ada, latest GenePT, and GenePT-Seed.
- L2 primary result, native-vector sensitivity, receipts, tests, and runbook.
- Completed-corpus GenePT-Seed and GenePT-Seed+GO-EXP embeddings with a matched
  GGI Accuracy/AUROC/AP comparison.

# How To Run

Follow `README.md`. Keep the Mac repository as source of truth, sync code with
`scripts/sync_code_to_server.sh`, and run materializing commands only under
`/data/yilangliu/GenePT-Seed` on the server.

# How To Test

On the server:

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python -m build --no-isolation
.venv/bin/genept-seed --help
```

# Key Files

- `README.md`: install and server workflow.
- `docs/EXPERIMENT_PROTOCOL.md`: fixed scientific contract.
- `docs/BASELINE_RESULTS.md`: measured comparison and limitations.
- `docs/results/GENEPT_SEED_GOEXP_GGI.json`: exact two-condition result receipt.
- `src/genept_seed/embedding.py`: Ark client, pacing, and resume logic.
- `src/genept_seed/benchmarks.py`: matched GGI estimator.

# Verification

- Local and server Ruff passed; 32 tests passed on each host.
- Wheel and sdist built successfully.
- 10,870/10,870 Doubao vectors, dimension 2,048, 100% selected coverage.
- Matching train/test counts and source/universe receipts across conditions.
- Doubao L2: 0.73223 accuracy, 0.82099 AUROC, 0.81147 AP.
- Completed-corpus GO-EXP L2: 0.73528 accuracy, 0.82411 AUROC, 0.81541 AP.
- Both new artifacts: 10,870/10,870 vectors, width 2,048; all GGI fairness
  identities match and GO-minus-base is +0.00305/+0.00313/+0.00394.
- Post-apply sync dry-run: no differences.
- API key removed from the tmux environment after use.
- Mac/server source sync is checksum-clean; the compact result receipt has
  matching SHA-256 `823b77a2...906` on both hosts.

# OKR Status

All revised, user-scoped KRs are achieved. Multi-task benchmarking was not part
of the selected experiment subset.

# Review And Iteration Summary

Six evidence-led iterations hardened data checksums, embedding manifests,
dimensions, corpus lineage, text-hash cache validation, and benchmark fairness.
Review 4 enforced the embedding-only boundary and returned `ship`.

# Known Gaps

The measured GO advantage is limited to the released GGI split. Other GenePT
experiments, downstream cell models, and external datasets remain untested. No
GraD-Pert training or evaluation is included.

# Recommended Next Steps

Rotate the shared Ark key. If expanding the study, add one orthogonal GenePT
task with the same receipt-first protocol rather than broadening claims now.

# Real User Feedback Status

No real-user feedback was collected or claimed.
