# Product

GenePT-Seed: a controlled GenePT variant that keeps the official NCBI + UniProt
gene text and replaces only the embedding backbone with Doubao.

# What Was Delivered

- Checksum-pinned GenePT/Gene2vec server data preparation.
- Resumable Ark embedding generation with safe Agent Plan endpoint validation.
- Safe batch/concurrency pacing derived from live 400/429 evidence.
- Matched GGI evaluation for paper Ada, latest GenePT, and GenePT-Seed.
- L2 primary result, native-vector sensitivity, receipts, tests, and runbook.

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
- `src/genept_seed/embedding.py`: Ark client, pacing, and resume logic.
- `src/genept_seed/benchmarks.py`: matched GGI estimator.

# Verification

- Server Ruff passed; 18 tests passed.
- Wheel and sdist built successfully.
- 10,870/10,870 Doubao vectors, dimension 2,048, 100% selected coverage.
- Matching train/test counts and source/universe receipts across conditions.
- Doubao L2: 0.73223 accuracy, 0.82099 AUROC, 0.81147 AP.
- Post-apply sync dry-run: no differences.
- API key removed from the tmux environment after use.

# OKR Status

All revised, user-scoped KRs are achieved. Multi-task benchmarking was not part
of the selected experiment subset.

# Review And Iteration Summary

Three evidence-led iterations hardened data checksums, embedding manifests and
dimensions, and benchmark lineage. Review 3 fixed the live batch-default defect
and returned `ship`.

# Known Gaps

The measured advantage is limited to the released GGI split. Other GenePT
experiments, downstream cell models, and external datasets remain untested.

# Recommended Next Steps

Rotate the shared Ark key. If expanding the study, add one orthogonal GenePT
task with the same receipt-first protocol rather than broadening claims now.

# Real User Feedback Status

No real-user feedback was collected or claimed.
