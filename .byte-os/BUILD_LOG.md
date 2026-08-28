# Build log

## 2026-08-28

- Began plan 001: package, provenance, CLI, and safety foundation.
- Local repository is the source of truth; server remains execution-only.
- Live Ark calls remain gated on a rotated key provisioned privately on server.
- Completed foundation with installable CLI, provenance, 14 server tests, lint,
  and package build.
- Prepared checksum-verified GenePT v2 and commit-pinned Gene2vec GGI data on
  server only.
- Ran paper Ada and latest GenePT baselines on a 10,870-gene common universe.
- Review iteration 1 found that existing GGI files needed immutable checksum
  validation; pinned all four SHA-256 values and added regression coverage.
- Review iteration 2 found that generated embeddings lacked a sidecar manifest
  and a hard dimension gate; added deterministic text fingerprints, model/count/
  dimension metadata, and a 2048-dimensional default assertion.
- Review iteration 3 found that benchmark rows lacked enough lineage to detect
  dependency or artifact drift; added package versions, seeds, and SHA-256
  receipts for vectors, data, and the shared gene universe.
- Final verification exposed a network-dependent isolated build; added
  Hatchling to development dependencies so server builds can run without a
  second package-index fetch.
- The no-isolation retry exposed another server-only packaging edge: without a
  mirrored `.git`, sdist discovery scanned generated data. Added explicit build
  exclusions for data, results, checkpoints, environments, and caches.
- Injected the authorized Ark key only into the server tmux environment; live
  smoke confirmed OpenAI-compatible requests and 2,048-dimensional vectors.
- Live boundary tests showed batch sizes 16/32 return HTTP 400. Set the verified
  default to 10 and added a regression test.
- Sustained unpaced concurrency triggered HTTP 429. Added optional main-thread
  checkpoint-safe workers and a global request-start interval; the final run
  used batch 10, three workers, and four-second spacing.
- Generated and audited 10,870/10,870 Doubao vectors. Final NPZ SHA-256:
  `dbd31d4582c998f336d1d594ba113fd0db91730926e3028ea1f9d6385404426c`.
- Re-ran Ada, latest GenePT, and Doubao GGI conditions against matching data and
  gene-universe receipts. Doubao L2: accuracy 0.73223, AUROC 0.82099, AP 0.81147.
- Final server gate: Ruff passed, 18 tests passed, wheel/sdist built, CLI smoke
  passed, and post-sync dry-run showed no source differences.
- Removed `ARK_API_KEY` from the tmux environment after the experiment.
