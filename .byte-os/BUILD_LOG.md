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
