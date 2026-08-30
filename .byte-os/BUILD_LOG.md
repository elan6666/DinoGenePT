# Build log

## 2026-08-28

- Froze a 17,730-label master universe from all five canonical GraD-Pert graph
  axes plus the fixed GGI universe; all 2,469 unique perturbation targets are
  members of the graph union.
- Materialized official UniProt, InterPro, Reactome, SIGNOR, and HPA snapshots
  on the server with source hashes. A duplicate HPA download corruption and a
  SIGNOR `DIRECT=t` schema mismatch were caught by gates, repaired, and rerun.
- Built three complete append-only corpora. Exact per-source coverage:
  UniProt structured 14,847; InterPro 14,731; Reactome 10,013; direct human
  protein-protein SIGNOR 4,255; HPA 14,658.
- The progressive corpus audit passed all five graph axes and target sets.
  Protein/Pathway/HPA changed 14,847/10,740/14,658 rows; all other rows remained
  byte-identical to the preceding condition.
- Started three checkpointed Doubao embedding runs. Independent four-second
  intervals caused one Pathway HTTP 429; all runs were safely resumed with
  twelve-second per-session intervals staggered by four seconds, preserving
  the validated aggregate four-second request-start rate.
- Pre-result delivery gate passed on both Mac and server: 36 tests, Ruff,
  isolated sdist/wheel build, and CLI smoke. The credential scan was tightened
  after correctly identifying `.env.example` as a placeholder rather than a
  leaked key; the refined real-token/non-placeholder scan passed.
- Added a matched-GGI comparison auditor and two regression tests. Its server
  preflight accepted latest GenePT, completed Seed, and Seed-GO as sharing all
  15 fairness fields, and will reject protocol drift before combining the three
  pending conditions. The local suite now contains 38 passing tests.
- Found and removed an order-dependent GGI edge: eight master-universe label
  groups differ only by case, while the legacy `as_dict()` uppercased every key.
  Fixed-universe benchmarks now prefer the exact requested label and permit
  only a unique case-fold fallback. The delivered Seed-GO baseline rerun remains
  byte-identical, proving no regression for existing uppercase-only artifacts.
- Added the final progressive-vector auditor for exact 17,730-label order,
  2,048 width, nonzero/finite vectors, exact 10,870 GGI labels, per-dataset graph
  and target coverage, common model, hashes, and explicit case-collision groups.
  The local suite now contains 40 passing tests.
- Completed all three exact checkpoints at 17,730/17,730, pending 0, width
  2,048. The locked finalizer aligned all artifacts and passed the five-dataset
  graph/target vector gate with no zero rows.
- Ran the fixed GGI protocol for Protein, ProteinPathway, and
  ProteinPathway-HPA. Metrics were 0.73415/0.82571/0.81872,
  0.74968/0.83615/0.82859, and 0.73975/0.83088/0.82407 respectively.
- The six-row auditor accepted all 15 fairness fields. ProteinPathway improved
  on Seed+GO by +0.01440 Accuracy, +0.01203 AUROC, and +0.01317 AP and was the
  best tested condition; HPA did not add a further gain.
- Copied only compact corpus, vector, and GGI receipts into `docs/results/`;
  raw data, vectors, checkpoints, and full logs remain server-only and ignored.
- Final local and server gates each passed 40 tests, Ruff, wheel/sdist build,
  CLI smoke, and JSON validation. Local diff/credential scans passed; server
  receipt hashes match the Mac and post-apply checksum sync is empty.
- Review 5 found no required changes and returned `ship`.
- Committed the tracked delivery as `daf95b23b82f621960171140a44e80892f0b2219`,
  pushed it to GitHub `main`, and verified the remote ref at the same commit.
  The user-owned untracked `uv.lock` was not staged or committed.

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
- Built GO-EXP from the completed corpus and fixed 10,870-gene GGI allowlist;
  9,597 genes received bounded experimental GO text. The rebuilt selected
  corpus is byte-identical to the prior selected output while its manifest now
  pins the completed-corpus base.
- Audited the GO SQLite checkpoint by exact gene/text/model/dimension rather
  than row count. Resumed the 8,539 stale or missing rows through the authorized
  Agent Plan API and reached 10,870/10,870 exact matches with no error log.
- Materialized base and GO NPZ files at width 2,048 and 100% selected coverage.
  SHA-256: `dbd31d45...426c` and `06269245...f21e`.
- Ran the fixed Gene2vec GGI evaluation only; no GraD-Pert model was trained.
  Base Accuracy/AUROC/AP: 0.73223/0.82099/0.81147. GO-EXP:
  0.73528/0.82411/0.81541.
- Cross-condition audit confirmed identical data/universe receipts, pair rows,
  L2 preprocessing, pair-sum operator, classifier, seed, and dependencies.
- Iterations 4--6 and review 4 captured corpus lineage, cache correctness,
  matched metrics, scope correction, and a `ship` verdict.
- Final sync audit found that the unanchored `results/` exclusion also matched
  `docs/results/`. Anchored generated-directory excludes at repository root so
  the compact tracked receipt synchronizes without touching server run outputs.
- Final Mac and server gates each passed 32 tests and Ruff; wheel/sdist and CLI
  checks passed on server. Post-apply rsync dry run was empty and the tracked
  result receipt hash matched on both hosts.
- Mirrored the rsync fix in `.gitignore`: root-anchored `/results/` keeps
  generated outputs excluded while allowing `docs/results/` receipts to be
  tracked.

## 2026-08-31 DinoGenePT

- Implemented one extensible perturbation model package with independent model
  and dataset folders, common evaluation, strict composed configs, and 21
  config-only ablations.
- Local static/unit gate passed with 61 tests and one expected Torch skip.
- The first server matrix exposed a BF16/FP32 MoE dispatch mismatch; explicit
  weight casting and CUDA BF16 regression tests fixed it.
- The second matrix exposed an in-place KoLeo `cdist` mask; a non-in-place
  masked fill and backward regression test fixed it.
- A critic audit prompted input/output hash binding, atomic completion,
  server/GPU enforcement, a model-plugin boundary, fixed training-variance iBOT
  selection, condition-weighted local loss, parameter-stable source rows, and
  route/collapse diagnostics.
- The final source-hashed matrix completed 21/21 rows on physical GPU 0, wrote
  and verified 21 checkpoints, then reused 21/21 receipts on an idempotency
  check. Peak allocation was 66--79 MiB.
- Final server gate passed 75 tests, Ruff, wheel/sdist, and both CLI smoke
  commands. Full attention passed permutation invariance; ordered KDA failed as
  expected. No run used physical GPU 1.
