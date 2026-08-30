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
- A 17,730-label master corpus covering every graph-axis and perturbation-target
  label across the five frozen GraD-Pert datasets plus the fixed GGI universe.
- Three append-only knowledge conditions and aligned 2,048-wide Doubao vector
  artifacts: Protein, ProteinPathway, and ProteinPathway-HPA.
- Compact corpus/vector/GGI audits and a six-condition matched comparison.
- Reactome-only and SIGNOR-only decomposition plus partner-masked and
  deterministic partner-shuffled controls over the complete 17,730-label
  corpus.
- Static source-versus-actual-text SIGNOR/GGI leakage receipts and a strict
  ten-seed gene-disjoint GGI evaluation over the frozen 10,870-gene universe.
- Four pinned GenePT property tasks under a shared 519-gene allowlist,
  ten repeated five-fold splits, logistic regression, RF500, and
  Accuracy/AUROC/AP reporting.
- Compact Priority 1/2 corpus, checkpoint, vector, leakage, fixed GGI,
  gene-disjoint GGI, and property comparison receipts.

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
- `docs/GRADPERT_MASTER_KNOWLEDGE.md`: complete-universe construction and result.
- `docs/results/PROGRESSIVE_KNOWLEDGE_CORPUS_AUDIT.json`: append-only coverage.
- `docs/results/PROGRESSIVE_KNOWLEDGE_VECTOR_AUDIT.json`: exact vector coverage.
- `docs/results/PROGRESSIVE_KNOWLEDGE_GGI.json`: six-condition matched GGI table.
- `docs/PRIORITY_1_2_EXPERIMENTS.md`: leakage controls, strict split protocol,
  four property tasks, complete results, and interpretation boundary.
- `docs/results/priority-1-2/`: final compact Priority 1/2 receipts.
- `src/genept_seed/embedding.py`: Ark client, pacing, and resume logic.
- `src/genept_seed/benchmarks.py`: matched GGI estimator.

# Verification

- Local and server Ruff passed; 40 tests passed on each host.
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
- All three progressive checkpoints reached 17,730/17,730 exact text/model
  matches, pending 0, width 2,048.
- Each aligned artifact has 17,730 nonzero finite vectors, all 10,870 exact GGI
  labels, and 100% exact graph/target coverage for all five datasets.
- Protein/ProteinPathway/ProteinPathway-HPA Accuracy/AUROC/AP are
  0.73415/0.82571/0.81872, 0.74968/0.83615/0.82859, and
  0.73975/0.83088/0.82407.
- All six rows share 15 fairness fields. The three new compact receipt hashes
  match between Mac and server: `875d6b75...ce16`, `814c2408...be3`, and
  `c35def5c...cb76`.
- Four control checkpoints are each 17,730/17,730 exact text/model hits,
  pending 0, width 2,048; aligned vectors cover all requested labels.
- The eight-row fixed GGI comparison, eight-row × ten-seed gene-disjoint
  comparison, and complete 8 × 4 × 2 property grid pass their fairness gates.
- Masked SIGNOR has zero actual partner-pair exposure; the shuffled control has
  no accidental test GGI exposure. Under strict gene-disjoint evaluation,
  masked/shuffled controls reach 0.82436/0.82427 AUROC versus 0.82271 for the
  named full pathway condition.
- Local and server verification each pass 50 tests, Ruff, build, CLI, JSON,
  script syntax, and secret scan. Post-apply checksum sync is clean.

# OKR Status

All revised, user-scoped KRs are achieved. The added GenePT property multitask
benchmark is complete; GraD-Pert evaluation remains outside the selected
embedding-only experiment.

# Review And Iteration Summary

Thirteen evidence-led iterations hardened data checksums, embedding manifests,
dimensions, corpus lineage, text-hash cache validation, exact-case selection,
complete-universe coverage, partner-control fidelity, strict split units, and
benchmark fairness. Review 6 verified the Priority 1/2 artifacts and returned
`ship`.

# Known Gaps

The fixed GGI result is complemented by ten gene-disjoint splits and four
GenePT property tasks, but no paired significance test was pre-registered.
Downstream cell models and external perturbation-expression datasets remain
untested. No GraD-Pert training or evaluation is included.

# Recommended Next Steps

Rotate the shared Ark key. If expanding the study, add paired uncertainty tests
or a separately scoped GraD-Pert prior ablation with the same receipt-first
protocol rather than broadening embedding-probe claims now.

# Real User Feedback Status

No real-user feedback was collected or claimed.

# Git Handoff

Delivery commit `daf95b23b82f621960171140a44e80892f0b2219` was pushed and
verified on GitHub `main`. The user-owned untracked `uv.lock` remains local and
was not included.

The Priority 1/2 implementation, documentation, and compact receipts were
committed as `24e3bd07b1cc1be7fc498c628a691b0a3f2cdbd5`, pushed, and verified
on GitHub `main`. The final follow-up commit changes only Byte OS closeout
metadata. The user-owned untracked `uv.lock` remains local and excluded.

## 2026-08-31 DinoGenePT package delivery

- Added the independent `dinogenept` package while retaining the frozen
  `genept_seed` prior-toolkit CLI.
- Added lazy model/dataset/evaluator registries, per-dataset config folders,
  config composition and validation, structured output identities, source and
  config hashes, and idempotent sequential matrices.
- Implemented the complete 21-row config-only ablation surface for one model:
  condition DINO, delta-iBOT, four source-only locals, sparse dynamic locals,
  semantic/missingness/iBOT controls, direction loss, attention residual,
  Dense/MoE/quantile/SiTU, KoLeo, hybrid KDA, and an explicit post-Top-K iBOT
  shortcut control.
- Added model plugins, input/output hash manifests, last-write completion
  receipts, physical-GPU enforcement, paired condition comparisons, and
  diagnostic receipts for MoE routing, KoLeo activity, and representation
  collapse.
- The server Adamson-mini 1-epoch matrix completed 21/21 rows on physical GPU 0
  with 66--79 MiB peak allocated memory. All 21 checkpoints passed artifact
  hash validation; a repeat reused 21/21 receipts and did not retrain. GPU 1 and
  GraD-Pert remained untouched.
- Local gates: 61 passed and one expected no-Torch skip. Server gates: 75
  passed, Ruff, wheel, sdist, old/new CLI smoke. The server project environment
  now includes its declared PyYAML dependency.
- The smoke is not a scientific comparison; real source-only priors, frozen
  official splits, repeated seeds, and full GEARS-scale training remain future
  work.
