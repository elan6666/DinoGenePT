# Review 7: DinoGenePT ship review

## Verdict

Ship the implementation and functional-smoke evidence.

The first critic pass was blocking. It identified stale runtime evidence,
incomplete input/output lineage, premature completion receipts, weak server/GPU
enforcement, a DinoGenePT-specific runner, iBOT selection leakage, local-loss
weighting, source-parameter drift, and missing route/collapse diagnostics. The
main implementation addressed these findings and regenerated the entire
source-hashed smoke matrix before this verdict.

The same critic then re-audited the repaired source and final receipts and
returned `acceptable` with no remaining blocker for the scoped functional
implementation plus one-epoch smoke delivery.

## Required boundaries

- Do not present one-epoch synthetic-prior metrics as biological performance.
- Keep Teacher expression access restricted to training-split conditions.
- Keep optional locals source-only and omit unavailable sources.
- Treat KDA as an ordered experimental ablation because it fails permutation
  invariance; full attention remains the default.
- Preserve the one-model/config-ablation distinction and common evaluator when
  adding future models.
- Do not claim scGPT initialization: the generic checkpoint compatibility gate
  is implemented, but an exact scGPT key adapter has not been built or tested.
- Require frozen official splits, real source-only priors, repeated seeds, and
  parameter/FLOP controls before interpreting model differences scientifically.

## Verification

- Local: 61 passed, one expected no-Torch skip, Ruff clean.
- Server: 75 passed, Ruff clean, wheel/sdist and both CLIs pass.
- Runtime: 21/21 rows complete with 21 hash-verified checkpoints; 21/21
  idempotently reused; 66--79 MiB peak; physical GPU 0 only.
- Lineage: one dataset fingerprint plus dataset/split/prior/source hashes across
  the matrix; artifact hashes revalidated before reuse.
- Invariance: full attention `5.96e-7 <= 1e-5`; ordered KDA `0.20075 > 1e-5`.
