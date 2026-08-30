# Iteration 14: DinoGenePT functional matrix

## Goal

Turn the discussed leakage-aware DINO/scGPT-style perturbation design into one
extensible package and execute every ablation path for one epoch.

## Changes

- Separated one model implementation, dataset adapters, common evaluation, and
  per-dataset/per-experiment configs.
- Implemented Base plus sparse source-only local views, condition-bag DINO,
  delta-iBOT, Base-only inference, MoE/KDA/SiTU/KoLeo/attention-residual
  ablations, semantic negative controls, and a post-Top-K iBOT shortcut control.
- Added chunked backed-data variance, BF16, sparse expert dispatch, GPU memory
  gates, sequential matrices, executable source hashes, and receipt reuse.
- Added model plugins, independent dataset folders and configs, one shared
  condition-macro evaluator, structured output identities, and paired
  condition-level comparison receipts.

## Evidence-led repairs

1. Real CUDA training found FP32 routing weights entering BF16 `index_add_`;
   the dispatch now casts weights to expert-output dtype and has a CUDA test.
2. Real KoLeo backward found an in-place `cdist` modification; the diagonal now
   uses non-in-place masking and has a backward test.
3. A separate critic found that reuse receipts did not bind every material
   input/output and that `complete` preceded checkpoint save. Dataset, split,
   prior, optional pretrained checkpoint, executable source, resolved config,
   metrics, and model checkpoint are now hash-bound; completion is last.
4. The critic also found an iBOT post-expression selection shortcut,
   source-count weighting, single-source parameter drift, thin model registry,
   and insufficient MoE diagnostics. The default iBOT set now comes from
   training-only variance, local loss is condition-weighted, all source
   adapters remain instantiated, the runner delegates through a model plugin,
   and route/load/collapse diagnostics are receipted.

## Result

The final 21-row matrix completed on physical GPU 0, saved and verified 21
checkpoints, and an immediate second invocation reused all 21 receipts. Server
tests passed 75/75; local tests passed 61 with one expected no-Torch skip. Full
attention passed the token permutation gate while KDA failed it as an explicitly
ordered ablation. This is a functional smoke, not an architecture ranking.
