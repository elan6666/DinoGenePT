# Iteration 7: Complete-universe corpus integrity

## Evidence

- Frozen master: 17,730 exact labels from the five graph-axis union and fixed
  GGI universe; 2,469/2,469 perturbation targets are included.
- Four-stage corpus audit: every stage has the exact master key set and order,
  no empty text, and source/output hashes.
- All five per-dataset graph axes and target sets pass exact coverage for every
  progressive condition.
- Append-only transitions pass for every gene. Changed/unchanged rows are
  14,847/2,883, 10,740/6,990, and 14,658/3,072.
- Optional-source absence therefore preserves the preceding text rather than
  adding generated, borrowed, or zero content.

## Finding and action

The original global count was too indirect to prove the user's per-dataset
coverage requirement. Added `audit-knowledge-corpora`, a regression test, and a
compact server receipt that fails on key/order drift, empty text, non-prefix
rewrites, or any missing graph/target label.

## Result

Core corpus completeness gate passes. Vector completeness and matched GGI
evidence remain pending and are assigned to iterations 8 and 9.
