# Iteration 13: Admit only complete matched results

## Evidence

- Reactome, SIGNOR, masked SIGNOR, and shuffled SIGNOR checkpoints each have
  17,730 exact text/model-hash hits, zero pending rows, and width 2,048.
- All four aligned vectors contain the exact 17,730-label master universe with
  100% requested coverage.
- The first final summary attempt rejected `seed-go-protein` because the frozen
  baseline receipt identifies itself as `genept-seed-go-protein`.
- After correcting only that local script label, the idempotent finalizer reused
  completed result JSONs and all fixed, gene-disjoint, and property comparison
  gates passed.

## Finding and action

A human-readable condition alias must never be substituted for the embedding
identity stored in a frozen result receipt. The finalizer now uses the receipt's
exact baseline label; no vector or benchmark was regenerated for this repair.

## Result

The fixed GGI comparison contains eight matched rows, gene-disjoint comparison
contains the same eight embeddings over ten identical 10,870-gene split
receipts, and the property comparison contains the complete 8 × 4 × 2 grid.
Compact receipts and an explicit task-dependent interpretation are ready for
ship review.
