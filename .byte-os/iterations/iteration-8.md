# Iteration 8: Exact vector completeness

## Evidence

- All three checkpoint audits reached 17,730/17,730 exact gene/text/model hits,
  pending 0, with dimensions exactly `[2048]`.
- Aligned artifacts preserve the frozen master order and contain 17,730 finite,
  nonzero vectors under one `doubao-embedding-vision` model.
- All 10,870 fixed GGI labels are exact matches after the exact-label-first
  selection fix; eight case-fold collision groups remain explicit.
- The final vector audit proves 100% graph-axis and perturbation-target coverage
  for K562, RPE1, Jurkat, HepG2, and Norman.

## Finding and action

Raw NPZ presence and shape alone could not prove correct text hashes, order, or
mixed-case identity. The finalizer therefore gates alignment on exact SQLite
checkpoint audits and then runs `audit-knowledge-vectors` before evaluation.

## Result

The three vector artifacts satisfy the complete 17,730-label delivery contract.
The compact receipt is `docs/results/PROGRESSIVE_KNOWLEDGE_VECTOR_AUDIT.json`.
