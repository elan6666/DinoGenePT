# Iteration 12: Freeze the gene universe before gene-disjoint splitting

## Evidence

- The first preflight partitioned 11,299 raw GGI genes and filtered missing
  vectors afterward; those outputs are invalid and isolated server-side.
- The fixed 10,870-gene vector allowlist induces 10,858 genes with at least one
  retained edge and 12 isolated genes.
- The accepted protocol partitions all 10,870 genes for seeds 42–51 into 8,696
  train and 2,174 test genes, with zero gene overlap for every seed.
- Universe filters and split receipts are identical across latest GenePT,
  Protein, ProteinPathway, and HPA.

## Finding and action

Filtering after partitioning changes the experimental unit and violates the
fixed-universe claim. Pair rows are now induced on the frozen allowlist first,
while isolated genes remain in the partition and vector universe.

## Result

Corrected existing-condition results pass the universe, split, normalization,
pair operator, estimator, dependency, and seed-grid comparison gates. Invalid
preflights are excluded from all final metrics.
