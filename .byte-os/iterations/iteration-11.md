# Iteration 11: Remove control-sentence confounding

## Evidence

- The first partner-control implementation also replaced the subject gene with
  `this protein`, changing more than partner identity.
- A temporary full ProteinPathway rebuild initially differed for 21 genes due
  publisher display-case normalization and mixed-case entity admission.
- After restoring original subject symbols, publisher display case, selection,
  and ordering, the rebuilt corpus and existing baseline have the identical
  SHA-256 `7a952fa7...aa3c`.

## Finding and action

The requested intervention is partner identity only. SIGNOR, masked, and
shuffled tasks were stopped before producing final NPZs, their corpora were
rebuilt, and checkpoint reuse was restricted by exact text hash. Reactome was
unchanged. All four tasks were restarted with one request start every four
seconds in aggregate.

## Result

The accepted controls preserve the existing named Pathway sentence contract.
Only source inclusion or the partner position varies.
