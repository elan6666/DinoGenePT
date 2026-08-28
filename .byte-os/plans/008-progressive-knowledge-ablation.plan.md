# Plan 008: Progressive knowledge ablation

Status: complete

## Acceptance criteria

- One frozen allowlist covers every canonical graph-axis label and perturbation
  target in all five GraD-Pert datasets, plus the fixed GGI universe.
- Every corpus and vector artifact contains all 17,730 exact labels.
- Optional knowledge is appended only for genes with real official-source rows;
  missing records preserve the previous condition's text.
- Three 2,048-wide Doubao artifacts pass exact text-hash checkpoint and axis
  alignment audits.
- GGI uses the same 10,870 genes, split, L2 normalization, pair addition,
  classifier, and seed as the delivered base/GO comparison.
- Accuracy, AUROC, AP, source hashes, coverage counts, limitations, tests,
  review, and delivery are recorded without running GraD-Pert.

## Execution state

- Universe, source snapshots, base/GO, and all three progressive corpora: done.
- Three checkpointed embedding sessions: complete at 17,730/17,730 exact hits.
- Alignment and five-dataset graph/target vector audit: complete.
- Fixed 10,870-gene GGI evaluation and six-condition fairness audit: complete.
- Iterations 7--9, ship review, compact receipts, and delivery records: complete.
