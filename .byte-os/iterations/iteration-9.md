# Iteration 9: Matched knowledge ablation

## Evidence

- The comparison auditor accepted all six conditions only after proving all 15
  fairness fields identical.
- Shared protocol: 10,870 genes, 249,630 training pairs, 20,342 test pairs, L2
  normalization, pair sum, logistic regression, random seed 42, and identical
  data/universe/dependency receipts.
- Seed+GO baseline: Accuracy/AUROC/AP 0.73528/0.82411/0.81541.
- Protein: 0.73415/0.82571/0.81872.
- ProteinPathway: 0.74968/0.83615/0.82859.
- ProteinPathway-HPA: 0.73975/0.83088/0.82407.

## Finding and action

The additive knowledge ladder is not monotonic: HPA remains better than
Seed+GO but is worse than ProteinPathway. Documentation now identifies
ProteinPathway as the best tested GGI condition and explicitly avoids claiming
that more text or HPA is generally beneficial.

## Result

The requested Accuracy/AUROC/AP ablation is complete under one fixed protocol.
No GraD-Pert training or evaluation was run.
