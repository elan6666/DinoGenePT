# Server baseline results

Run date: 2026-08-28 (Asia/Shanghai)

Execution host: `yilangliu@10.24.1.91`; no benchmark was run on the Mac.

## Matched GGI benchmark

Shared gene universe: 10,870 genes with NCBI+UniProt text. The fixed split
retained 249,630 training pairs and 20,342 test pairs for both conditions.

| Embedding | Accuracy | AUROC | Average precision |
|---|---:|---:|---:|
| Paper GenePT, Ada, L2 | 0.71040 | 0.78865 | 0.78530 |
| Latest GenePT, protein knowledge + embedding-3-large, L2 | 0.70627 | 0.79299 | 0.78289 |
| GenePT-Seed, completed NCBI + UniProt corpus + Doubao, L2 | 0.73223 | 0.82099 | 0.81147 |
| GenePT-Seed+GO-EXP, completed corpus + GO + Doubao, L2 | 0.73528 | 0.82411 | 0.81541 |
| Seed-GO-Protein | 0.73415 | 0.82571 | 0.81872 |
| **Seed-GO-ProteinPathway** | **0.74968** | **0.83615** | **0.82859** |
| Seed-GO-ProteinPathway-HPA | 0.73975 | 0.83088 | 0.82407 |

Against latest GenePT, GenePT-Seed improves accuracy by 0.02596, AUROC by
0.02800, and average precision by 0.02858 on this one matched benchmark.
Adding GO-EXP to the completed GenePT-Seed corpus improves accuracy by 0.00305,
AUROC by 0.00313, and average precision by 0.00394. These are absolute metric
deltas (about 0.30, 0.31, and 0.39 percentage points), not relative changes.

Against GenePT-Seed+GO-EXP, the Protein layer changes the metrics by
-0.00113/+0.00160/+0.00330. ProteinPathway improves all three by
+0.01440/+0.01203/+0.01317 and is the best tested condition. Adding HPA still
improves on GenePT-Seed+GO-EXP by +0.00447/+0.00676/+0.00865, but falls below
ProteinPathway; this is a measured ablation result rather than evidence that
HPA knowledge is generally harmful.

## Sensitivity and receipts

Without L2 normalization, GenePT-Seed reached 0.72953 accuracy, 0.81657 AUROC,
and 0.80684 average precision. The conclusion is therefore not created by the
primary normalization choice.

All L2 conditions used 249,630 training pairs, 20,342 test pairs, data
receipt `4a84ee1f...47ec`, and gene-universe receipt `d12aca5c...6e6a`. The
two new Doubao artifacts each contain 10,870 vectors of dimension 2,048 with
100% coverage. Their SHA-256 values are `dbd31d45...426c` (completed corpus)
and `06269245...f21e` (completed corpus + GO-EXP). All fairness fields other
than the embedding condition and vector hash are identical. The
machine-readable receipt is
[`results/GENEPT_SEED_GOEXP_GGI.json`](results/GENEPT_SEED_GOEXP_GGI.json).
The six-condition progressive receipt is
[`results/PROGRESSIVE_KNOWLEDGE_GGI.json`](results/PROGRESSIVE_KNOWLEDGE_GGI.json),
and its vector/coverage gate is
[`results/PROGRESSIVE_KNOWLEDGE_VECTOR_AUDIT.json`](results/PROGRESSIVE_KNOWLEDGE_VECTOR_AUDIT.json).

This is evidence on the released GGI split only. It does not establish general
superiority across the other GenePT tasks, cell types, or downstream models.
No GraD-Pert model was trained or evaluated for this result.
