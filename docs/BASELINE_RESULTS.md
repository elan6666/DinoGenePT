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
| GenePT-Seed, identical protein knowledge + Doubao, L2 | **0.73223** | **0.82099** | **0.81147** |

Against latest GenePT, GenePT-Seed improves accuracy by 0.02596, AUROC by
0.02800, and average precision by 0.02858 on this one matched benchmark.

## Sensitivity and receipts

Without L2 normalization, GenePT-Seed reached 0.72953 accuracy, 0.81657 AUROC,
and 0.80684 average precision. The conclusion is therefore not created by the
primary normalization choice.

All three L2 conditions used 249,630 training pairs, 20,342 test pairs, data
receipt `4a84ee1f...47ec`, and gene-universe receipt `d12aca5c...6e6a`. The
Doubao artifact contains 10,870 vectors of dimension 2,048 with 100% coverage;
its SHA-256 is `dbd31d45...426c`.

This is evidence on the released GGI split only. It does not establish general
superiority across the other GenePT tasks, cell types, or downstream models.
