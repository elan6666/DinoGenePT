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

The latest official embedding improves AUROC by 0.00434 here, while accuracy
and average precision are lower. This is mixed evidence, not a blanket win.

## Pending condition

GenePT-Seed/Doubao is not yet measured because `ARK_API_KEY` is absent on the
server. The credential pasted in chat was treated as compromised and was not
used. No Doubao score is claimed.
