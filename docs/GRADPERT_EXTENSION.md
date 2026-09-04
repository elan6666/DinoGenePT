# GraD-Pert missing-target corpus extension

## Scope

This extension addresses the 17 perturbation symbols that are absent from both
the frozen GraD-Pert GenePT `emb_b` artifact and the official GenePT v2
`NCBI_UniProt_summary_of_genes.json`. It does not modify either official
artifact.

The generated text keeps the exact perturbation symbol as its key, resolves it
to an explicitly pinned human NCBI GeneID, records any current NCBI symbol, and
adds the reviewed UniProtKB function comment when available.

## Reviewed identifier mappings

| Perturbation symbol | NCBI GeneID | Current NCBI symbol |
|---|---:|---|
| C12orf45 | 121053 | NOPCHAP1 |
| C1orf109 | 54955 | AIRIM |
| C9orf16 | 79095 | BBLN |
| CCDC130 | 81576 | YJU2B |
| DDN | 23109 | DDN |
| FAM207A | 85395 | SLX9 |
| FRG2 | 448831 | FRG2 |
| INTS2 | 57508 | INTS2 |
| MBTPS1 | 8720 | MBTPS1 |
| PHB | 5245 | PHB1 |
| POLR2B | 5431 | POLR2B |
| SPATA5 | 166378 | AFG2A |
| SPATA5L1 | 79029 | AFG2B |
| TDGF1 | 6997 | CRIPTO |
| WDR61 | 80349 | SKIC8 |
| WDR92 | 116143 | DNAAF10 |
| ZNF720 | 124411 | KRABD5 |

All 17 records have NCBI summaries. Fifteen have reviewed UniProtKB function
comments; FRG2 and ZNF720/KRABD5 have reviewed accessions but no function
comment in the queried field.

## Server receipts

Generated on the server under `/data/yilangliu/DinoGenePT`:

| Artifact | SHA-256 |
|---|---|
| `data/genept/NCBI_UniProt_summary_of_genes.extended.json` | `e7c76d017a016617aa5ff04b78bf0f5d639b8738b6a97e3a5172b5f791d1d266` |
| `data/genept/NCBI_UniProt_summary_of_genes.extended.manifest.json` | `4188da2edcf13cbb96043874a93e0bff0f6ffd4b8aec1d6b1ee57fc62409a9cd` |
| `data/embeddings/gradpert-missing-targets-doubao.npz` | `3b16ad4ec193685cf3ef5221d19f04db0684b24ba5df1eff1f906399a914b8e3` |
| `data/embeddings/gradpert-missing-targets-doubao.npz.manifest.json` | `5b3dfa869dcf0c1f58ecb4b9dad08d38b000c12d2779b4ba4c07bf2a43ed5c81` |

The vector artifact contains exactly the 17 requested case-sensitive keys,
uses `doubao-embedding-vision`, has shape `17 x 2048`, and contains only finite
float32 values.

## Integration boundary

The 17-vector artifact is a verified supplement, not a standalone replacement
for GraD-Pert `emb_b`. The frozen artifact is `1536`-dimensional Ada output;
these vectors are `2048`-dimensional Doubao output. Concatenating or filling
only missing rows would confound the model comparison and is forbidden.

A GraD-Pert-ready GenePT-Seed condition must encode the complete selected graph
axis with the same extended corpus, Doubao model, dimension, normalization, and
case-preserving symbol policy. It should be registered as a new prior and hash
contract, leaving the frozen `emb_b` ablations unchanged.
