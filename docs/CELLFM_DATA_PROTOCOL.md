# CellFM data selection and reproduction gates

Updated 2026-09-07. Latest user decision supersedes the earlier first-run
five-dataset campaign: CellFM Adamson/Norman downstream, 10 LoRA epochs each;
pretraining source is now delegated beyond CellFM/scFoundation, excluding scGPT.
Two pretraining epochs, default model only, no ablation runs.

## Pretraining

CellFM is a candidate, no longer a mandatory source. Never use scGPT's data
route or a downstream PBMC dataset relabeled as CellFM pretraining. The paper describes 100 million human
cells. Its data-availability section points to Supplementary Table S4 and
Zenodo record 15138665. The inspected archive contains downstream datasets;
availability of the complete processed pretraining corpus is **not yet verified**.
Do not interpret this as proof that the corpus is unavailable.

Before materialization/training, resolve downloadable shards or accessions,
licenses, checksums, gene mapping, counts scale, QC, actual cell counts, and
study/donor overlap with downstream validation/test data. Freeze a manifest.
If reconstruction from raw public studies is required, document this explicitly;
it is not automatically an exact copy of the author's processed corpus.
Download and process data on the server only.

## First perturbation campaign

The paper's genetic perturbation experiments use **Adamson and Norman**.
The inspected Zenodo archive lists `adamson.h5ad` and `norman.h5ad`.
No additional perturbation datasets are automatically scheduled.

The official notebook at commit
`bfed59c0e34103231165d69b97927ecc888d623c`,
`tutorials/Perturbation/GenePerturbation.ipynb`, contains:

| Setting | Observed notebook value |
|---|---|
| data_name | norman-1000 |
| split / seed | simulation / 3 |
| train_gene_set_size | 0.75 |
| epochs | 15 |
| batch / gradient accumulation / test batch | 10 / 5 / 32 |
| hidden size / learning rate | 512 / 0.005 |
| model_type / finetune_method | emb / frozen |

These are notebook values for its GEARS-based experiment, **not confirmed
universal Adamson settings or suitable defaults for our Transformer**.
The notebook's `norman-1000` path is not the archive filename: resolve its
preprocessing and axis selection before claiming exact reproduction. A value
of 0.75 in `train_gene_set_size` is not a universal cell-level train fraction.
Resolve GEARS implementation/version, split membership, DE definitions,
gene-embedding loading and any additional publication seeds from source.

CellFM evaluates embeddings within GEARS. Our knowledge-local perturbation
model is a distinct method on the same data protocol; it must not be labeled
an exact reproduction of CellFM+GEARS. Do not introduce upstream runtime model
imports contrary to the native-package requirement.

## Evaluation

Retain independently implemented GraD-Pert metric formulas as additional
metrics. For paper reproduction, also freeze the CellFM/GEARS top-20 DE Pearson
delta and MSE implementations and aggregation. Never relabel one as another.
Keep protocol IDs, data/gene-axis/split hashes, references and control sampling
in result receipts. CellFM-split and GraD-Pert-split scores are separate tracks.
Choose validation-only checkpoint selection before formal runs.

## Evidence

- [Paper and data availability](https://www.nature.com/articles/s41467-025-59926-5)
- [Official data archive](https://zenodo.org/records/15138665)
- [Pinned perturbation notebook](https://github.com/biomed-AI/CellFM/blob/bfed59c0e34103231165d69b97927ecc888d623c/tutorials/Perturbation/GenePerturbation.ipynb)

Status: source audit in progress; no CellFM data training has been launched.
