# Real CellFM training IO audit — not training results

Remote receipt: `/data/yilangliu/DinoGenePT/results/cellfm-training-io-v1.json`.
SHA256: `efe771e14156773d66024ae4bd407098a3352903780abdc2335b551732a0934c`.

| Dataset | Cells × genes | Train post cells / epoch | Bags / epoch | Train groups read | Validation / test groups |
|---|---:|---:|---:|---:|---:|
| Adamson | 47,795 × 1,069 | 30,779 | 3,872 | 52 | 6 / 20 |
| Norman | 80,506 × 1,049 | 38,457 | 4,856 | 103 | 24 / 97 |

Train cell counts in this table exclude controls, unlike the inclusive split
counts in the data protocol. Every primary train-post cell is visited once per
epoch. At least one real bag was materialized for every train condition/context;
all validation/test condition populations and their fixed 300 ordered control
draws were read through the prediction/evaluation interfaces.

Measured zero-expression genes remain eligible. Continuous published values and
the complete reduced axes are unchanged. X is held as CSR, only current bags
become dense; upstream uns/DE is not read by this training loader. Array/metadata
order and pinned source/audit/mapping/vocabulary hashes are checked at startup.

Reproduce on server:

```sh
.venv/bin/python scripts/audit_cellfm_training_io.py \
  --cellfm data/perturbation/cellfm-v1 \
  --vocabulary data/pretraining/census-two-atlas-500k-selection-v1/genes.json \
  --output results/cellfm-training-io-recheck.json
```

Use a new output path; existing receipts are preserved. This checks data access
and sampling, not model predictions, GPU capacity, training epochs or scientific
scores. Formal pretraining and LoRA fine-tuning remain outstanding.
