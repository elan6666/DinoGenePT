# GenePT-Seed

GenePT-Seed keeps the official GenePT NCBI + UniProt text and selected
gene-level evaluation protocols, replacing only the embedding backbone with
`doubao-embedding-vision` through the Volcano Ark Agent Plan endpoint.

This is an independent research variant, not an official GenePT release.

## Comparison contract

| Condition | Text | Embedding |
|---|---|---|
| Paper reference | NCBI | `text-embedding-ada-002` |
| Latest official GenePT | NCBI + UniProt | `text-embedding-3-large` |
| GenePT-Seed | Identical NCBI + UniProt | `doubao-embedding-vision` |

The comparison fixes genes/pairs, labels, splits, classifiers, metrics, and
random seeds. It reports coverage and native dimensions rather than assuming
that a larger vector is automatically better. The primary comparison adds
`--normalize` to L2-normalize every gene vector; omitting it reproduces the
official notebook's native-vector preprocessing as a sensitivity analysis.

## Safety first

An API key pasted into chat is compromised. Rotate it. Never put a key in this
repository or a shell command. Provision a new key privately on the server as
`ARK_API_KEY`. The software reports only whether the variable exists.

Use only the plan endpoint:

```text
https://ark.cn-beijing.volces.com/api/plan/v3
```

The standard `/api/v3` endpoint is intentionally rejected by default because
the subscription console warns that it incurs separate charges.

## Install

Development installation:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
genept-seed --help
```

## Server workflow

The Mac repository is the source of truth. Code is synchronized to
`/data/yilangliu/GenePT-Seed`; downloads, embedding generation, and benchmarks
run only there.

```bash
genept-seed doctor
genept-seed data prepare-genept --output data/genept
genept-seed data prepare-ggi --output data/ggi
genept-seed data ggi-genes --data data/ggi --output data/ggi/genes.txt
genept-seed audit-texts \
  --texts data/genept/NCBI_UniProt_summary_of_genes.json \
  --genes data/ggi/genes.txt \
  --output-selected data/ggi/genes-with-text.txt
genept-seed embed \
  --texts data/genept/NCBI_UniProt_summary_of_genes.json \
  --genes data/ggi/genes-with-text.txt \
  --checkpoint checkpoints/doubao.sqlite3 \
  --output data/embeddings/genept_seed_doubao.npz
genept-seed benchmark ggi \
  --name latest-genept \
  --vectors data/genept/GenePT_gene_protein_embedding_model_3_text.pickle \
  --trusted-pickle --normalize --data data/ggi \
  --genes data/ggi/genes-with-text.txt --output results/ggi-latest.json
genept-seed benchmark ggi \
  --name genept-seed \
  --vectors data/embeddings/genept_seed_doubao.npz \
  --normalize --data data/ggi --genes data/ggi/genes-with-text.txt \
  --output results/ggi-seed.json
```

Use `--limit 20` with embedding generation for the first paid smoke test.

## Tests

```bash
python -m pytest
python -m ruff check .
python -m build
```

## Provenance

- [Official GenePT repository](https://github.com/yiqunchen/GenePT), pinned by
  this project to commit `3602699e7425a7be577771f8f07e218db6c79b9f`.
- [GenePT Zenodo v2](https://zenodo.org/records/10833191), expected archive MD5
  `3f6ce4317e3a0091978ae5cb8fbf05a3`.
- [Gene2vec GGI benchmark](https://github.com/jingcheng-du/Gene2vec/tree/master/predictionData).
  The downloader pins commit `6236e0b21fbc367bf8ff5695ed0cc1443861bd1e`.

Generated data, embeddings, checkpoints, and results are deliberately excluded
from Git. Compact manifests and result tables can be copied back after review.

Use `scripts/sync_code_to_server.sh` for the checksum-based source mirror. It
defaults to a dry run; pass `--apply` only after inspecting the file list.
