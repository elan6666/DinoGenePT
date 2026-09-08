# DinoGenePT

Gene identity update (2026-09-08): native, frozen-HGNC identity audits and an
opt-in knowledge adapter now link old names/approved symbols/Ensembl IDs without
rewriting input axes or existing vectors. Missing identity, source text and
pretraining token coverage are reported separately. See
[implementation, usage and real-data findings](docs/GENE_IDENTITY.md).
Current experiment/runtime state is authoritative in [.byte-os/STATE.md](.byte-os/STATE.md);
older launch plans below do not override the running500k one-epoch campaign.

Current default-model engineering status: native LoRA runner/evaluator and
single-/dual-RTX5090 capacity probes are implemented and tested. Formal2+10+10
epochs are still pending complete pretraining materialization. The factored
KDA candidate failed its BF16 gate; a new order-preserving batched version passes
full-input numerical and single-card tests, with matching two-rank validation
still required. See [launch recipe and evidence](docs/DEFAULT_PRETRAINING_LAUNCH.md)
and [earlier capacity audit](docs/CUDA_CAPACITY_2026_09_07.md).

DinoGenePT is a research project for DINO-style cell representation learning
and perturbation prediction with GenePT knowledge views. The current
package preserves its auditable GenePT knowledge subsystem: it keeps the official NCBI + UniProt text and selected gene-level
evaluation protocols while replacing the embedding backbone with
`doubao-embedding-vision` through the Volcano Ark Agent Plan endpoint.

The previous cell-model implementation was removed for redesign; the project
identity and Python package remain DinoGenePT. This is an independent research
project, not an official GenePT or DINOcell release.

A replacement cell model is being implemented from the design in
[`docs/CELL_DINO_KNOWLEDGE_LOCAL_DESIGN.md`](docs/CELL_DINO_KNOWLEDGE_LOCAL_DESIGN.md).
Native backbone, five-loss pretraining runner, complete-epoch LoRA perturbation
runner and shared evaluator now exist. CPU fixtures cover two pretraining epochs
followed by ten LoRA epochs, exact resume/RNG, two-rank pretraining DDP,
frozen-backbone LoRA and source omissions. The native CellFM
simulation splitter matches pinned upstream fixtures. Released CellFM files,
their reduced gene axes, native splits and exact pretrained-ID mappings are
frozen. A 500k-train/20k-donor-heldout Census selection is frozen and raw-count
shard extraction is underway. Real evaluator-only state is frozen. Independent
knowledge text covers a new 1,777-gene axis union, with API vector supplementation
underway. Formal GPU capacity, pretraining and fine-tuning remain incomplete.
No cell-model experimental results are claimed.

The current plan specifies 12 blocks / 768 hidden width, single-direction hybrid
KDA, fixed DINO/iBOT/KoLeo plus two CellFM reconstruction losses, and local-count
ablations 2/4/8 in the design. The active campaign runs **only the default**:
2 pretraining epochs, then Adamson and Norman each 10 LoRA fine-tuning epochs.
See [data selection](docs/PRETRAINING_DATA_SELECTION.md),
[source fidelity](docs/NATIVE_SOURCE_LEDGER.md), and the [engineering and evaluation plan](docs/ENGINEERING_EVALUATION_DESIGN.md)
for native implementation, two-RTX-5090 DDP, performance statistics, compact
artifacts and GraD-Pert-method perturbation evaluation. CellFM reduced data are
not the GraD-Pert canonical datasets; formula/procedure alignment is not score comparability.

The native training entrypoints are `dinogenept pretrain --config ...` and
`dinogenept finetune --config ... [--resume ...]`; install `.[training,census,dev]`
on the server. They require pinned data/knowledge/checkpoint/evaluation artifacts,
not just a dataset name. See the [current LoRA protocol](docs/CELLFM_LORA_PROTOCOL.md).

## Comparison contract

| Condition | Text | Embedding |
|---|---|---|
| Paper reference | NCBI | `text-embedding-ada-002` |
| Latest official GenePT | NCBI + UniProt | `text-embedding-3-large` |
| GenePT-Seed | Identical NCBI + UniProt | `doubao-embedding-vision` |
| GenePT-Seed+GO-EXP | Completed NCBI + UniProt + bounded GO-EXP | `doubao-embedding-vision` |
| Seed-GO-Protein | Previous + reviewed UniProt structured fields + InterPro | `doubao-embedding-vision` |
| Seed-GO-Protein-Reactome | Protein + Reactome only | `doubao-embedding-vision` |
| Seed-GO-Protein-SIGNOR | Protein + direct human SIGNOR only | `doubao-embedding-vision` |
| Seed-GO-ProteinPathway | Previous + Reactome + direct human SIGNOR relations | `doubao-embedding-vision` |
| Seed-GO-ProteinPathway controls | Previous with SIGNOR partner symbols masked or shuffled | `doubao-embedding-vision` |
| Seed-GO-ProteinPathway-HPA | Previous + Human Protein Atlas summaries | `doubao-embedding-vision` |

The comparison fixes genes/pairs, labels, splits, classifiers, metrics, and
random seeds. It reports coverage and native dimensions rather than assuming
that a larger vector is automatically better. The primary comparison adds
`--normalize` to L2-normalize every gene vector; omitting it reproduces the
official notebook's native-vector preprocessing as a sensitivity analysis.

## Safety first

Never put an API key in this repository, a command argument, or a run log.
Provision it only in the server process environment as `ARK_API_KEY`; the
software reports only whether the variable exists. Rotate any key that has been
shared through chat after the run finishes.

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
dinogenept --help
```

## Server workflow

The Mac repository is the source of truth. Code is synchronized to
`/data/yilangliu/DinoGenePT`; downloads, embedding generation, and benchmarks
run only there.

```bash
dinogenept doctor
dinogenept data prepare-genept --output data/genept
dinogenept data prepare-ggi --output data/ggi
dinogenept data ggi-genes --data data/ggi --output data/ggi/genes.txt
dinogenept audit-texts \
  --texts data/genept/NCBI_UniProt_summary_of_genes.json \
  --genes data/ggi/genes.txt \
  --output-selected data/ggi/genes-with-text.txt
dinogenept embed \
  --texts data/genept/NCBI_UniProt_summary_of_genes.json \
  --genes data/ggi/genes-with-text.txt \
  --checkpoint checkpoints/doubao.sqlite3 \
  --output data/embeddings/genept_seed_doubao.npz \
  --batch-size 10 --max-workers 3 --request-interval 4
dinogenept benchmark ggi \
  --name latest-genept \
  --vectors data/genept/GenePT_gene_protein_embedding_model_3_text.pickle \
  --trusted-pickle --normalize --data data/ggi \
  --genes data/ggi/genes-with-text.txt --output results/ggi-latest.json
dinogenept benchmark ggi \
  --name genept-seed \
  --vectors data/embeddings/genept_seed_doubao.npz \
  --normalize --data data/ggi --genes data/ggi/genes-with-text.txt \
  --output results/ggi-seed.json
```

The completed-corpus GO comparison uses the same selected GGI genes and
checkpoint-safe API client:

```bash
dinogenept data build-go-exp-texts \
  --base data/genept/NCBI_UniProt_summary_of_genes.extended.json \
  --genes data/ggi/genes-with-text.txt \
  --gaf data/go-exp/2026-08-05/HUMAN-uniprot.gaf.gz \
  --obo data/go-exp/2026-08-05/go-basic.obo \
  --output data/genept/NCBI_UniProt_extended_GOEXP_safe_GGI_10870.json \
  --manifest data/genept/NCBI_UniProt_extended_GOEXP_safe_GGI_10870.manifest.json

dinogenept embed \
  --texts data/genept/NCBI_UniProt_extended_GOEXP_safe_GGI_10870.json \
  --genes data/ggi/genes-with-text.txt \
  --checkpoint checkpoints/doubao-goexp-safe.sqlite3 \
  --output data/embeddings/genept_seed_extended_goexp_GGI_10870.npz \
  --batch-size 10 --max-workers 3 --request-interval 4
```

### Extending missing perturbation targets

The official GenePT v2 corpus is frozen and does not cover every current or
historical human gene symbol. Keep the official artifact unchanged and create
an explicitly named extension instead:

```bash
dinogenept data extend-genept-texts \
  --base data/genept/NCBI_UniProt_summary_of_genes.json \
  --genes configs/gradpert_missing_targets.txt \
  --output data/genept/NCBI_UniProt_summary_of_genes.extended.json \
  --manifest data/genept/NCBI_UniProt_summary_of_genes.extended.manifest.json

dinogenept embed \
  --texts data/genept/NCBI_UniProt_summary_of_genes.extended.json \
  --genes configs/gradpert_missing_target_symbols.txt \
  --checkpoint checkpoints/gradpert-extension-doubao.sqlite3 \
  --output data/embeddings/gradpert-missing-targets-doubao.npz \
  --preserve-gene-case
```

The allowlist accepts `SYMBOL [NCBI_GENE_ID|-] [ENSEMBL_GENE_ID]`; pin the
GeneID whenever an old symbol is ambiguous and provide an Ensembl ID for genes
that have no NCBI record. The extension validates each requested symbol against
human NCBI Gene, records renamed symbols and GeneIDs, and appends the reviewed
UniProtKB function comment when available. It never overwrites an official
text. Use one embedding model and width for the entire downstream prior; do not
append 2,048-dimensional Doubao vectors to the frozen 1,536-dimensional Ada
`emb_b` artifact. The reviewed mappings and server hashes are recorded in
[`docs/GRADPERT_EXTENSION.md`](docs/GRADPERT_EXTENSION.md).

The exact Nadig Jurkat 2,809-gene corpus and GO-EXP construction are documented
in [`docs/GRADPERT_JURKAT_GOEXP.md`](docs/GRADPERT_JURKAT_GOEXP.md).

### Complete GraD-Pert knowledge universe

The progressive knowledge artifacts use one 17,730-gene master allowlist: the
13,759-gene union of all five canonical GraD-Pert graph axes, plus the fixed
10,870-gene GGI universe (6,899 genes overlap). All 2,469 unique perturbation
targets are already members of the graph union. Build it from the frozen
GraD-Pert canonical files rather than maintaining a handwritten list:

```bash
dinogenept data build-gradpert-union \
  --gradpert-root /data/yilangliu/GraD-Pert/data-vnext-a942114 \
  --extra-genes data/ggi/genes-with-text.txt \
  --output data/universes/gradpert-ggi-master-genes.txt \
  --manifest data/universes/gradpert-ggi-master-manifest.json

dinogenept data prepare-knowledge-sources --output data/knowledge/current
dinogenept data build-knowledge-texts \
  --base data/corpora/seed-go-master.json \
  --genes data/universes/gradpert-ggi-master-genes.txt \
  --uniprot data/knowledge/current/uniprot-human-reviewed.tsv \
  --interpro data/knowledge/current/interpro.entry.list \
  --reactome data/knowledge/current/reactome.UniProt2Reactome.txt \
  --signor data/knowledge/current/signor.human.tsv \
  --hpa data/knowledge/current/hpa.proteinatlas.tsv.zip \
  --profile protein-pathway-hpa \
  --output data/corpora/seed-go-protein-pathway-hpa-master.json \
  --manifest data/corpora/seed-go-protein-pathway-hpa-master.manifest.json
```

Every output retains every master gene. Optional sections are sparse: a source
is appended only when that exact normalized gene has a real record. An absent
GO, InterPro, Reactome, SIGNOR, or HPA record leaves the previous text unchanged;
it is never replaced by zero text, a nearest gene, or generated biology. Legacy
graph feature labels absent from current NCBI/HGNC/UniProt receive only the
auditable identity string `Gene Symbol <label>` so the graph axis stays complete.
See [`docs/GRADPERT_MASTER_KNOWLEDGE.md`](docs/GRADPERT_MASTER_KNOWLEDGE.md).

After embedding, use `dinogenept data align-axis-vectors` before a strict
downstream prior: the generic NPZ writer sorts symbols, while GraD-Pert requires
the exact frozen graph-axis order. The aligner rejects any missing, extra, or
duplicate symbol and writes a hash manifest.

Use `--limit 20` with embedding generation for the first paid smoke test.
The defaults use the live-verified Agent Plan batch limit of 10. The optional
three-worker configuration above spaces request starts by four seconds; this
completed the full run without sustained-rate HTTP 429 failures.

## Reproduced result

All rows below use the fixed Gene2vec GGI split, the same 10,870-gene universe,
L2 normalization, pair-vector addition, logistic regression, and seed 42.

| Condition | Accuracy | AUROC | Average precision |
|---|---:|---:|---:|
| Latest official GenePT | 0.70627 | 0.79299 | 0.78289 |
| GenePT-Seed | 0.73223 | 0.82099 | 0.81147 |
| GenePT-Seed+GO-EXP | 0.73528 | 0.82411 | 0.81541 |
| Seed-GO-Protein | 0.73415 | 0.82571 | 0.81872 |
| **Seed-GO-ProteinPathway** | **0.74968** | **0.83615** | **0.82859** |
| Seed-GO-ProteinPathway-HPA | 0.73975 | 0.83088 | 0.82407 |

ProteinPathway is best on all three metrics. Relative to Seed+GO it gains
0.01440 accuracy, 0.01203 AUROC, and 0.01317 average precision. HPA remains
above Seed+GO but does not improve on ProteinPathway, so more text is not
automatically better. These results are evidence for this released GGI split,
not for GraD-Pert prediction quality. See
[`docs/BASELINE_RESULTS.md`](docs/BASELINE_RESULTS.md) and the compact
[`GGI comparison receipt`](docs/results/PROGRESSIVE_KNOWLEDGE_GGI.json).

### Leakage controls and orthogonal property tasks

The follow-up study separates Reactome from SIGNOR, masks or deterministically
shuffles SIGNOR partner symbols, adds a repeated gene-disjoint GGI protocol,
and reproduces four GenePT gene-property tasks on one matched gene allowlist.
The exact protocol, pinned label sources, static SIGNOR overlap audit, and
interpretation boundary are documented in
[`docs/PRIORITY_1_2_EXPERIMENTS.md`](docs/PRIORITY_1_2_EXPERIMENTS.md).
All four control vectors reached 17,730/17,730 exact checkpoint hits and 100%
master-universe coverage. The final fixed GGI, ten-seed gene-disjoint GGI, and
64-row property comparison receipts are committed under
[`docs/results/priority-1-2/`](docs/results/priority-1-2/).

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
