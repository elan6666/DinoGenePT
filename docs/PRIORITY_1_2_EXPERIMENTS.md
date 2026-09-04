# Pathway leakage controls and GenePT property replication

## Scope

This delivery asks two questions without training or running GraD-Pert:

1. Does the fixed GGI improvement attributed to the pathway layer survive
   Reactome/SIGNOR decomposition, partner-name controls, and a gene-disjoint
   split?
2. Do the frozen embeddings transfer to four orthogonal gene-property tasks
   used by GenePT under one shared repeated-cross-validation protocol?

All Doubao conditions preserve the exact, case-sensitive 17,730-label master
corpus. Missing source annotations leave the preceding text unchanged. No
source absence is imputed, generated, mapped to a neighbour, or represented by
a zero vector.

## Priority 1: pathway and leakage controls

The pathway condition is decomposed into the following controlled texts:

| Condition | Protein layer | Reactome | SIGNOR partner identity |
|---|---:|---:|---|
| Protein | yes | no | absent |
| Protein+Reactome | yes | yes | absent |
| Protein+SIGNOR | yes | no | real partner symbol |
| Protein+Reactome+SIGNOR | yes | yes | real partner symbol |
| Protein+Reactome+masked SIGNOR | yes | yes | `a partner protein` |
| Protein+Reactome+shuffled SIGNOR | yes | yes | deterministic permutation, seed 42 |

Masking keeps relation direction, effect, and mechanism while removing the
partner symbol. Shuffling keeps the number and form of SIGNOR statements but
breaks the real gene-partner assignment. It is a negative control, not a new
biological corpus.

The control builder preserves the original SIGNOR display case, relation
selection, ordering, and subject-gene symbol. As a regression gate, rebuilding
the unmodified ProteinPathway profile with the new code produces the exact same
corpus SHA-256 as the existing baseline:
`7a952fa7feaf2f19f3e810e8b50532ee0cfd479deb93d62b47dfebedcc9daa3c`.
Only the intended source inclusion or partner position changes in a control.

The frozen SIGNOR snapshot contains 12,637 direct human protein pairs, including
185 self-relations. After the per-gene deterministic eight-relation bound,
10,308 non-self pairs expose the other gene's symbol in at least one actual
corpus SIGNOR section. Self symbols are reported separately because retaining a
relation's subject must not be misclassified as partner leakage. Overlap with
the released GGI split is sparse but label-enriched:

| Split | Label | Pairs | Source overlap | Actual text exposure | Text rate |
|---|---:|---:|---:|---:|---:|
| Train | 0 | 132,561 | 17 | 13 | 0.0098% |
| Train | 1 | 130,455 | 355 | 308 | 0.2361% |
| Test | 0 | 10,600 | 2 | 2 | 0.0189% |
| Test | 1 | 10,848 | 24 | 23 | 0.2120% |

The same detector validates the controls. Masked SIGNOR exposes zero real
SIGNOR partner pairs and zero GGI partner pairs. Deterministic shuffling leaves
377 non-self SIGNOR pairs as accidental text matches; among the GGI rows,
12 positive training pairs and no test pairs remain accidental matches. Thus
masking removes partner identity completely, while shuffling strongly disrupts
identity but honestly retains chance collisions rather than claiming a perfect
zero-overlap permutation.

Therefore, the fixed pair split is retained for comparability but cannot alone
establish leakage-robust transfer. A second protocol partitions the same frozen
10,870 genes into train/test gene sets for seeds 42–51, retains only pairs whose
two genes lie in the same partition, and drops every cross-partition pair.
Twelve allowlisted genes are isolated in the induced GGI graph; they remain in
the frozen gene partition and vector universe but contribute no pair row. Each
receipt records train/test gene hashes, isolated-gene count, pair counts, class
counts, and zero gene overlap. Every embedding sees the same split receipts, L2
normalization, pair sum, and logistic-regression estimator.

An implementation preflight initially partitioned all 11,299 genes present in
the raw GGI files and relied on vector filtering afterward. Those outputs are
invalid for the fixed-universe claim and are retained server-side only under an
explicit `invalid-prefilter-11299` path. A second preflight correctly detected
the 12 isolated allowlisted genes before model fitting. Neither preflight is
included in final metrics. The accepted implementation filters every pair to
the induced graph first, partitions all 10,870 allowlisted genes including the
12 isolated nodes, and hard-checks the same universe/split receipt across every
embedding.

## Priority 2: four GenePT property tasks

The task builder pins the official GenePT reference commit and exact historical
source revisions/checksums for:

- long-range versus short-range transcription factors;
- dosage-sensitive versus dosage-insensitive transcription factors;
- bivalent versus non-methylated promoters;
- bivalent versus H3K4-only promoters.

Ensembl IDs are mapped through the frozen HGNC snapshot. Genes assigned to both
classes of one binary task are removed from both sides. The matched allowlist
contains 519 task genes shared by latest GenePT and the 17,730-label Protein
universe; each condition is filtered through this same allowlist before folds
are constructed. Per-task retained class counts are:

| Task | Positive | Negative |
|---|---:|---:|
| Long-range vs short-range TF | 48 | 11 |
| Dosage-sensitive vs insensitive TF | 110 | 191 |
| Bivalent vs non-methylated | 76 | 31 |
| Bivalent vs H3K4-only | 77 | 72 |

The primary extension uses repeated stratified five-fold cross-validation with
seeds 42–51. Logistic regression and a 500-tree random forest use identical
folds for every embedding. Accuracy, AUROC, and average precision are reported
as fold-level mean, standard deviation, minimum, and maximum. The comparison
auditor rejects mismatched task data, allowlist hash, normalization, dependency
versions, seed grid, fold grid, or task/model grid.

The pinned GenePT notebook was checked directly: it assigns long-range TF,
dosage-sensitive TF, and bivalent genes to label 1; it uses five-fold
`StratifiedKFold`, logistic regression, and random forest. This project
reproduces those four task definitions and class directions, then applies a
matched robustness extension rather than claiming notebook-score identity. The
official notebook uses unshuffled folds, default estimator settings, live
MyGene symbol resolution, and only AUROC. Here, source revisions and HGNC
mapping are frozen, cross-class overlaps are removed, folds are repeated over
ten explicit seeds, RF uses 500 trees, and Accuracy/AP are added. Every
embedding is evaluated under this same extended protocol.

## Reproduction commands

Run materialization only on the configured server. The examples below show one
control; substitute the documented profile/vector labels for the others.

```bash
dinogenept data build-knowledge-texts \
  --base data/corpora/seed-go-master.json \
  --genes data/universes/gradpert-ggi-master-genes.txt \
  --uniprot data/knowledge/current/uniprot-human-reviewed.tsv \
  --interpro data/knowledge/current/interpro.entry.list \
  --reactome data/knowledge/current/reactome.UniProt2Reactome.txt \
  --signor data/knowledge/current/signor.human.tsv \
  --profile protein-pathway-signor-masked \
  --output data/corpora/seed-go-protein-pathway-signor-masked-master.json \
  --manifest data/corpora/seed-go-protein-pathway-signor-masked-master.manifest.json

dinogenept audit-signor-ggi-leakage \
  --data data/ggi \
  --signor data/knowledge/current/signor.human.tsv \
  --corpus data/corpora/seed-go-protein-pathway-signor-masked-master.json \
  --output results/priority-1/signor-ggi-leakage-masked.json

dinogenept benchmark ggi-gene-disjoint \
  --name seed-go-protein-pathway-signor-masked \
  --vectors data/embeddings/seed-go-protein-pathway-signor-masked-master.npz \
  --data data/ggi --genes data/ggi/genes-with-text.txt --normalize \
  --seeds 42,43,44,45,46,47,48,49,50,51 --test-fraction 0.2 \
  --output results/priority-1/gd-masked.json

dinogenept benchmark properties \
  --name seed-go-protein-pathway-signor-masked \
  --vectors data/embeddings/seed-go-protein-pathway-signor-masked-master.npz \
  --tasks data/properties/genept_property_tasks.csv \
  --genes data/properties/common-protein-latest-genes.txt --normalize \
  --folds 5 --seeds 42,43,44,45,46,47,48,49,50,51 \
  --output results/priority-2/properties-masked.json
```

Use `audit-ggi-comparison`, `audit-gene-disjoint-comparison`, and
`audit-property-comparison` to build the final compact receipts. The repository
script `scripts/finalize_priority_experiments.sh` applies the exact complete
condition grid idempotently after all four vector gates pass.

## Results

All four new checkpoints contain 17,730 exact text/model-hash hits, zero
pending rows, and only 2,048-dimensional vectors. Their aligned artifacts each
contain all 17,730 case-sensitive master labels and cover the requested
universe at 100%. The fixed-split auditor accepted all eight rows under one
10,870-gene universe, data receipt, L2 normalization, pair-sum operator,
logistic-regression estimator, seed, and dependency environment.

### Fixed released GGI split

| Condition | Accuracy | AUROC | Average precision |
|---|---:|---:|---:|
| Latest GenePT | 0.70627 | 0.79299 | 0.78289 |
| Protein | 0.73415 | 0.82571 | 0.81872 |
| Protein+Reactome | 0.74216 | 0.83331 | 0.82766 |
| Protein+SIGNOR | 0.73872 | 0.82848 | 0.82106 |
| Protein+Reactome+SIGNOR | **0.74968** | **0.83615** | 0.82859 |
| Protein+Reactome+masked SIGNOR | 0.74432 | 0.83374 | 0.82713 |
| Protein+Reactome+shuffled SIGNOR | 0.74594 | 0.83443 | **0.82860** |
| Protein+Reactome+SIGNOR+HPA | 0.73975 | 0.83088 | 0.82407 |

The named full pathway condition leads fixed-split Accuracy and AUROC, but its
margin over masked/shuffled partner controls is only 0.00374–0.00536 Accuracy
and 0.00172–0.00241 AUROC. The shuffled control is effectively tied on AP.
This makes direct SIGNOR partner-name recall an insufficient explanation for
the whole fixed-split gain.

### Strict gene-disjoint GGI

Each of ten identical split receipts partitions exactly 10,870 genes into
8,696 train and 2,174 test genes with zero overlap. Depending on seed,
171,471–175,327 training pairs and 10,179–11,156 test pairs remain after all
cross-partition edges are dropped. Values are mean ± sample standard deviation
over seeds 42–51.

| Condition | Accuracy | AUROC | Average precision |
|---|---:|---:|---:|
| Latest GenePT | 0.72672 ± 0.00972 | 0.81276 ± 0.00951 | 0.80365 ± 0.01128 |
| Protein | 0.73684 ± 0.01376 | 0.82141 ± 0.01147 | 0.80877 ± 0.01488 |
| Protein+Reactome | 0.73755 ± 0.01323 | 0.82307 ± 0.01096 | 0.81042 ± 0.01441 |
| Protein+SIGNOR | 0.73746 ± 0.01332 | 0.82230 ± 0.01240 | 0.80894 ± 0.01695 |
| Protein+Reactome+SIGNOR | 0.73632 ± 0.01324 | 0.82271 ± 0.01118 | 0.81008 ± 0.01510 |
| Protein+Reactome+masked SIGNOR | 0.73813 ± 0.01225 | **0.82436 ± 0.01062** | 0.81161 ± 0.01460 |
| Protein+Reactome+shuffled SIGNOR | **0.73849 ± 0.01360** | 0.82427 ± 0.01108 | **0.81227 ± 0.01432** |
| Protein+Reactome+SIGNOR+HPA | 0.73758 ± 0.01380 | 0.82324 ± 0.01131 | 0.81050 ± 0.01561 |

The full named pathway row no longer leads once test genes are disjoint.
Reactome-only, masked, shuffled, and HPA rows are close, with differences well
inside the across-seed standard deviations. No significance test was
pre-registered, so these numbers support robustness to direct pair leakage but
not a ranking claim between the closely grouped Seed controls.

### Four GenePT property tasks

The complete receipt contains 64 condition/task/model rows. Each row reports
50 fold scores (ten seeds × five folds), including Accuracy, AUROC, AP,
standard deviation, minimum, and maximum. The concise table below is an
unweighted descriptive mean of the four task-level means; it does not replace
the per-task receipt.

| Embedding | LR Accuracy | LR AUROC | LR AP | RF Accuracy | RF AUROC | RF AP |
|---|---:|---:|---:|---:|---:|---:|
| Latest GenePT | **0.83552** | **0.89314** | 0.92874 | **0.84882** | **0.88304** | 0.92790 |
| Protein | 0.82762 | 0.85591 | 0.92575 | 0.84419 | 0.86957 | 0.93111 |
| Protein+Reactome | 0.82907 | 0.85998 | 0.92824 | 0.84571 | 0.86261 | 0.92947 |
| Protein+SIGNOR | 0.82247 | 0.86209 | 0.92922 | 0.84730 | 0.87483 | **0.93299** |
| Protein+Reactome+SIGNOR | 0.82588 | **0.86558** | 0.93060 | 0.84628 | 0.85823 | 0.92731 |
| Protein+Reactome+masked SIGNOR | 0.82469 | 0.86552 | **0.93119** | 0.84515 | 0.86493 | 0.93085 |
| Protein+Reactome+shuffled SIGNOR | 0.82464 | 0.86229 | 0.92989 | 0.84711 | 0.85736 | 0.92873 |
| Protein+Reactome+SIGNOR+HPA | 0.82546 | 0.86419 | 0.93048 | 0.84763 | 0.85891 | 0.92946 |

The macro average is dominated by heterogeneous tasks, especially the small
48/11 long-range/short-range TF task. Per-task AUROC means show the actual
trade-off:

| Embedding | LR long/short | LR dosage | LR bivalent/non-methylated | LR bivalent/H3K4-only |
|---|---:|---:|---:|---:|
| Latest GenePT | **0.79452** | 0.90730 | 0.91289 | **0.95786** |
| Protein | 0.63333 | 0.92608 | 0.91756 | 0.94666 |
| Protein+Reactome | 0.64207 | 0.92746 | 0.92192 | 0.94847 |
| Protein+SIGNOR | 0.65748 | 0.92803 | 0.91791 | 0.94495 |
| Protein+Reactome+SIGNOR | **0.66600** | 0.92864 | 0.92183 | 0.94583 |
| Protein+Reactome+masked SIGNOR | 0.66419 | **0.92876** | 0.92231 | 0.94683 |
| Protein+Reactome+shuffled SIGNOR | 0.65104 | 0.92820 | 0.92246 | 0.94747 |
| Protein+Reactome+SIGNOR+HPA | 0.65937 | 0.92765 | **0.92309** | 0.94666 |

Latest GenePT remains clearly stronger on long/short-range TF and
bivalent/H3K4-only AUROC. Seed variants improve dosage-sensitive AUROC/AP and
slightly improve bivalent/non-methylated AUROC/AP. Therefore the result is a
task-dependent transfer profile, not universal superiority of either embedding
family.

Compact, reviewable evidence is committed under
[`docs/results/priority-1-2/`](results/priority-1-2/): corpus/checkpoint/vector
gates, the three leakage receipts, the fixed and gene-disjoint comparisons,
the frozen property-task/universe manifests, and the full property comparison.

## Interpretation boundary

These are embedding probes. They do not train or evaluate GraD-Pert, do not
measure perturbation-expression prediction, and do not justify a downstream
causal or biological-superiority claim.
