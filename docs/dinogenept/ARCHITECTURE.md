# DinoGenePT architecture

## Scope

DinoGenePT currently targets unseen genetic-perturbation expression prediction
under GEARS-style condition splits. It is a new model package beside the frozen
`genept_seed` corpus/embedding toolkit. The implementation does not import
TriShift, Scouter, GraD-Pert, DINOcell, scGPT, or GEARS at runtime.

There is one registered model family: `dinogenept`. Architectural and objective
studies are config ablations of this model. Future model families get a new
folder and registry entry while continuing to use the common dataset and
evaluation contracts.

## Training flow

For one training condition `g`:

1. Sample a small bag of control cells and an independent bag of cells observed
   after perturbing `g`. These are Monte Carlo population samples, not paired
   biological cells.
2. Build the mandatory Student global view from each control cell plus the Base
   target embedding, where Base is NCBI + UniProt text embedded by the selected
   GenePT-Seed text model.
3. Build the unmasked Teacher view from post-perturbation expression. Average
   Teacher and Student logits within the condition bag before DINO matching.
4. If optional annotations exist, choose at most one source for that condition
   and epoch using a deterministic source-balanced cycle. Build another Student
   view from the same control representation plus that source-only prior.
5. Independently replace selected post-expression values with a learned mask
   embedding for delta-iBOT. Gene identity remains visible; the token target is
   post-minus-control expression. The default gene set is fixed from
   training-split variance before optimization. Selecting genes from the
   unmasked post cell is retained only as an explicit shortcut control.
6. Predict the complete post-perturbation expression vector from the Base view.
   The decoder adds a learned delta to the sampled control expression.
7. Update the Teacher backbone and DINO head by EMA. The Teacher has no prior
   adapter, expression decoder, or optimizer gradients.

At primary inference, only sampled controls and Base are used. Predictions are
averaged over controls. Optional knowledge improves representation learning but
does not become an inference-time coverage requirement.

## Knowledge views

| View | Text encoded before training | Availability rule | Inference |
|---|---|---|---|
| Base | NCBI + UniProt functional description | Mandatory for every target | Yes |
| GO | GO-EXP only | Omit when any target lacks GO-EXP | No |
| Protein | InterPro + reviewed UniProt structure fields only | Omit when any target lacks the source | No |
| Pathway | Reactome + direct human SIGNOR only | Omit when any target lacks the source | No |
| HPA | Tissue, cell-type, and subcellular localization only | Omit when any target lacks the source | No |

Each source has its own input adapter but all views share the cell Transformer
and DINO projection head. Separate DINO heads would make cross-view alignment
less constrained, so “multiple local heads” are implemented as variable local
views with source-specific input adapters and one shared objective head. All
four adapters remain instantiated in single-source ablations; only view
scheduling changes, so the four local rows and the dynamic-local row have the
same parameter surface.

No optional corpus includes Base text. No missing source is represented by a
zero vector or learned missing token. Source-shuffled and availability-only
controls preserve the missingness pattern while destroying semantic content.

## Objectives

The default training objective is a weighted sum of:

- population prediction MSE on the condition-bag mean;
- centered, temperature-scaled condition-level DINO cross entropy for Base;
- the same DINO loss for at most one available optional local per condition;
- delta-iBOT MSE at masked gene-token positions;
- an optional MoE balancing term.

Local DINO losses are concatenated and averaged over selected perturbation
conditions, not averaged once per source; sparse sources therefore do not gain
extra per-condition weight. Direction-aware prediction, KoLeo, shuffled
delta-iBOT targets, and post-Top-K iBOT selection are explicit ablations. KoLeo
is off by default because biologically related perturbations must be allowed to
stay close. Training receipts include feature standard deviation, KoLeo-active
fraction, and per-layer MoE entropy, load coefficient of variation, and dead
expert fraction.

## Leakage boundary

- The perturbation-condition split is resolved before training.
- Teacher post-expression comes only from training conditions. Validation and
  test post-expression never enter the Teacher, Student, priors, router, or
  feature selection.
- Gene variance selection uses training-condition cells and the shared control
  pool only, in row chunks; it does not use test-condition outcomes.
- Base and optional priors are external gene knowledge. Relation sources still
  require benchmark-overlap audits and sanitized variants before inductive
  claims.
- Evaluation is Base-only and condition-macro. Results are additionally
  stratified by optional-source count and source pattern to expose annotation
  density shortcuts.
- Formal runs require a frozen split manifest and reject deterministic fixture
  priors. Base coverage fails closed; optional coverage remains sparse.
- Run reuse recomputes byte hashes for the dataset, split manifest, every prior,
  optional pretrained checkpoint, executable source, and material outputs. A
  `complete` receipt is written only after metrics, resolved config, and model
  checkpoint are atomically materialized.

## Cell Transformer

The backbone embeds selected gene identities and expression values, prepends a
CLS token, and optionally prepends target-prior tokens. It has no absolute gene
position embedding. Standard full self-attention is the default.

Available config ablations are:

- block attention residual retrieval between Transformer blocks;
- top-layer residual dense adapters;
- condition-routed sparse latent MoE adapters with actual top-k expert dispatch;
- auxiliary-loss or auxiliary-loss-free quantile routing balance;
- SiTU-style gated feed-forward layers;
- bidirectional KDA layers interleaved with global attention.

The KDA implementation is a readable recurrence for correctness experiments,
not a fused production kernel. KDA is order-sensitive. DinoGenePT orders gene
tokens by absolute expression rank, but KDA remains an experimental ablation and
must be reported with the permutation audit. In the final server smoke, full
attention had maximum CLS difference `5.96e-7` after token permutation and
passed the `1e-5` gate; hybrid KDA had difference `0.20075` and intentionally
failed that invariance gate.

## Extension contracts

### New model

Add `src/dinogenept/models/<model_id>/` and expose a plugin implementing the
common model protocol: construction, trainer/Teacher creation, split
prediction, and permutation audit. Register that plugin in
`src/dinogenept/registry.py`. The orchestration layer no longer imports
DinoGenePT-specific Teacher or trainer classes, so another model can retain the
same dataset adapters, output hierarchy, input receipts, and evaluator.

The current checkpoint loader accepts common state-dictionary containers,
filters exact name-and-shape matches, records the checkpoint SHA-256 and match
counts, and enforces a configurable minimum matched-parameter fraction. It is
not an scGPT-specific key mapper; a claimed scGPT initialization requires a
separate adapter and its own compatibility receipt.

### New dataset

Add `src/dinogenept/datasets/<dataset_family>/`, normalize it into
`PerturbationData`, register a stable dataset ID, and add one config directory
under `configs/datasets/`. Dataset IDs are output identities; K562 and RPE1 are
therefore separate even though they share a Replogle loader family. The current
adapter contract assumes one shared control pool. A dataset with cell-type,
donor, or batch-matched controls must add those context fields and a matching
control-access policy before use; silently pooling such controls is forbidden.

### New ablation

Add one YAML overlay under
`configs/experiments/dinogenept/ablations/`. Do not copy the model. Matrices
preflight a common model ID, dataset config, evaluation config, and seed before
starting any row.

## Performance and execution safeguards

- AnnData can stay backed; variance is computed in row chunks and only the
  selected cell-by-gene block is materialized.
- Datasets are cached once within a sequential matrix.
- AMP uses BF16 on CUDA. MoE dispatch evaluates only selected experts.
- Optimizer gradients use `set_to_none`; GPU peak memory is reset and recorded
  for each row.
- The runner checks free memory and applies a per-process memory fraction before
  model construction.
- Server-enforced configs also require the canonical work root,
  `CUDA_VISIBLE_DEVICES=0`, a CUDA device, and record the resolved physical GPU
  index and UUID.
- Matrix rows never execute concurrently. Completed receipts are reused only
  after all input and output hashes match.

## Current evidence boundary

The committed implementation has passed a 21-row, one-epoch functional smoke
on Adamson mini, including checkpoint save/validation and a 21/21 idempotent
reuse check. It has not completed a full scGPT/GEARS reproduction, formal
GenePT-Seed source-only embedding build, multi-seed comparison, or trained-model
biological evaluation. Smoke metrics must not be used to select an
architecture. Architectural comparisons that change parameter count or active
compute also need parameter- or FLOP-matched controls in the formal study.
