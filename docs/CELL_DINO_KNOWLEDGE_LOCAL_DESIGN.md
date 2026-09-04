# Cell-DINO with GenePT knowledge locals: design proposal

Status: **research design draft; not an implemented or validated model**

## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Origin mode: design plan
- Origin date: 2026-09-04
- Verification status: source-checked design, experimentally unverified
- Version label: cell_dino_genept_design_v1

This document records the candidate cell model for DinoGenePT. The current
implementation remains limited to auditable GenePT gene-text corpora and
embeddings while the replacement model is being designed. Approving this design
will add the cell model back under the DinoGenePT package without restoring the
retired implementation.

## 1. Decisions and open choices

| Topic | Current decision |
|---|---|
| Cell backbone family | **Undecided.** Start from a configurable single-tower cell encoder, but do not yet name a specific architecture as the final backbone. |
| Expression representation | Continuous normalized expression; no expression binning. |
| Self-distillation | Student optimized by gradient descent; same-architecture teacher updated by EMA. |
| Pretraining objectives | Cell-level DINO, gene-token-level iBOT, and continuous masked-expression reconstruction. |
| Cell views | Independent proportional global/local gene crops inspired by DINO v1. The default Student uses 2 local views; fixed-HVG-K, mixed sampling, and local-count changes are ablations. |
| Perturbation views | A primary perturbation anchor plus a variable number of source-only GenePT knowledge locals. |
| Primary perturbation anchor | **Undecided.** Compare a pretrained cell-gene token against the NCBI+UniProt GenePT text embedding. |
| Optional architecture modules | AttnRes, KDA, and LatentMoE are independent plug-ins and ablations, not default components. |
| Inference contract | Use only the selected primary perturbation anchor; optional knowledge locals are training-only unless a separately labelled fused-inference experiment is run. |

The central hypothesis is that a cell encoder can first learn expression and
gene-context structure through multi-view self-distillation, then use multiple
external knowledge views during perturbation training without requiring every
knowledge source at inference.

## 2. Two-stage model

```text
Stage 1: cell representation pretraining

one cell
  |- teacher anchor view: unaugmented or weakly augmented cell
  |- student global crop(s): 40%-100% of model-visible genes
  `- student local crop(s):   5%-40% of model-visible genes
                    |
           shared Cell Transformer
                    |
      DINO-CLS + iBOT-token + masked-expression


Stage 2: perturbation training

control cell + primary perturbation anchor  ---> conditional student ---+
matched perturbed cell, augmented           ---> observed student ------+--> DINO target
control cell + available knowledge local(s) ---> local students --------+
unaugmented train-split perturbed cell       ---> EMA teacher -----------+

primary conditional student ---> expression decoder ---> predicted post-expression
```

The word `local` has two related but distinct meanings:

1. A **cell local** in Stage 1 is a subset of the genes observed in one cell.
2. A **knowledge local** in Stage 2 is a source-specific description of one
   perturbation target, such as GO-EXP or Reactome.

Both implement local-to-global alignment, but they are not interchangeable data
augmentations and must be evaluated separately.

## 3. Cell tokenization and backbone interface

For a visible gene \(g\) with continuous expression \(x_g\), the initial token
is

\[
h_g^{(0)} = E_{gene}(g)
            + \operatorname{MLP}_{expr}(\log(1+x_g))
            + E_{state}(g).
\]

`E_state` distinguishes observed, expression-masked, and other explicitly
defined token states. A `[CLS]` token summarizes the cell. During perturbation
training, one or more projected perturbation tokens are prepended.

Baseline invariants:

- no absolute positional embedding for gene tokens;
- token identity is carried by gene ID, not sequence index;
- cropping and permutation must preserve gene IDs;
- continuous expression is retained rather than discretized into bins;
- batch, donor, cell type, and perturbation labels are not silently injected as
  token features;
- DINO, iBOT, reconstruction, and perturbation decoders are heads outside the
  backbone so the backbone can be compared fairly.

The exact backbone is intentionally unresolved. A provisional full-attention
control may use 12 layers, hidden width 512, 8 heads, FFN width 2048, and a
maximum of 1,200-2,048 visible genes. These are starting values for a matched
control, not the adopted model identity.

## 4. Stage 1 views: adapting DINO crops to cells

### 4.1 Proportional crop definition

Let \(S(x)\) be the model-visible gene set for cell \(x\), after the dataset's
fixed train-only filtering and token-cap policy, and let \(N=|S(x)|\).

For every independently sampled view:

\[
n_v = \operatorname{clip}(\lceil r_v N\rceil, n_{min}, N).
\]

The DINO-v1-inspired ranges are:

- global crop: \(r_g \sim U(0.40, 1.00)\);
- local crop: \(r_l \sim U(0.05, 0.40)\).

This is an analogy to image crop **area**, not a claim that genes form a 2-D
spatial region. Genes are sampled without replacement from the visible set.
Each crop is sampled independently:

- a local crop need not be contained in a global crop;
- two local crops may have little or no overlap;
- two global crops need not contain exactly the same genes;
- effective token counts and pairwise overlaps are recorded for every run.

A small minimum-token safeguard is necessary for sparse cells. It must be fixed
before a formal run and its resulting effective crop ratio must be logged. The
initial comparison should test no floor beyond data validity, a 128-token floor,
and the DINOcell-like aggressive 256-gene view.

### 4.2 Sampling policies to compare

| ID | Global policy | Local policy | Question |
|---|---|---|---|
| V0 | Full unaugmented anchor | Fixed random 256 genes | DINOcell-like reference |
| V1 | Random 40%-100% | Random 5%-40% | Direct proportional DINO-v1 adaptation |
| V2 | Random 40%-100% | Fixed train-HVG K | Does a fixed biological subset outperform proportional crops? |
| V3 | Random 40%-100% | Proportional HVG-biased sampling | Does HVG bias help without fixing local length? |
| V4 | Random 40%-100% | 50% train-HVG-biased + 50% uniform random | Does mixed sampling retain low-expression regulatory information? |

`HVG` must be computed from the training partition only. Per-cell top-expressed
genes are not equivalent to train-set HVGs and must have a separate label if
tested.

### 4.3 Number of Student local views

The default is **2 local views per cell**. `Local head count` in experiment
names means the number of Student local crops, not separate trainable projection
heads. All local views share one Student encoder and one DINO projection head;
changing the count therefore changes view coverage and compute, not model
capacity.

| ID | Local views per cell | Purpose |
|---|---:|---|
| LC0 | 0 | Global-only control; measures whether local-to-global training helps. |
| LC1 | 1 | Lowest-cost local-view condition. |
| **LC2** | **2** | **Default configuration.** |
| LC4 | 4 | Tests whether broader independent gene coverage improves learning. |
| LC8 | 8 | DINO-v1-style high-multi-crop reference; expected to be substantially more expensive. |

The crop distribution and all model parameters remain fixed across LC0-LC8.
Because more local views expose more tokens and produce more Student-Teacher
pairs, report both of the following comparisons:

1. **fixed-update comparison:** same cells, optimizer steps, and epochs; answers
   whether extra views help under the practical training schedule;
2. **token/compute-matched comparison:** adjust cells per batch or gradient
   accumulation so the total processed gene-token count is matched; answers
   whether the gain comes from view multiplicity rather than additional compute.

For every cell, average DINO loss over its valid Student-Teacher view pairs
before averaging the batch. Otherwise LC8 would receive more loss weight than
LC2 merely because it creates more pairs. Record total tokens, realized crop
ratios, pairwise crop overlap, FLOPs or step time, and peak memory.

This Stage 1 local-count ablation is separate from the number of available
Stage 2 knowledge locals. Knowledge-local count is determined by real annotation
availability and is normalized per perturbation condition.

### 4.4 Augmentation order

Crop selection and value masking are different operations:

1. define the model-visible gene set;
2. independently sample a global or local crop;
3. apply expression masking within the selected crop;
4. optionally apply bounded count thinning or Gaussian expression noise;
5. retain every selected gene ID even when its expression is masked.

This separation allows the model to distinguish "gene absent from this view"
from "gene present but expression hidden."

## 5. Student, teacher, and anti-collapse operations

### 5.1 DINOcell-aligned default

The student and teacher have the same encoder and projection-head architecture,
but separate parameters. The student is updated by backpropagation. The teacher
is stop-gradient and updated by an exponential moving average:

\[
\theta_T \leftarrow m\theta_T + (1-m)\theta_S,
\]

with a candidate cosine schedule from \(m=0.996\) toward 1.0.

The DINOcell-aligned default is:

- teacher: an unaugmented target cell;
- student: one or more gene-downsampled views with optional Gaussian expression
  noise and expression masking;
- student and teacher outputs: probability distributions over learned
  prototypes;
- loss: cross-entropy from teacher distributions to student distributions.

DINO v1 instead sends two global crops to the teacher. That alternative should
be preserved as a direct ablation rather than mixed invisibly into the default:

| ID | Teacher input | Student input |
|---|---|---|
| TCH0 | One unaugmented full view | Global and local crops |
| TCH1 | Two independent global 40%-100% crops | The same global crops plus local crops |
| TCH2 | One full view plus one weak 70%-100% crop | Global and local crops |

### 5.2 Centering and sharpening

For teacher logits \(z_T\), DINO probabilities are computed after subtracting a
running center and applying a low teacher temperature:

\[
p_T = \operatorname{softmax}((z_T-c)/\tau_T).
\]

The running center is updated from teacher logits, not from expression values:

\[
c \leftarrow \mu c + (1-\mu)\operatorname{mean}_{batch}(z_T).
\]

Centering limits domination by one prototype dimension; sharpening discourages
uniform assignments. Neither is optional bookkeeping: both are part of the
anti-collapse mechanism. In distributed training, center statistics must be
all-reduced across workers before the EMA update.

DINOcell proposes **conditional centering** to avoid dominant nuisance
covariates organizing the prototype space. For covariate group \(k\):

\[
p_T^{(k)} = \operatorname{softmax}((z_T-c_k)/\tau_T).
\]

We should not assume that conditional centering is always better. It may remove
real biological variation when a group is small or confounded with the target.
The comparison is:

| ID | Center | Constraint |
|---|---|---|
| C0 | One global center | DINO reference |
| C1 | Center per nuisance-covariate group | Never group by perturbation identity or test label |
| C2 | Shrinkage conditional center | Sparse group center shrinks toward global center |

For C2, a candidate shrinkage weight is

\[
\tilde c_k=w_kc_k+(1-w_k)c_{global},\qquad
w_k=\frac{n_k}{n_k+\rho}.
\]

Allowed grouping variables must be declared before training, for example donor,
experimental batch, or a cell context that is explicitly treated as nuisance.
If a dataset contains only one valid group, conditional centering reduces to
global centering.

Collapse monitoring must include prototype entropy, number of occupied
prototypes, teacher/student output entropy, CLS feature standard deviation, and
the largest-prototype share.

## 6. Stage 1 objectives

The three objectives supervise different levels and require separate heads.

### 6.1 Cell-level DINO

Teacher global `[CLS]` outputs supervise Student global and local `[CLS]`
outputs, excluding identical-view pairs:

\[
L_{DINO}=\frac{1}{|P|}\sum_{(t,s)\in P}
H(p_T^{CLS}(t),p_S^{CLS}(s)).
\]

This encourages cell identity and state to remain stable across different gene
subsets and mild measurement noise.

### 6.2 Gene-token-level iBOT

For a gene \(g\) selected in a Student crop and then expression-masked, the
Student token target is the Teacher representation of the same gene:

\[
L_{iBOT}=\frac{1}{|M|}\sum_{g\in M}
H(p_T^{token}(g),p_S^{token}(g)).
\]

Alignment is by gene ID, never by sequence position. A masked gene retains its
identity while its expression value is hidden. Genes absent from a crop have no
iBOT term for that Student view.

### 6.3 Continuous masked-expression reconstruction

A separate numerical head predicts the original continuous normalized
expression:

\[
L_{MEX}=\frac{1}{|M|}\sum_{g\in M}
\operatorname{Huber}(\hat x_g,x_g).
\]

iBOT and masked-expression may share mask positions to reduce computation, but
they must not share their output head: iBOT predicts a latent/prototype target,
whereas masked-expression predicts a number.

The Stage 1 objective is

\[
L_{pre}=\lambda_D L_{DINO}+\lambda_I L_{iBOT}+\lambda_E L_{MEX}.
\]

Start with expression reconstruction active from step zero and warm the DINO
and iBOT weights from zero. Final weights should be chosen using per-loss
gradient norms, not copied from image models.

Required objective ablations:

- P0: masked-expression only;
- P1: masked-expression + DINO;
- P2: masked-expression + DINO + iBOT;
- P3: P2 with proportional global/local crops and the default LC2 local count;
- P4: P3 with the LC0/LC1/LC2/LC4/LC8 local-count ablation;
- P5: the best local count with the fixed-HVG or mixed crop-policy ablation.

## 7. Stage 2 perturbation views aligned with DINOcell

For a train-split perturbed target cell, construct three student-view classes.

### 7.1 Conditional primary view

The Student receives a sampled control cell and one or more perturbation tokens:

\[
v_{primary}=(x_{control},\;A_{primary}(e_g^{primary})).
\]

This is the only view used by the primary expression predictor and is the only
required view at inference.

### 7.2 Observed view

Following DINOcell, sample an independent perturbed cell with the same training
condition and required covariates, then augment it with gene cropping, expression
masking, or bounded noise. It anchors conditional predictions to representations
that can be learned directly from observed perturbation outcomes.

The observed Student cell and Teacher cell should be different biological cells
when possible. This prevents exact-input identity matching.

### 7.3 Knowledge-local conditional views

For every available source \(s\), replace the primary perturbation token with a
source-only embedding:

\[
v_s=(x_{control},\;A_s(e_{g,s})).
\]

These are additional conditional Student views. They never enter the Teacher.
The Teacher remains an unaugmented, covariate-matched, train-split perturbed
cell.

Control, observed, and Teacher cells are population samples, not falsely paired
single cells. DINO matching should therefore be performed on condition/covariate
bag means rather than arbitrary cell indices.

## 8. Primary perturbation-anchor ablation

Calling NCBI+UniProt the permanent `Base` would prejudge the result. The design
therefore distinguishes a **primary perturbation anchor** from optional knowledge
locals.

Two anchor candidates are mandatory:

1. `CellGene`: a gene representation learned in Stage 1 by the cell model;
2. `TextBase`: the GenePT-Seed NCBI+UniProt embedding.

### 8.1 Defining the pretrained cell-gene token

The clean primary definition is the frozen Stage 1 gene-identity embedding after
pretraining, projected through a perturbation adapter. It is available only for
genes in the frozen pretraining vocabulary.

A context-averaged gene prototype is a separate experiment, not an implicit
replacement:

\[
e_g^{proto}=\operatorname{mean}_{x\in D_{external}}
h_g(x).
\]

Such prototypes may use only the external pretraining corpus or training-split
controls. They must never average validation/test perturbed cells.

The gene token is frozen in the strict unseen-perturbation experiment. Fine-
tuning a target-specific gene embedding is a separate transductive ablation
because it can memorize perturbation identities observed during training.

### 8.2 Symmetric comparison

First compare the two anchor sources without GO/Protein/Pathway/HPA so that the
role of the old Base is identifiable:

| ID | Inference anchor | Training-only local | Purpose |
|---|---|---|---|
| A0 | CellGene | None | Does Stage 1 itself learn a useful perturbation prior? |
| A1 | CellGene | TextBase | User-proposed design: pretrained gene token is primary and NCBI+UniProt becomes another view. |
| A2 | TextBase | None | Historical GenePT-primary reference. |
| A3 | TextBase | CellGene | Symmetric reverse-view control. |
| A4 | CellGene + TextBase fusion | None | Both sources required at inference; secondary upper-bound experiment. |

Only after A0-A3 are compared should optional knowledge locals be added to the
winning inference anchor. A4 has a different deployment contract and must not be
presented as directly equivalent to single-anchor inference.

Fairness requirements:

- one matched target-gene universe for A0-A4;
- identical splits, control-cell samples, backbone initialization, decoder,
  optimizer, epoch budget, and random seeds;
- report Stage 1 vocabulary coverage and GenePT text coverage independently;
- fail closed when the selected inference anchor is missing;
- omit a missing training-only local instead of filling it with zeros;
- include a shuffled-TextBase local control to test semantic contribution.

## 9. Knowledge-local corpus contract

Optional source-only views are:

| View | Allowed content | Missing-source behavior |
|---|---|---|
| TextBase | NCBI + UniProt base description | Omit local |
| GO | GO-EXP only | Omit local |
| Protein | InterPro + reviewed UniProt structural fields | Omit local |
| Pathway | Reactome + direct human SIGNOR | Omit local |
| HPA | Tissue, cell-type, and subcellular localization | Omit local |

Optional views do not repeat the primary anchor text. A GO local is GO-only,
not TextBase+GO. Missing sources are not represented by zero vectors, generated
text, nearest-gene vectors, or a learned missing token.

For a combination perturbation, source \(s\) is available only if every target
gene has a real record for \(s\). Multiple target tokens are pooled by a
permutation-invariant mechanism or retained as an unordered token set.

The number of knowledge locals varies by condition. To prevent highly annotated
genes receiving more gradient, either sample one available optional source per
condition per step with a source-balanced cycle, or average all local losses
within condition before averaging conditions. The former is the compute-safe
default.

## 10. Stage 2 losses and inference

The primary view predicts a post-perturbation expression vector, preferably as a
delta from the sampled control:

\[
\hat x_{post}=x_{control}+f_{dec}(h_{primary}).
\]

A DINOcell-aligned decoder uses the Student `[CLS]` state to generate FiLM scale
and shift parameters for gene representations before dense expression
projection. A simpler parameter-matched decoder is required as a control.

For condition bag \(b\), use mean Student and Teacher representations rather
than pretending randomly sampled cells are paired:

\[
\bar z_S^{(b)}=\frac{1}{B_S}\sum_i z_{S,i},\qquad
\bar z_T^{(b)}=\frac{1}{B_T}\sum_j z_{T,j}.
\]

The initial fine-tuning objective is

\[
L_{pert}=L_{pred}^{primary}
       +\alpha L_{DINO}^{primary}
       +\beta\operatorname{mean}_{s\in A(g)}L_{DINO}^{local,s}
       +\gamma L_{DINO}^{observed}.
\]

The primary branch is the only expression-prediction branch in the clean
deployment-aligned baseline. Letting knowledge locals predict expression is a
separate auxiliary-head ablation because those sources are absent at inference.
Perturbation-stage delta-iBOT is also a later ablation; it is not part of the
minimal Stage 2 objective.

## 11. Optional backbone modules

The backbone exposes three independent interfaces:

```text
token mixer:  FullAttention | KDA
depth mixer:  StandardResidual | AttnRes
FFN:          Dense | LatentMoE
```

### AttnRes

Test first. It changes access across network depth without imposing a gene
order. A candidate 12-layer model uses three four-layer blocks and retrieves
from prior block summaries with RMS-normalized learned weights. The added branch
should be initialized near zero so training begins close to the baseline.

### LatentMoE

Test second. A scaled candidate places 8 routed latent experts plus one shared
expert in the top four layers, uses Top-2 routing and latent width 128, and logs
expert load CV, router entropy, and dead-expert rate. It requires a parameter-
matched dense adapter control. Source-type routing must be audited so apparent
specialization is not merely recognition of a GO or Pathway adapter.

### KDA

Keep experimental. Gene tokens have no natural order, whereas KDA is order
sensitive. Any KDA run requires bidirectional processing, periodic full
attention, random permutation augmentation, output realignment by gene ID, and
a permutation audit:

\[
\Delta_{perm}=\|h_{CLS}(x)-h_{CLS}(\pi(x))\|.
\]

At 1,200-2,048 tokens, full attention is a viable control. KDA should be retained
only if it improves perturbation prediction or compute enough to justify its
ordering sensitivity.

No combined AttnRes+KDA+MoE model is promoted before each module wins an
independent, parameter- and compute-aware ablation.

## 12. Leakage and shortcut controls

### Expression leakage

- resolve all perturbation splits before preprocessing;
- exclude validation/test perturbed cells from Teacher inputs, feature
  selection, normalization fitting, and context-prototype construction;
- do not include benchmark test perturbation outcomes in Stage 1 pretraining;
- compute HVGs and variance filters from the permitted training data only;
- keep Teacher stop-gradient and outside the optimizer.

### Knowledge leakage

- pin source versions, dates, evidence codes, and text hashes;
- audit GO, UniProt, Reactome, and SIGNOR provenance for papers underlying the
  benchmark;
- compare real, source-shuffled, and benchmark-publication-sanitized texts;
- describe external biological knowledge as an allowed prior, not as label-free
  evidence.

### Missingness shortcuts

- compare true knowledge locals with source-shuffled locals while retaining the
  same availability mask;
- add an availability-only control that contains no semantic text;
- stratify results by knowledge-source availability pattern;
- normalize local loss per perturbation condition, not per available source.

### Identity shortcuts

- in strict unseen-target evaluation, freeze the pretrained CellGene anchor;
- do not add a trainable target-ID residual to a frozen text anchor;
- report results separately for perturbation targets seen and unseen during
  perturbation training;
- distinguish "gene observed in external expression pretraining" from
  "perturbation outcome observed during downstream training."

## 13. Experiment ladder

Run experiments in this order:

1. **Backbone control:** choose a stable full-attention provisional backbone.
2. **Objective ladder:** P0-P3 to isolate DINO, iBOT, masked expression, and
   proportional cropping.
3. **Local-count ladder:** LC0/LC1/LC2/LC4/LC8, with LC2 as the default and a
   compute-matched sensitivity analysis.
4. **Crop-policy ladder:** compare random proportional, fixed-HVG-K,
   HVG-biased, and mixed sampling at the selected local count.
5. **Teacher/center ladder:** TCH0-TCH2 and C0-C2.
6. **Anchor ladder:** A0-A4 before adding optional knowledge sources.
7. **Knowledge ladder:** add GO, Protein, Pathway, and HPA one at a time, then
   dynamic locals and missingness controls.
8. **Architecture ladder:** AttnRes, parameter-matched dense adapter,
   LatentMoE, and finally KDA.
9. **Combination:** combine only independently successful modules.

Every row fixes dataset versions, split manifests, model-visible genes, output
genes, optimizer, update budget, decoder, cell-bag sampler, and seeds unless the
row explicitly names that factor.

Perturbation expression prediction is a regression task. Primary reporting
should include the benchmark-aligned MSE and correlation metrics, including
delta Pearson and differential-expression-gene metrics where applicable.
Accuracy, AUROC, and average precision remain appropriate for separate GGI or
classification probes, not as replacements for perturbation-regression metrics.

## 14. Unresolved decisions

The following remain open and must not be silently fixed by implementation:

- final Cell Transformer backbone family and size;
- 1,200 versus 2,048 maximum visible genes;
- minimum token floor for proportional local crops;
- exact continuous-expression normalization and count-thinning process;
- global versus conditional versus shrinkage conditional centering;
- raw pretrained gene-ID token versus external-cell contextual prototype;
- expression decoder architecture and full output-gene universe;
- external Stage 1 pretraining corpus and benchmark-cell exclusion rules;
- whether optional knowledge locals remain training-only after the first study.

## 15. Method provenance

- DINO v1 contributes same-architecture Student/EMA-Teacher training,
  centering, sharpening, and two-global-plus-multiple-local multi-crop
  distillation. Its official crop configuration motivates the 40%-100% global
  and 5%-40% local ratios; the gene-set adaptation is ours.
- DINOcell contributes gene-token downsampling, Gaussian expression noise,
  unaugmented-cell Teacher targets, conditional and observed Student views,
  perturbation-token adapters, a FiLM expression decoder, and conditional
  centering. The local copy reviewed for this proposal is
  `/Users/elan/Documents/论文/dino cell.pdf` (bioRxiv preprint posted
  2025-12-19, DOI `10.64898/2025.12.16.694747`).
- iBOT/DINOv2 motivate the separate masked gene-token latent objective. That
  objective is an extension to the DINOcell-aligned design, not a feature being
  attributed to DINOcell.
- DinoGenePT's current GenePT-Seed subsystem supplies versioned Base, GO,
  Protein, Pathway, and HPA text embeddings. It does not yet implement the cell
  model described here.
