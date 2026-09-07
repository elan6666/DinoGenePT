# DinoGenePT: current model and experiment design

Status: **native backbone, five-loss runner and knowledge-local model CPU-tested; formal data/CUDA integration
and formal experimental validation pending**.
Updated 2026-09-07, version `dinogenept_design_v2`. This replaces v1;
historical GenePT embedding results remain unchanged.

## 1. Locked decisions

| Item | Current decision |
|---|---|
| Package | Native dinogenept code; preserve the existing GenePT subsystem |
| Scale | 12 backbone blocks, width 768, at most 2048 gene input tokens |
| Main mixer | Single-direction KDA + global attention: [KDA,KDA,KDA,Global] x 3 |
| Global layer | Kimi-style Gated MLA, explicitly adapted to noncausal visible-gene attention; ordinary MHA is a separate control |
| Short convolution | Off by default; on only in its ablation |
| Kimi fidelity | Source-faithful unchanged mechanisms; record cell-task/scale adaptations |
| Values | Continuous, no expression binning |
| Pretraining candidates | Exclude zero-expression genes |
| Pretraining data | Public continuous-expression corpus, autonomously selected for dual5090; scGPT route excluded; source/overlap audit required |
| First perturbation campaign | CellFM Adamson and Norman data/protocol; not the previous five-dataset campaign |
| Fine-tuning candidates | Include measured zero-expression genes; distinguish padding |
| Fine-tuning parameters | LoRA on frozen pretrained backbone plus trainable new task/source heads; not full-backbone updates |
| Active training budget | Default only: 2 complete pretraining epochs; Adamson and Norman each 10 fine-tuning epochs; no ablations now |
| Views | Two teacher globals; two student globals and default two locals |
| Local-count sweep | **2, 4, 8**; shared encoder and projection head |
| FFN expansion sweep | **1 and 4**, no expansion-2 run |
| Pretraining losses | DINO + iBOT + KoLeo + both CellFM expression losses |
| Loss experiments | **No weight sweeps or loss-removal ablations in the active campaign** |
| Hardware | Two RTX 5090; require measured capacity before formal runs |
| Evaluation | Independently implement the exact frozen GraD-Pert five-dataset contract |

ERetNet is the CellFM structural reference, not the main experiment. Kimi
alignment does not mean silently enabling all K3 modules or copying its scale.
Each FFN/residual experiment must resolve its selected variant explicitly.

## 2. Cell inputs and verified reference behavior

For raw-count pretraining only, normalize before cropping:
\[
x_g=\log(1+10^4c_g/T),\qquad h_g^0=E_g+E_x(x_g).
\]
Do not renormalize each crop or log-transform processed values again.
Perturbation inputs preserve the selected protocol's per-dataset processed
scale, not an extra CellFM transform. CellFM and GraD-Pert protocols are distinct.

CellFM's continuous value encoder has intermediate width 256:
\[
u=\operatorname{LeakyReLU}(xW_1),\qquad
E_x(x)=\operatorname{softmax}(uW_2+\alpha u)W_3.
\]
Alpha starts at zero. Masked expression uses E_gene + E_mask. Measured zero
uses E_gene + E_x(0) and remains valid. Padding contributes neither loss nor
state update; never derive fine-tuning validity from expression > 0.

Pretraining selects at most 2048 nonzero genes without replacement, default
weights proportional to log1p(count), as in CellFM. Compare uniform sampling
while retaining continuous expression: this borrows a sampling idea, not
scGPT binning. Sampling and ordering are independent. Default ordering is
declared gene-ID order, never hidden-target expression rank. Fine-tuning
uniform capped sampling over measured mapped candidates is the proposed
policy; zero genes have nonzero selection probability and the full output
axis remains fixed.

CellFM source reference: bfed59c0e34103231165d69b97927ecc888d623c.

- Prepare.mask(random=False) selects 20% reconstruction targets R, actually
  hides 80% of R, and leaves the remaining targets unchanged. Guard empty R.
- iBOT supervises truly hidden positions M only; numerical losses use R.
- model.py::encode restricts source keys to unmasked positions in the first
  half, then admits all valid positions. The 12-block adaptation is six/six.
- MHRetention uses global Q(K^T V), ReLU Q/K, headwise SRMSNorm, SiLU output
  gating and length scaling, not a causal scan.
- SGLU has intermediate width d, not 4d. Recalculate depth-dependent DeepNorm
  initialization/scales for 12 blocks.
- SCrna(prep=True) also filters high detected-gene counts and transforms high
  library counts. Do not assume the example is a complete pretraining recipe
  or apply it silently to processed benchmark matrices.

## 3. Views, teacher and fixed pretraining objective

Independently sample from the capped visible set U:
\[
n_v=\max(1,\lceil r_v|U|\rceil),\quad
r_G\sim U(0.4,1),\quad r_L\sim U(0.05,0.4).
\]
No 128-token floor by default. Record lengths and overlaps; locals need not
be contained in globals. Teacher receives two clean globals. Student receives
their masked counterparts plus K locals, K in {2,4,8}.

Teacher starts from student, is stop-gradient/eval-mode, and follows cosine
EMA momentum 0.996 toward 1. Proposed center momentum is 0.9, student
temperature 0.1, teacher temperature 0.04 to 0.07 over the first 10% of
optimizer updates. Maintain separate cell/token centers; reduce sums/counts
across ranks. These are project initial settings, not CellFM defaults.

DINO uses teacher softmax((logits-center)/temperature) targets, excluding the
corresponding same-global pair. Average its 2+2K valid pairs per cell, then
the batch. The total loss scale stays stable but the local-pair fraction
increases with K; report this rather than claiming weighting is unchanged.
No alternative normalization sweep is active.

iBOT matches each truly hidden gene to the clean teacher of the same global
by gene ID. Average per cell, across globals, then batch. Cross-crop token
distillation is not default and would require explicit intersections.

CellFM numerical heads:
\[
\hat x_g^{gene}=W_2\operatorname{LeakyReLU}(W_1h_g),\qquad
\hat x_g^{cell}=\sigma(W_cE_g)^Th_{CLS}.
\]
\[
L_E=\operatorname{mean}_{g\in R}(\hat x_g^{gene}-x_g)^2,\qquad
L_C=\operatorname{mean}_{g\in R}(\hat x_g^{cell}-x_g)^2.
\]
The sigmoid mapping acts on the gene embedding, not CLS. Average each MSE
per cell and across the two student globals; do not scale it with local count.

KoLeo acts on student backbone global CLS before the DINO MLP. Normalize
u_i = h_i / max(norm(h_i), eps), then find nearest OTHER cell n(i) within
each global-view batch:
\[
L_K^{(a)}=-\operatorname{mean}_i
\log(\|u_i-u_{n(i)}\|_2+10^{-8}),\qquad
L_K=L_K^{(1)}+L_K^{(2)}.
\]
Do not repel the two views of the same cell. This is unlabeled spreading,
not orthogonality. Use FP32 normalization/distances. Default neighborhood is
the per-rank forward microbatch, following the inspected DINOv2 behavior;
gradient accumulation does not enlarge it. Require at least two distinct
cells per rank and report neighbor-pool size.

\[
\boxed{L_{pre}=L_E+L_C+w(t)[L_D+L_I+0.1L_K]}
\]
w(t) rises from zero to one over the first 10% of optimizer updates.
Terminal weights **1,1,1,1,0.1** are fixed. Log raw/weighted losses and
occasional gradient diagnostics; no coefficient or objective-removal sweep.

Proposed cell/token head sizes respectively:
d -> 2048 -> 256 -> 8192 and d -> 2048 -> 256 -> 4096,
with source-audited DINO projection semantics. Prototype count is independent
of 2048-dimensional text embeddings.

## 4. Architecture and view experiments

| ID | Factor | Settings |
|---|---|---|
| AT-main | Main mixer | Single-direction KDA + Gated MLA, 3:1 |
| AT-bi | Direction | Same layout; shared-parameter forward/reverse scans, realign and average |
| AT-ere | Reference | CellFM ERetNet |
| AT-ere-hybrid | Hybrid control | ERetention + identical global-layer placement |
| AT-full | Reference | Noncausal full attention |
| AT-pure | Supplemental | Pure KDA; not the official full Kimi architecture |
| CONV | Short convolution | Off default vs proposed kernel size 3 |
| DIM | Hidden width | 768, 1024, 1536, 2048; fixed 12 blocks |
| FFN | Dense expansion | 1 vs 4; activation fixed |
| RES | Depth mixing | CellFM DeepNorm vs Block AttnRes; Full AttnRes supplemental |
| MOE | FFN structure | Dense vs scaled LatentMoE; auxiliary balancing vs QB; SiTU variant |
| LC | Local crops | 2 default, 4, 8 |
| SAMPLE | Selection | CellFM weighted vs uniform continuous-value sampling; fixed-HVG crop control |

Retain official KDA decay, delta update, Q/K normalization and output gates
unless explicitly adapted. KDA stays order-sensitive without short convolution.
The proposed early-layer mask adaptation disables hidden-token writing AND
decay (beta=0, alpha=1), but allows reads. Padding never updates state.
Later layers can exchange hidden representations without seeing true values.
Short-convolution variants must prevent masked-value contamination of Q/K/V.

The final global layer lets CLS read all valid genes. Pure-KDA CLS readout
must be specified before its run; a front CLS in a causal-only stack is not a
whole-cell representation. Two directions do not restore permutation invariance.
Audit permuted CLS/predictions on exactly the same genes, dropout disabled.

CellFM reference head width is 32 (24 heads at d=768). Do not silently force
that onto Kimi KDA/MLA: key/value head widths and latent ranks must be read from
pinned code and scaled explicitly. DIM results must identify fixed vs scaled
head widths/ranks. Same tokens do not mean equal FLOPs; report both fixed-update
and measured compute/time comparisons.

AttnRes uses learned pseudo-query dot RMS-normalized sources, softmax over
depth and original values. Block summaries accumulate sublayer outputs and
the current partial sum, retaining the input embedding source. Do not replace
this with a near-zero residual adapter or layer DeepNorm scaling on top.
Twelve transformer blocks contain attention and FFN sublayers; group boundaries
must say which they count. Three transformer blocks/group is a proposed
small-model setting, not an official K3 default.

MoE is config-selected, not silently active. A capacity-screening proposal is
8 routed experts, Top-2, one full-width shared expert, latent width d/4, top
four blocks. Shared experts operate at full width; routed aggregates are
RMS-normalized before up-projection. Preserve QB timing; freeze bias at inference.
Define teacher non-gradient buffer synchronization. Measure routing rather
than asserting biological experts. Include distinct total-parameter-matched
and active-compute-matched Dense controls.

## 5. Perturbation knowledge views

Stage-1 crop count is independent of Stage-2 source availability.
\[
z_g^s=\operatorname{DoubaoEmbedding}(\operatorname{Text}_s(g)),\quad
v_s=[CLS,A_s(z_g^s),\operatorname{CellTokens}(x_{ctrl})].
\]
The source vector is 2048-dimensional; the adapter maps to model width.

| Source | Independent text |
|---|---|
| TextBase | NCBI + UniProt function descriptions |
| GO | GO-EXP only |
| Protein | InterPro + UniProt structural fields |
| Pathway | Reactome + SIGNOR |
| HPA | Tissue, cell-type and subcellular descriptions |

Historical cumulative vectors are not source-only locals. Validate text hashes
before reuse. Missing source means omit local, not fabricate/zero-fill.
For combinations, every target must have the source. Current design uses all
available locals, normalizes within condition, and processes them in bounded
chunks for memory. A source-count cap would be an explicit sampling change.

Compare frozen pretrained gene-ID anchor (CellGene) vs TextBase: each alone
and with the other as training-only local. Context-averaged prototypes are a
separate candidate. Missing required anchor fails coverage, not silent row
removal. Train-only embedding adaptation is not automatically transductive;
held-out outcome exposure is a separate issue.

Conditional student sees control + main anchor; knowledge students replace
the anchor; an observed student sees augmented train-post cells; teacher sees
independent matched train-post cells. Control/post are population samples,
not true pairs. Proposed training bags use eight controls/eight post cells,
condition/context matched. Average teacher probabilities, not interchangeable
mean logits. This bag adaptation is ours, not claimed as published DINOcell code.

A fixed CLS-conditioned gene decoder predicts the full canonical axis via
pred_post = input_ctrl + predicted_delta. Proposed fine-tuning loss is primary
bag prediction MSE plus 0.1 each for primary, knowledge-mean and observed
DINO terms. No loss sweep is active. Pin decoder/optimizer/epoch settings before
launch. The active user budget is 10 epochs per dataset with LoRA on the frozen
backbone plus trainable new heads; CellFM's GEARS notebook uses 15 epochs, which
is a reference rather than our current budget.

### Default implementation choices (no ablation launched)

For the delegated default run, choose **TextBase as the primary anchor** and
GO/Protein/Pathway/HPA as available training-only locals. CellGene remains the
explicit alternative anchor config, not a silently added local. TextBase must
cover every perturbation target. Source vectors are detached, individually
L2-normalized, then mapped by source-specific bias-free Linear(2048,768) and
LayerNorm. Combination targets are sorted by frozen vocabulary ID with the
matching vector rows reordered together. This is our canonical-order policy.

Use LoRA rank16, alpha32, dropout0.05 at the enumerated mixer projections.
Frozen backbone weights include gene/value embeddings, gates and norms; the
pretrained cell projection head and new source/decoder heads are trainable.
The cell head is initialized from pretraining, not asserted to be a new random
head. Teacher is an EMA copy, always eval/stop-gradient, with its own center.

The native low-rank full-axis decoder (`cell/perturbation.py`) is:

\[
q_{bg}=\sigma(W_g e_g + W_x x^{ctrl}_{bg}),\quad
\Delta_{bg}=s\,q_{bg}^{\top}W_c h_b/\sqrt{128},\quad
\hat x_{bg}=x^{ctrl}_{bg}+\Delta_{bg}.
\]

Here `e_g` is the frozen pretrained gene-ID embedding, `h_b` the conditional
Student CLS, `W_g,W_c:768→128`, `W_x:1→128`, and scalar `s` starts at one.
Inputs/targets use the same continuous normalized expression scale. Predictions
are unconstrained deltas; no undocumented clipping or re-normalization. This
factorization avoids constructing a full B×G×768 expression-token tensor and
is a declared new decoder, **not** a claim about CellFM or DINOcell's decoder.

For independent condition/context-matched bags, use
`MSE(mean(pred),mean(train_post))`, plus 0.1 times each of primary, available-local
mean and observed-view DINO. Teacher target is the mean of per-cell softmax
probabilities, with temperature0.07, center momentum0.9; Student temperature0.1.
Missing all optional locals contributes zero knowledge loss, not a fake zero
embedding. EMA follows 0.996→1 after optimizer steps. No post-expression argument
exists in `predict`; full post bags can only enter the training forward path.
`datasets/populations.py` now constructs condition/context-matched bags.
Every train post cell is a primary reconstruction target once per epoch;
within-condition chunks are balanced at <=8 cells. Teacher/observed post bags
and control bags are independent size8 draws, without replacement when the pool
is large enough and with replacement only for a smaller pool. This is unpaired
population learning, not fabricated single-cell pairing. Condition bags are
shuffled deterministically each epoch. Held-out post rows cannot enter these
training samplers. Evaluation uses frozen ordered 300-control draws per condition.

Fine-tuning gene candidates include real zero-expression positions. Uniform
capping of the measured canonical axis (max2048), followed by pretrained-ID
sorting, is a declared stage-2 sampling choice; observed-view augmentation masks
20%, teacher views remain clean. No second normalization or binning is applied.
Both delivered CellFM axes fit below this cap. The native data interface now
validates frozen H5AD/observation/split/mapping/vocabulary hashes and identities,
loads only X/obs/var (not uns/DE), retains host CSR storage and densifies only
requested bags. Actual Adamson/Norman data IO passed for every train condition
and all validation/test control/truth populations. This is data integration,
not a completed ten-epoch LoRA run or numerical model result.

`datasets/knowledge.py` requires TextBase coverage on the requested complete
gene axis. GO/Protein/Pathway/HPA require source-only corpus receipts and exact
embedding text/model/dimension fingerprints; cumulative corpus provenance is
rejected. A local is omitted if any combination target lacks that source. Empty
sources need no API call/artifact and are never converted to zero vectors.

`cell/transfer.py` initializes from the selected **pretraining Student**, not
the EMA Teacher. It pins the checkpoint and full-run completion receipt, checks
exact vocabulary identity, and refuses incomplete formal pretraining or fixture
weights. A best checkpoint from an earlier complete epoch is valid only after
the entire requested pretraining run has finished. Downstream optimizer/RNG and
Teacher center are new task state; they are not resumed from stage 1.
The ten-epoch LoRA optimization/checkpoint loop and shared evaluator are now
implemented in `cell/finetune.py`. CPU fixtures pass full 2→10 training transfer,
exact interrupted resume and frozen-backbone checks; this is not formal GPU
training. See [the frozen campaign protocol](CELLFM_LORA_PROTOCOL.md) for optimizer,
selection, GPU isolation and evaluator-only real-data state.

## 6. Execution, leakage and unresolved gates

See [engineering/evaluation contract](ENGINEERING_EVALUATION_DESIGN.md) for the
native implementation, dual-5090 plan, instrumentation, storage and exact metrics.

Resolve splits first. No held-out post-expression in pretraining, teacher,
HVG fitting or prototypes. Evaluation-only DE and the prescribed Systema
train+validation reference must never enter model inputs/training. Audit
pretraining overlap and source dates/evidence; retain shuffled/availability
knowledge controls.

Before implementation/runs, resolve: pretraining corpus/exclusions; frozen
Kimi implementation and reduced head/rank mapping; selected residual/FFN
default; decoder; fixed fine-tuning schedule; measured dual-GPU capacity.
Missing source behavior must be labelled, not guessed.

## 7. Sources

- [CellFM pinned source](https://github.com/biomed-AI/CellFM/tree/bfed59c0e34103231165d69b97927ecc888d623c): config.py, data_process.py, model.py, retention.py, train.py, utils.py.
- [Kimi K3](https://github.com/MoonshotAI/Kimi-K3), [report](https://arxiv.org/html/2607.24653v1); pin implementation before coding.
- [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals).
- [DINOv2 KoLeo](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py), [training](https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/ssl_meta_arch.py); freeze commit before coding.
- Local dino cell.pdf, DOI 10.64898/2025.12.16.694747: conditional/observed view concepts; unavailable code must not be invented.
