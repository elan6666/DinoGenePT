# DinoGenePT ablation matrix

All rows below use the same `dinogenept` model registry entry. They differ only
through composed YAML configuration. The smoke matrix is
`configs/experiments/dinogenept/smoke-matrix.yaml`.

| Config | Isolated question |
|---|---|
| `00-supervised` | Does supervised condition-mean prediction run without self-distillation? |
| `01-dino-base` | What does condition-level Base DINO add? |
| `02-dino-ibot` | What does delta-iBOT add without optional knowledge? |
| `03-local-go` | Does source-only GO local training help? |
| `04-local-protein` | Does source-only Protein local training help? |
| `05-local-pathway` | Does source-only Pathway local training help? |
| `06-local-hpa` | Does source-only HPA local training help? |
| `07-dynamic-locals` | Do sparse, variable, source-balanced locals help together? |
| `08-source-shuffled` | Is any gain semantic rather than source availability? |
| `09-availability-only` | Can annotation density alone explain a gain? |
| `10-ibot-shuffled` | Can delta-iBOT exploit a gene-identity shortcut? |
| `11-direction-aware` | Does an explicit direction penalty help? |
| `12-attention-residual` | Does block-level residual retrieval help? |
| `13-dense-adapter` | Does a parameter-matched dense residual adapter help? |
| `14-moe-aux` | Does sparse MoE with auxiliary balancing help? |
| `15-moe-quantile` | Does auxiliary-loss-free quantile balancing help? |
| `16-moe-situ` | Does SiTU-style gating improve the MoE expert? |
| `17-backbone-situ` | Does SiTU-style gating improve backbone feed-forwards? |
| `18-koleo` | Does optional representation spreading help without collapse? |
| `19-hybrid-kda` | Does ordered KDA plus periodic global attention help? |
| `20-ibot-post-topk` | How much shortcut signal appears if masked genes are selected from the unmasked post cell instead of the fixed training-variance set? |

## Fair comparison contract

A formal matrix must hold fixed:

- dataset fingerprint and frozen perturbation-condition split;
- selected gene axis and all preprocessing;
- random seed, condition bags, and inference control count;
- Base prior artifact and embedding model;
- evaluator, condition-macro aggregation, and bootstrap settings;
- training budget except when the budget itself is the named ablation.

Single-source GO/Protein/Pathway/HPA rows retain all source adapters and differ
only in which local view is scheduled. The matrix writes candidate-minus-
reference metrics and paired condition bootstrap intervals against
`supervised-only`. Architecture rows that change total parameters or active
compute still require separate parameter- or FLOP-matched controls before a
scientific comparison.

Report MSE, normalized MSE, Pearson on expression and deltas, direction
accuracy, top-DE metrics when available, source-availability strata, peak GPU
memory, and uncertainty over repeated seeds. Compare the main row against both
source-shuffled and availability-only controls before attributing a gain to
knowledge semantics.

## Smoke receipt

The server run completed all 21 rows with one epoch and no parallel GPU jobs.
Its purpose was branch coverage, BF16/backward validation, output-layout
validation, and resource auditing. It used synthetic prior vectors and a tiny
gene/condition subset; numerical rankings are deliberately not interpreted.

Two defects were found and fixed during the matrix: BF16/FP32 MoE dispatch
mixing and an in-place KoLeo diagonal mask. Dedicated regression tests cover
both. A critic pass then tightened dataset/split/prior/checkpoint/source hashes,
atomic completion receipts, GPU0 enforcement, plugin boundaries, local-loss
weighting, MoE/KoLeo diagnostics, fixed training-variance iBOT selection, and
single-source parameter fairness. All 21 rows saved hash-verified checkpoints;
a second invocation reused 21/21 completed receipts without training.
