# Native default LoRA campaign

Historical campaign record below. Current fine-tuning loss/pairing defaults are
defined in [OBSERVED_FINETUNING_LOSSES.md](OBSERVED_FINETUNING_LOSSES.md), which
supersedes the fixed 0.1 weights, independent overlapping bags and unconditional
observed masking described below. Runtime budget/LR are defined by current code
and resolved configs, not this old campaign record.

Implementation and CPU integration tests are complete; formal CUDA runs are not.
This is our declared adaptation, not the unavailable DINOcell training code or
CellFM's 15-epoch GEARS notebook. Active budget remains two pretraining epochs,
then ten complete LoRA epochs for each released CellFM Adamson/Norman dataset.

## Training defaults

- Main anchor: TextBase; source-only GO/Protein/Pathway/HPA optional locals.
- LoRA rank16, alpha32, dropout0.05, exact audited mixer targets. Frozen backbone
  includes gene/value embeddings, norms and gates. New heads/adapters and the
  initialized pretraining cell projection head train normally.
- AdamW, lr1e-4, betas(0.9,0.999), weight decay0.01, gradient clip1.0.
  Linear warmup over 10% of optimizer steps, then cosine to 1% of peak lr.
  These optimizer choices are ours, not a claimed CellFM LoRA default.
- Primary condition/context bags <=8 cells, other bags size8, four-bag gradient
  accumulation. Weight uneven bags by primary cell count; visit every primary
  train-post cell once per epoch. No early stopping before the required ten epochs.
- Fixed loss: reconstruction +0.1(primary DINO + available-local mean DINO
  + observed DINO). No loss ablation. Teacher EMA updates once per optimizer step.
- Continuous published expression retained, all measured genes including zeros
  eligible, uniform cap2048. Both reduced axes fit in full. Observed view masks
  20%; clean Teacher/control views. No new bins or normalization.
- Validation-only `txpert_macro_pearson_delta` selects best by strict improvement;
  ties keep earlier best. Test is evaluated only after all ten epochs, using best.

## GPU isolation and runtime

Pretraining uses two-rank DDP. Fine-tuning is one process per dataset, so two
available cards can host independent Adamson/Norman jobs. `cuda_index` selects an
entry in an existing CUDA_VISIBLE_DEVICES list, otherwise an nvidia-smi physical
index. Before CUDA initialization the runner resolves the exact UUID, refuses
unrelated processes on that card and restricts itself to that UUID. Each job
requires a fresh process. No implicit cross-GPU contexts for RNG serialization:
save/restore only the current device's state. CPU fixtures do not establish fit.

`dinogenept finetune --config <resolved.json> [--resume <last.pt>]` consumes
`data` (CellFMDataset pinned arguments), `pretrained` (checkpoint and completion
paths/hashes), `knowledge` (verified bundle's knowledge object), `evaluation`
(frozen path/hash), `perturbation` and `training` options. Full artifact-bound
configs can be resolved once the real pretraining checkpoint and vector bundle
exist; the runner does not replace missing artifacts with random weights.

Output: resolved_config.json, input_identity.json, metrics.jsonl, last.pt,
best.pt, validation-epochNN.json, test.json and completion.json. No giant PKL
predictions. The evaluator retains at most one condition's [300,G] predictions.
Loss components/weights, tokens/cells, LR, gradient norm, throughput, data/step
time, peak allocated VRAM and parameter counts are recorded. Invocation markers
distinguish resumed work; partial steps after the durable checkpoint may replay.

## Evaluator method and frozen real files

Read-only reference: GraD-Pert `276d7baac57c6a4130b5e43563cc7eccd8c8ff7e`,
evaluation/state.py, evaluation/metrics.py and data/controls.py. No runtime import.

Control policy: seed20260824, SHA256 condition/dataset/split-derived 128-bit PCG64,
sample truth-cell contexts with replacement, then sample their compatible
controls with replacement, preserving exactly 300 ordered rows. Earlier IO tests'
seed42 helper draws are **not** used by this formal evaluator.

DE: Scanpy1.12.4 t-test, rankby_abs=True, full delivered gene axis. Apply upstream
non-dropout eligibility, take first20, then exclude targets without refilling.
Systema uses the equal-condition-weight train+validation noncontrol centroid.
These references/DE/truth never become training features. The three distinct
Pearson metrics use the shared native evaluator and retain undefined reasons.

Server state at `data/perturbation/cellfm-evaluation-v1/`:

| File | Conditions | DE gene range | SHA256 |
|---|---:|---:|---|
| adamson.json | 26 | 19–20 | e2cacc19a73db3f3eff56f9934c8ed446fa7fa9fc4524747ae596d9072eb15c5 |
| norman.json | 121 | 18–20 | 2dd8f80b17bbd7a3290ffcd9eee5f5efc2fe25c94997231840e2f410e9314804 |

These are the released **reduced-axis CellFM data/simulation splits**, not the
full GraD-Pert canonical datasets. Norman's lane/gemgroup metadata do not define
new batches here; both datasets use the released cell-type context. GraD-Pert's
Norman registry likewise declares an upstream single-batch policy, but data
identity and preprocessing remain different. Do not compare numeric scores as
if canonical rows/splits/genes were identical.
