# Shared update policy and independent 50k/Jurkat vocabulary

## Update policy (2026-09-09)

- Pretraining, fine-tuning and capacity probes now use one native optimizer
  grouping function. Trainable `nn.Embedding` weights receive weight_decay0;
  other trainable parameters retain weight_decay0.01. The gene table is not
  frozen during pretraining. Previously unseen rows stay exactly unchanged
  when other genes learn, verified by a three-update regression test.
- This follows CellFM's exclusion of embedding parameters from decay, but is
  not a copy of its whole optimizer policy: our grouping selects module type,
  and does not additionally change all bias/norm decay policies. Adam momentum
  can still change a previously observed but currently absent gene. We do not
  claim a row-wise optimizer freeze or clear historical optimizer state.
- FineTuneOptions defaults: AdamW betas(0.9,0.95), base LR2e-4, DINOv2
  square-root scaling at1024,16% step warmup from0, cosine decay to1e-6,
  no cycle restarts. Legacy explicitly selected schedules remain available.
- Stage2 currently trains one process per dataset. Its nominal scaling batch
  is bag_size *accumulation (default8*4=32 primary cells), not number of views,
  teacher/post replicas or knowledge sources. As in pretraining, a partial
  final window does not change the nominal scaled LR. With global108 the
  pretraining peak is6.495190528e-5; default fine-tuning32 gives3.535533906e-5.
  This aligns the mechanism, not an identical numeric LR across unequal batches.
- Both stages call the same teacher_momentum function: zero-based iteration,
 0.994 cosine toward1, once after a successful optimizer update. Independent
  centering/statistics and stage-specific objectives remain unchanged.
- New checkpoints carry update_policy=dinov2-shared-ema-gene-no-decay-v1.
  Old optimizer resumes fail before model mutation. Deliberate weights-only
  transfer remains separate; old checkpoints/data/recipes are not rewritten.
- Fine-tuning still freezes the original gene table via LoRA. This request
  does not enable learning new embedding rows. Frozen external knowledge
  vectors are not modified. No training or monitoring was launched.

References: CellFM@72c9f4a9580a3716058c184900ed14a65151ed8f/utils.py
set_weight_decay; our existing DINOv2 LR/EMA source audit and schedule.py.
`configs/cell/finetune_update_policy.json` contains valid training-option
overrides for new resolved configurations. Historical sealed configs are not
silently migrated.

## Independent vocabulary bundle

Server directory: `/data/yilangliu/DinoGenePT/data/vocabulary/genecompass50k-jurkat-v1`.
Old active vocabulary/data/checkpoints remain untouched. Source GraD-Pert files
are read-only. No Jurkat expression values, outcomes or evaluation predictions
are read. Split metadata is the frozen canonical1335/445/592 condition split.

| Input/output | Count |
|---|---:|
| Previous human vocabulary |23113|
| Actually present in published50k arrays (before random crops) |19827|
| Jurkat complete graph axis |6506|
| Jurkat expression/prediction axis |5000|
| Jurkat targets across train/val/test, excluding ctrl |2372|
| Joint identity-key vocabulary |20783|
| Embedding rows including padding0 |20784|
| Joint keys absent from observed50k identity set |956|

Files:50k_observed_vocabulary.json (original observed Ensembl IDs),
vocabulary.json (joint identity keys),old_to_new.json (PAD0,unobserved old IDs=-1),
source_token_ids.json (order-preserving axes and targets),token_sources.json,
identity_audit.json and receipt.json. Counts refer to published50k retained
tokens, not all genes in original untruncated matrices or actual random-crop
coverage. Every retained50k input token maps, with zero cells acquiring duplicate
tokens under the mapping. No expression values were merged or transformed.

Resolved entries use frozen HGNC IDs; unresolved Ensembl IDs retain exact stable
IDs, and unresolved Jurkat names receive an explicit source namespace. This
preserves all input labels without inventing equivalence. The956 added keys
are not956 verified novel biological genes: unresolved identities could overlap.

### Explicit activation blockers

- 50k:18026 resolved,1800 HGNC-unverified Ensembl IDs,1 ambiguous.
- Jurkat graph:6410 resolved,94 unresolved,2 ambiguous.
- Jurkat expression axis:4906 resolved,94 unresolved.
- Jurkat targets:2371 resolved;QARS is ambiguous (HGNC:3418 / HGNC:9751).
- C20orf197 and MIR646HG are two distinct existing axis columns mapping to
  HGNC:27659. Both columns and their positions are preserved; no sum/drop/mean.

The bundle is generated and provenance-audited, but is **not ready to replace
the active training data**. Identity ambiguity/duplicate-column policy must be
resolved, then independently remap datasets and update vocabulary-bound model
configs. Current downstream loaders also need explicit HGNC-key mapping support;
the old Ensembl-only mapping interface must not be bypassed. A new table alone
does not train the956 added keys, and existing checkpoints cannot simply be
loaded by position into it. Downstream test gene identities were used to define
coverage (transductive vocabulary knowledge), not held-out expression labels;
do not claim an entirely unknown-gene vocabulary evaluation.

Compact source/output hashes: [receipt](results/JOINT_VOCABULARY_20260909.json).
Builder refuses overwrites, validates data/HGNC hashes and the canonical split,
and checks input metadata hashes again after the build.

Validation:46 focused model/schedule/runner tests passed, including CPU-only
fine-tuning and exact resume. A second suite of25 identity/bundle/optimizer
tests passed (two optimizer tests overlap). Scoped Ruff and diff checks passed.
No GPU training or benchmark result is claimed.
